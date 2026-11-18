# SPDX-License-Identifier: Apache-2.0
"""Per-engine-group KV geometry for hybrid (multi-group) PD transfer.

The PD leg historically assumed one uniform KV geometry for the whole
model: ``(num_layers, 2, chunk_size, num_kv_heads, head_size)`` derived
from the model config. Hybrid models (e.g. Qwen3.8-27B: full-attention
GQA layers interleaved with Gated-DeltaNet layers) break that
assumption: each engine KV group has its own shape and dtype, and
recurrent-state groups page a per-boundary state snapshot instead of
per-token attention KV.

This module derives one :class:`KVGroupGeometry` per engine KV group
from the tensors vLLM hands to ``register_kv_caches`` -- no model-config
guessing, no runtime probing:

* attention entries -- a per-layer ``(k, v)`` tensor tuple, a merged
  ``[2, num_blocks, block_size, H, D]`` tensor, or a single
  ``[num_blocks, block_size, H, D]`` cache -- group by ``(H, D)``; the
  per-chunk storage shape is ``(layers, 2, chunk_size, H, D)``;
* recurrent entries -- a ``[conv_state, ssm_state, ...]`` state-view
  list sharing one padded page per block ``(conv | ssm | pad)``, or a
  raw ``[num_blocks, 1, 1, C]`` int8 page view -- become recurrent
  groups whose per-page storage shape is ``(layers, C)`` uint8. The
  bytes are opaque: content-aware processing (rotary blending, hashing
  over K/V values) must never touch recurrent pages.

Single-group attention models collapse to a one-element list -- the
N=1 special case that keeps single-geometry models bit-for-bit
unchanged. The builder never guesses: any entry it cannot classify
(e.g. MLA ``(k, v)`` pairs, whose halves differ in shape and live in
separate storages) makes the whole build return ``None`` so the caller
falls back to the legacy model-config geometry. Only a *confirmed*
state list that violates the shared-page contract raises, because a
byte-level transfer over such a layout would land misaligned.
"""

# Standard
from collections import OrderedDict
from dataclasses import dataclass
import math

# Third Party
import torch


@dataclass(frozen=True)
class KVGroupGeometry:
    """Storage geometry of one engine KV group on the PD leg.

    Attention groups store one object per chunk with the legacy
    ``[layers, 2, chunk_size, num_kv_heads, head_size]`` layout.
    Recurrent groups store one object per state page with an opaque
    ``[layers, page_size_bytes]`` uint8 layout.
    """

    layer_indices: tuple[int, ...]
    shape: torch.Size
    dtype: torch.dtype
    recurrent: bool = False
    page_size_bytes: int = 0

    @property
    def num_layers(self) -> int:
        return len(self.layer_indices)


def _raw_page_view_bytes(entry) -> int | None:
    """Byte width of a raw ``[num_blocks, 1, 1, C]`` int8 page view.

    Newer vLLM versions register the mamba layer's whole padded page as
    one int8 view instead of a state-view list. An attention cache can
    never collide with this signature: paged attention caches put the
    block size (>= 16) in dim 1, never a 1 x 1 pair.
    """
    if (
        isinstance(entry, torch.Tensor)
        and entry.ndim == 4
        and entry.shape[1] == 1
        and entry.shape[2] == 1
        and entry.dtype in (torch.int8, torch.uint8)
    ):
        return int(entry.shape[3])
    return None


def _attention_geometry(entry) -> tuple[int, int] | None:
    """``(num_kv_heads, head_size)`` for per-layer attention entries.

    Accepts the three per-layer forms the Ascend worker registers:
    ``(k, v)`` tensor pairs (same shape and dtype), merged
    ``[2, num_blocks, block_size, H, D]`` tensors, and single
    ``[num_blocks, block_size, H, D]`` caches. Returns ``None`` for
    everything else (recurrent state lists included).
    """
    if isinstance(entry, torch.Tensor):
        if entry.ndim == 5 and entry.shape[0] == 2:
            return int(entry.shape[-2]), int(entry.shape[-1])
        if entry.ndim == 4:
            return int(entry.shape[-2]), int(entry.shape[-1])
        return None
    if isinstance(entry, (tuple, list)) and len(entry) == 2:
        k, v = entry
        if (
            isinstance(k, torch.Tensor)
            and isinstance(v, torch.Tensor)
            and k.shape == v.shape
            and k.dtype == v.dtype
        ):
            return int(k.shape[-2]), int(k.shape[-1])
    return None


def _state_list_page_bytes(entry, name) -> int | None:
    """Padded page size derived from a ``[state, ...]`` view list.

    vLLM lays mamba states out as one padded page per block
    ``(conv | ssm | pad)``: every state is a view into the same
    backing storage, the first one sits at the page base, and its
    block stride spans the whole page -- the contract upstream
    LMCache's mamba-page-view asserts. Entries whose tensors live in
    separate storages (e.g. MLA ``(k, v)`` pairs) are not state
    lists; return ``None`` so the caller treats them as unknown.

    A confirmed state list that violates the page contract raises:
    a byte-level transfer over it would land misaligned.
    """
    if not isinstance(entry, (tuple, list)) or len(entry) < 2:
        return None
    for state in entry:
        if not isinstance(state, torch.Tensor):
            raise ValueError(
                f"layer {name}: recurrent state entry holds a "
                f"non-tensor {type(state).__name__}"
            )
    storages = {state.untyped_storage().data_ptr() for state in entry}
    if len(storages) != 1:
        return None
    first = entry[0]
    if first.storage_offset() != 0:
        raise ValueError(
            f"layer {name}: recurrent state does not view the page "
            f"base (storage_offset={first.storage_offset()}); byte-"
            f"level page transfer would land misaligned"
        )
    page = first.stride(0) * first.element_size()
    content = sum(
        math.prod(state.shape[1:]) * state.element_size() for state in entry
    )
    if page < content:
        raise ValueError(
            f"layer {name}: page {page} bytes cannot hold the states "
            f"({content} bytes); unexpected recurrent layout"
        )
    return page


def build_group_geometries(
    kv_caches: dict,
    kv_dtype: torch.dtype,
    chunk_size: int,
) -> list[KVGroupGeometry] | None:
    """Partition registered KV caches into per-group storage geometries.

    Layers are bucketed by attention geometry ``(H, D)`` and by
    recurrent page width; groups come back ordered by each group's
    first layer index, so the result is deterministic.

    Args:
        kv_caches: The ``{layer_name: cache}`` dict vLLM registers.
        kv_dtype: Config-resolved cache dtype for attention storage
            objects (matches the legacy single-group metadata).
        chunk_size: PD chunk size in tokens (attention object width).

    Returns:
        One :class:`KVGroupGeometry` per detected group, or ``None``
        when any entry is unclassifiable or nothing is registered --
        the caller then falls back to the legacy model-config
        geometry.

    Raises:
        ValueError: A confirmed recurrent state list violates the
            shared-page contract (state not at page base, or a page
            that cannot hold the states).
    """
    attn_groups: "OrderedDict[tuple[int, int], list[int]]" = OrderedDict()
    rec_groups: "OrderedDict[int, list[int]]" = OrderedDict()

    for idx, (name, entry) in enumerate(kv_caches.items()):
        if entry is None:
            continue
        page = _raw_page_view_bytes(entry)
        if page is not None:
            rec_groups.setdefault(page, []).append(idx)
            continue
        geometry = _attention_geometry(entry)
        if geometry is not None:
            attn_groups.setdefault(geometry, []).append(idx)
            continue
        page = _state_list_page_bytes(entry, name)
        if page is not None:
            rec_groups.setdefault(page, []).append(idx)
            continue
        # Unclassifiable entry (e.g. MLA pairs): the caller falls
        # back to the legacy model-config geometry.
        return None

    if not attn_groups and not rec_groups:
        return None

    geometries: list[KVGroupGeometry] = [
        KVGroupGeometry(
            layer_indices=tuple(indices),
            shape=torch.Size(
                (len(indices), 2, chunk_size, heads, head_dim)
            ),
            dtype=kv_dtype,
        )
        for (heads, head_dim), indices in attn_groups.items()
    ]
    geometries.extend(
        KVGroupGeometry(
            layer_indices=tuple(indices),
            shape=torch.Size((len(indices), page)),
            dtype=torch.uint8,
            recurrent=True,
            page_size_bytes=page,
        )
        for page, indices in rec_groups.items()
    )
    geometries.sort(key=lambda geometry: geometry.layer_indices[0])
    return geometries
