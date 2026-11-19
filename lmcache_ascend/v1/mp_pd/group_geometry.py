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
* MLA/DSA plane entries (DeepSeek V2/V3/V3.2/V4 on vLLM-Ascend) -- a
  tuple of ``[num_blocks, block_size, 1, width_i]`` planes whose widths
  or dtypes differ (latent, rope, dsa index, V4 scale) -- group by the
  flattened hidden width; the per-chunk storage shape is ONE flat
  plane ``(layers, 1, chunk_size, 1, summed_width)`` (``kv_size == 1``,
  the transfer kernels carve the planes back out per layer);
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
    layer_names: tuple[str, ...]
    shape: torch.Size
    dtype: torch.dtype
    recurrent: bool = False
    mla: bool = False
    page_size_bytes: int = 0

    @property
    def num_layers(self) -> int:
        return len(self.layer_indices)


def flat_state_page_view(entry, page_bytes: int) -> torch.Tensor:
    """Flat ``[num_blocks, page_bytes]`` uint8 view of a state pool.

    Accepts either form vLLM registers for a recurrent layer: the raw
    ``[num_blocks, 1, 1, C]`` int8 page view, or a ``[state, ...]``
    view list whose first state anchors the shared padded page (see
    :func:`_state_list_page_bytes`). The returned view aliases the
    pool storage -- no bytes are copied.

    Raises:
        ValueError: The entry matches no known form, or the shared
            page contract is violated (would corrupt byte transfers).
    """
    if isinstance(entry, torch.Tensor):
        if (
            entry.ndim != 4
            or entry.shape[1] != 1
            or entry.shape[2] != 1
            or entry.shape[3] != page_bytes
        ):
            raise ValueError(
                f"raw page view {tuple(entry.shape)} does not match a "
                f"{page_bytes}-byte state page"
            )
        return entry.reshape(entry.shape[0], page_bytes)
    if isinstance(entry, (tuple, list)) and len(entry) >= 2:
        first = entry[0]
        if not isinstance(first, torch.Tensor):
            raise ValueError(
                f"state entry holds a non-tensor "
                f"{type(first).__name__}"
            )
        if (
            first.storage_offset() != 0
            or first.stride(0) * first.element_size() != page_bytes
        ):
            raise ValueError(
                "recurrent state page contract violated: the first "
                f"state does not anchor a {page_bytes}-byte page"
            )
        elems = page_bytes // first.element_size()
        view = first.as_strided((first.shape[0], elems), (first.stride(0), 1))
        return view.view(torch.uint8)
    raise ValueError(
        f"unsupported recurrent state pool entry {type(entry).__name__}"
    )


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


def _mla_plane_geometry(entry) -> tuple[int, torch.dtype] | None:
    """``(flat hidden width, dtype)`` for an MLA/DSA plane tuple.

    vLLM-Ascend registers DeepSeek MLA/DSA/V4 layers as a tuple of NP
    paged planes ``[num_blocks, block_size, 1, width_i]`` with differing
    widths -- and, on V4, differing dtypes (e.g. an int8 latent with an
    fp16 scale plane). The transferred object is ONE flat plane of the
    summed width in the first plane's dtype (upstream's
    ``NL_X_NP_X_NB_BS_ONE_HS`` contract; the transfer kernels carve the
    planes back out per layer). Planes may be views into one shared
    pool with non-zero storage offsets -- that is normal here, unlike
    on the recurrent state-page path.

    Standard ``(k, v)`` tuples with identical shapes and dtypes never
    reach this check (the attention branch catches them first); the
    same-shape guard below is a second line of defense.
    """
    if not isinstance(entry, (tuple, list)) or len(entry) < 1:
        return None
    for plane in entry:
        if not isinstance(plane, torch.Tensor):
            return None
        if plane.ndim != 4 or int(plane.shape[2]) != 1:
            return None
    first = entry[0]
    if any(int(p.shape[0]) != int(first.shape[0]) for p in entry):
        return None
    if any(int(p.shape[1]) != int(first.shape[1]) for p in entry):
        return None
    widths = tuple(int(p.shape[3]) for p in entry)
    dtypes = tuple(p.dtype for p in entry)
    if len(entry) == 2 and widths[0] == widths[1] and dtypes[0] == dtypes[1]:
        return None  # standard 1-head (k, v); the attention branch owns it
    total_bytes = sum(
        w * p.element_size() for w, p in zip(widths, entry, strict=True)
    )
    itemsize = first.element_size()
    if total_bytes % itemsize != 0:
        raise ValueError(
            f"MLA plane byte total {total_bytes} is not a multiple of "
            f"the primary plane itemsize {itemsize}; cannot flatten"
        )
    return total_bytes // itemsize, first.dtype


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
        One :class:`KVGroupGeometry` per detected group, ordered with
        attention groups first (each by first layer index) and
        recurrent groups after -- the combined-object protocol treats
        the FIRST shape as the primary attention group (token
        dimension, partial-chunk narrowing), and interleaved hybrid
        models (e.g. Qwen3.8) start with a linear layer. ``None``
        when any entry is unclassifiable or nothing is registered --
        the caller then falls back to the legacy model-config
        geometry.

    Raises:
        ValueError: A confirmed recurrent state list violates the
            shared-page contract (state not at page base, or a page
            that cannot hold the states).
    """
    attn_groups: "OrderedDict[tuple[int, int], list[int]]" = OrderedDict()
    attn_names: "OrderedDict[tuple[int, int], list[str]]" = OrderedDict()
    mla_groups: "OrderedDict[tuple[int, torch.dtype], list[int]]" = (
        OrderedDict()
    )
    mla_names: "OrderedDict[tuple[int, torch.dtype], list[str]]" = (
        OrderedDict()
    )
    rec_groups: "OrderedDict[int, list[int]]" = OrderedDict()
    rec_names: "OrderedDict[int, list[str]]" = OrderedDict()

    for idx, (name, entry) in enumerate(kv_caches.items()):
        if entry is None:
            continue
        page = _raw_page_view_bytes(entry)
        if page is not None:
            rec_groups.setdefault(page, []).append(idx)
            rec_names.setdefault(page, []).append(name)
            continue
        geometry = _attention_geometry(entry)
        if geometry is not None:
            attn_groups.setdefault(geometry, []).append(idx)
            attn_names.setdefault(geometry, []).append(name)
            continue
        mla = _mla_plane_geometry(entry)
        if mla is not None:
            hidden, mla_dtype = mla
            mla_groups.setdefault((hidden, mla_dtype), []).append(idx)
            mla_names.setdefault((hidden, mla_dtype), []).append(name)
            continue
        page = _state_list_page_bytes(entry, name)
        if page is not None:
            rec_groups.setdefault(page, []).append(idx)
            rec_names.setdefault(page, []).append(name)
            continue
        # Unclassifiable entry (e.g. MLA pairs): the caller falls
        # back to the legacy model-config geometry.
        return None

    if not attn_groups and not mla_groups and not rec_groups:
        return None

    geometries: list[KVGroupGeometry] = [
        KVGroupGeometry(
            layer_indices=tuple(indices),
            layer_names=tuple(attn_names[(heads, head_dim)]),
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
            layer_names=tuple(mla_names[(hidden, mla_dtype)]),
            # One flat plane of the summed width (kv_size == 1);
            # the transfer kernels carve the per-plane layout back
            # out of this contiguous hidden dimension.
            shape=torch.Size((len(indices), 1, chunk_size, 1, hidden)),
            dtype=mla_dtype,
            mla=True,
        )
        for (hidden, mla_dtype), indices in mla_groups.items()
    )
    geometries.extend(
        KVGroupGeometry(
            layer_indices=tuple(indices),
            layer_names=tuple(rec_names[page]),
            shape=torch.Size((len(indices), page)),
            dtype=torch.uint8,
            recurrent=True,
            page_size_bytes=page,
        )
        for page, indices in rec_groups.items()
    )
    geometries.sort(
        key=lambda geometry: (geometry.recurrent, geometry.layer_indices[0])
    )
    return geometries
