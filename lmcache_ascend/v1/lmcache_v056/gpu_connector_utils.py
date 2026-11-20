# Frozen copy of LMCache v0.5.6 `lmcache/v1/gpu_connector/utils.py`,
# trimmed to the functions consumed by the trimmed gpu_connectors.py
# (block/head/page geometry accessors and format discovery for the paged
# V2 connector). The remaining facade functions served the removed
# in-process connector classes and are not kept.
# Import-path adjustments: kv_format -> dev common.kv_format.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
# Standard
from typing import Optional

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.common.kv_format import detect_format, get_spec
from lmcache.v1.common.kv_format.types import DiscoverableKVCache, LayoutHints
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)


def normalize_kv_and_discover_format(
    kv_caches: DiscoverableKVCache,
    serving_engine: EngineType,
    layout_hints: "LayoutHints | None" = None,
) -> tuple["lmcache_native.EngineKVFormat", DiscoverableKVCache]:
    """Normalize ``kv_caches`` into canonical form and discover its Engine KV format.

    Thin wrapper over
    :func:`lmcache.v1.gpu_connector.kv_format.detect_format`; see that
    function for the full contract.

    Args:
        kv_caches: The KV cache tensors (possibly nested lists of tensors).
        serving_engine: Which serving engine produced the caches.
        layout_hints: See :class:`LayoutHints`.

    Returns:
        ``(engine_kv_format, normalized_kv_caches)``. Callers must use the
        returned tensor structure for subsequent operations -- it shares
        storage with the input but may be a permuted view.
    """
    return detect_format(kv_caches, serving_engine, layout_hints)


def get_num_blocks(
    kv_caches: DiscoverableKVCache, engine_kv_format: "lmcache_native.EngineKVFormat"
) -> int:
    """Return the number of blocks from ``kv_caches``.

    Raises:
        ValueError: For NBBS-fused formats with no separate block axis.
    """
    return get_spec(kv_caches, engine_kv_format).num_blocks()


def get_block_size(
    kv_caches: DiscoverableKVCache,
    engine_kv_format: "lmcache_native.EngineKVFormat",
    layer_idx: int = 0,
) -> int:
    """Return the block size (tokens per block) for layer ``layer_idx``.

    ``layer_idx`` is honoured only for per-layer formats where BS may
    differ across layers (e.g. mixed-compression MLA pools). For
    cross-layer formats BS is shared across layers and ``layer_idx``
    is ignored.

    Raises:
        ValueError: For NBBS-fused formats with no separate block axis.
    """
    return get_spec(kv_caches, engine_kv_format).block_size(layer_idx)


def get_head_size(
    kv_caches: DiscoverableKVCache,
    engine_kv_format: "lmcache_native.EngineKVFormat",
    layer_idx: int = 0,
) -> int:
    """Return the head size for a layer (defaults to layer 0)."""
    return get_spec(kv_caches, engine_kv_format).head_size(layer_idx)


_BLOCK_AXIS_FORMATS: frozenset = frozenset(
    {
        lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
        lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS,
        lmcache_native.EngineKVFormat.NL_X_NP_X_NB_BS_ONE_HS,
        # Under vLLM's blocks-first layouts (BLHNC / BLNHC) these views
        # stride(0) spans every layer's bytes for the block; when the cache
        # is layer-compact, stride(0) is simply the tight per-block step.
        lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_CS,
        lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_CS,
        lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_HS,
    }
)


def resolve_block_stride_and_log_layout(
    kv_caches: DiscoverableKVCache,
    engine_kv_format: "lmcache_native.EngineKVFormat",
    layer_idx: int,
    group_idx: int,
) -> Optional[int]:
    """Resolve the per-block stride for a KV layer group and log its layout.

    Single entry point for :class:`KVLayerGroupsManager` to obtain the
    ``block_stride_elems`` value for :class:`PageBufferShapeDesc` and emit
    a one-shot layout audit line. All ``EngineKVFormat``-aware reasoning is
    kept here so callers never touch a "representative KV cache" tensor.

    * Block-axis formats (:data:`_BLOCK_AXIS_FORMATS`): ``stride(0)`` is
      the per-block step and is returned as-is. A value larger than the
      tight stride indicates dim-0 padding (e.g. DeepSeek V4 compressor
      caches sharing a KV pool with larger attn groups).
    * Other formats: dim-0 is not the block axis, so ``None`` is
      returned and ``shape_desc`` falls back to the tight stride. Any
      dim-0 padding in such formats is rejected with ``ValueError``
      since downstream kernels cannot honour it.

    Args:
        kv_caches: Full KV cache structure (already normalised).
        engine_kv_format: Format of ``kv_caches``.
        layer_idx: 0-based layer index used as the layout probe.
        group_idx: 0-based group index, used only for logging.

    Returns:
        ``stride(0)`` for block-axis formats; ``None`` otherwise.

    Raises:
        ValueError: Non-block-axis format carries dim-0 padding.
    """

    def _pick_layout_probe_tensor() -> torch.Tensor:
        # Layout probe only (shape/stride/storage_offset/dtype); not
        # the K/V slice fed to the transfer kernel.
        # - Cross-layer formats: ``kv_caches`` is the single backing
        #   tensor packing all layers along dim-1; indexing dim-0
        #   (= NB) would yield a per-block slice, so return the whole
        #   tensor (its ``stride(0)`` is the authoritative per-NB step).
        # - SGL MHA: outer list is K/V (length 2), inner list is
        #   per-layer; K & V share shape/stride by construction.
        # - Other formats: ``kv_caches`` is already a per-layer list.
        if lmcache_native.is_cross_layer(engine_kv_format):
            if not isinstance(kv_caches, torch.Tensor):
                raise TypeError(
                    "Cross-layer EngineKVFormat expects a single backing "
                    f"torch.Tensor, got {type(kv_caches).__name__}."
                )
            return kv_caches
        if lmcache_native.is_kv_list(engine_kv_format):
            return kv_caches[0][layer_idx]  # type: ignore[index,return-value]
        if lmcache_native.is_kv_second_tuple(engine_kv_format):
            return kv_caches[layer_idx][0]  # type: ignore[index,return-value]
        return kv_caches[layer_idx]  # type: ignore[index,return-value]

    rep = _pick_layout_probe_tensor()

    block_stride_elems: Optional[int]
    if engine_kv_format in _BLOCK_AXIS_FORMATS and rep.ndim > 0:
        block_stride_elems = int(rep.stride(0))
    else:
        # Non-block-axis format: detect forbidden dim-0 padding.
        if rep.ndim >= 2:
            tight_dim0 = 1
            for d in range(1, rep.ndim):
                tight_dim0 *= int(rep.shape[d])
            padding = int(rep.stride(0)) - tight_dim0
            if padding > 0:
                raise ValueError(
                    "resolve_block_stride_and_log_layout: group's probe "
                    f"tensor has dim-0 padding ({padding} elements per "
                    f"block) but engine_kv_format={engine_kv_format!r} is not "
                    "a supported dim-0-padded format (only "
                    "NL_X_NB_BS_HS is); downstream transfer kernels "
                    "cannot honour this padding and would read/write "
                    "wrong bytes. "
                    f"layer_idx={layer_idx}, shape={tuple(rep.shape)}, "
                    f"stride={tuple(rep.stride())}, "
                    f"tight_stride0={tight_dim0}, "
                    f"storage_offset={int(rep.storage_offset())}, "
                    f"dtype={rep.dtype}."
                )
        block_stride_elems = None

    # Best-effort layout audit log; the log line itself must not raise.
    shape = tuple(rep.shape)
    stride = tuple(rep.stride())
    try:
        inner = 1
        for s in shape[1:]:
            inner *= int(s)
        padding_per_block = stride[0] - inner if stride else 0
    except Exception:
        padding_per_block = -1
    try:
        storage_nbytes = rep.untyped_storage().nbytes()
    except Exception:
        storage_nbytes = -1
    logger.info(
        "Group %d first-layer tensor: layer_idx=%d shape=%s "
        "stride=%s is_contiguous=%s dtype=%s device=%s "
        "storage_offset=%d numel=%d storage_nbytes=%d "
        "padding_per_block=%d",
        group_idx,
        layer_idx,
        shape,
        stride,
        rep.is_contiguous(),
        rep.dtype,
        rep.device,
        rep.storage_offset(),
        rep.numel(),
        storage_nbytes,
        padding_per_block,
    )

    return block_stride_elems
