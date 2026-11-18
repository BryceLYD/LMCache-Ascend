# SPDX-License-Identifier: Apache-2.0
"""Group-geometry derivation for hybrid PD transfer (CPU-only fakes)."""

# Third Party
import pytest
import torch

# First Party
from lmcache_ascend.v1.mp_pd.group_geometry import build_group_geometries

CHUNK = 256


def _attn_pair(blocks=2, block=16, heads=2, head_dim=8):
    shape = (blocks, block, heads, head_dim)
    return (
        torch.empty(shape, dtype=torch.bfloat16),
        torch.empty(shape, dtype=torch.bfloat16),
    )


def _state_list(pages=5, page_bytes=1280, conv_bytes=768):
    """[conv, ssm] views sharing one padded page per block."""
    raw = torch.empty((pages, page_bytes), dtype=torch.uint8)
    conv = (
        raw[:, :conv_bytes]
        .view(torch.bfloat16)
        .view(pages, 3, conv_bytes // 6)
    )
    ssm = (
        raw[:, conv_bytes:]
        .view(torch.float32)
        .view(pages, 2, (page_bytes - conv_bytes) // 8)
    )
    return [conv, ssm]


def test_single_attention_group_matches_legacy_shape():
    kv_caches = {f"model.layers.{i}": _attn_pair() for i in range(4)}
    geometries = build_group_geometries(kv_caches, torch.bfloat16, CHUNK)

    assert len(geometries) == 1
    group = geometries[0]
    assert not group.recurrent
    assert group.layer_indices == (0, 1, 2, 3)
    assert group.shape == torch.Size((4, 2, CHUNK, 2, 8))
    assert group.dtype is torch.bfloat16
    assert group.page_size_bytes == 0


def test_merged_and_single_tensor_attention_forms_group():
    merged = torch.empty((2, 4, 16, 2, 8), dtype=torch.bfloat16)
    single = torch.empty((4, 16, 2, 8), dtype=torch.bfloat16)
    kv_caches = {"layer.0": merged, "layer.1": single}

    geometries = build_group_geometries(kv_caches, torch.bfloat16, CHUNK)

    assert len(geometries) == 1
    assert geometries[0].layer_indices == (0, 1)
    assert geometries[0].shape == torch.Size((2, 2, CHUNK, 2, 8))


def test_hybrid_attention_plus_recurrent_state_list():
    kv_caches = {
        "layer.0": _attn_pair(),
        "layer.1": _state_list(),
        "layer.2": _attn_pair(),
    }

    geometries = build_group_geometries(kv_caches, torch.bfloat16, CHUNK)

    assert len(geometries) == 2
    attn, rec = geometries
    assert attn.layer_indices == (0, 2)
    assert attn.shape == torch.Size((2, 2, CHUNK, 2, 8))
    assert not attn.recurrent
    assert rec.recurrent
    assert rec.layer_indices == (1,)
    assert rec.page_size_bytes == 1280
    assert rec.shape == torch.Size((1, 1280))
    assert rec.dtype is torch.uint8


def test_raw_page_view_entry_is_recurrent():
    entry = torch.empty((5, 1, 1, 1000), dtype=torch.int8)
    geometries = build_group_geometries(
        {"layer.0": entry}, torch.bfloat16, CHUNK
    )

    assert len(geometries) == 1
    group = geometries[0]
    assert group.recurrent
    assert group.page_size_bytes == 1000
    assert group.shape == torch.Size((1, 1000))
    assert group.dtype is torch.uint8


def test_none_entry_is_skipped():
    kv_caches = {"layer.0": None, "layer.1": _attn_pair()}
    geometries = build_group_geometries(kv_caches, torch.bfloat16, CHUNK)

    assert len(geometries) == 1
    assert geometries[0].layer_indices == (1,)


def test_state_view_not_at_page_base_raises():
    pages, page_bytes = 5, 1280
    raw = torch.empty((pages + 1, page_bytes), dtype=torch.uint8)
    conv = torch.as_strided(
        raw, (pages, 3, 128), (page_bytes, 128, 1),
        storage_offset=page_bytes,
    )
    ssm = torch.as_strided(
        raw, (pages, 2, 64), (page_bytes, 64, 1),
        storage_offset=page_bytes + 768,
    )

    with pytest.raises(ValueError, match="does not view the page base"):
        build_group_geometries(
            {"layer.0": [conv, ssm]}, torch.bfloat16, CHUNK
        )


def test_page_smaller_than_states_raises():
    pages = 5
    raw = torch.empty((pages, 1280), dtype=torch.uint8)
    conv = torch.as_strided(raw, (pages, 3, 128), (384, 128, 1))
    ssm = torch.as_strided(
        raw, (pages, 2, 64), (1280, 64, 1), storage_offset=768
    )

    with pytest.raises(ValueError, match="cannot hold the states"):
        build_group_geometries(
            {"layer.0": [conv, ssm]}, torch.bfloat16, CHUNK
        )


def test_unrecognized_entry_returns_none():
    assert (
        build_group_geometries({"layer.0": object()}, torch.bfloat16, CHUNK)
        is None
    )


def test_empty_or_all_none_registry_returns_none():
    assert build_group_geometries({}, torch.bfloat16, CHUNK) is None
    kv_caches = {"layer.0": None, "layer.1": None}
    assert build_group_geometries(kv_caches, torch.bfloat16, CHUNK) is None


def test_mla_style_pair_returns_none():
    k = torch.empty((2, 16, 1, 512), dtype=torch.bfloat16)
    v = torch.empty((2, 16, 64), dtype=torch.bfloat16)

    assert build_group_geometries({"layer.0": (k, v)}, torch.bfloat16, CHUNK) is None
