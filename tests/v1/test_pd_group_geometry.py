# SPDX-License-Identifier: Apache-2.0
"""Group-geometry derivation for hybrid PD transfer (CPU-only fakes)."""

# Third Party
import pytest
import torch

# First Party
from lmcache_ascend.v1.mp_pd.group_geometry import (
    build_group_geometries,
    flat_state_page_view,
)

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


def test_linear_first_interleaving_keeps_attention_first():
    """Qwen3.8-style layering: the FIRST layer is linear, but the
    attention group must still come back first (the combined-object
    protocol reads the primary shape from index 0)."""
    kv_caches = {
        "layer.0": _state_list(),
        "layer.1": _attn_pair(),
        "layer.2": _state_list(),
    }

    geometries = build_group_geometries(kv_caches, torch.bfloat16, CHUNK)

    assert len(geometries) == 2
    attn, rec = geometries
    assert not attn.recurrent
    assert attn.layer_indices == (1,)
    assert rec.recurrent
    assert rec.layer_indices == (0, 2)


def test_none_entry_is_skipped():
    kv_caches = {"layer.0": None, "layer.1": _attn_pair()}
    geometries = build_group_geometries(kv_caches, torch.bfloat16, CHUNK)

    assert len(geometries) == 1
    assert geometries[0].layer_indices == (1,)


def test_state_pool_nonzero_offset_classifies_page_packed():
    """vLLM-Ascend layers share one pool: the conv state sits at a
    layer-dependent non-zero storage offset, but its per-block stride
    still spans the full padded page -- that is the page-packed form,
    and it must classify (upstream v0.5.6 wrongly demanded offset 0)."""
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

    geometries = build_group_geometries(
        {"layer.0": [conv, ssm]}, torch.bfloat16, CHUNK
    )

    assert len(geometries) == 1
    group = geometries[0]
    assert group.recurrent
    assert group.page_size_bytes == 1280
    assert group.shape == torch.Size((1, 1280))


def test_slice_planes_form_region_is_summed_states():
    """vLLM-Ascend <= v0.29 registers conv and SSM as separate dense
    contiguous slices: no shared page, region = summed state bytes."""
    conv = torch.empty((5, 3, 2560), dtype=torch.bfloat16)
    ssm = torch.empty((5, 2, 64), dtype=torch.float32)

    geometries = build_group_geometries(
        {"layer.0": [conv, ssm]}, torch.bfloat16, CHUNK
    )

    assert len(geometries) == 1
    group = geometries[0]
    assert group.recurrent
    # conv 15360B + ssm 512B, no page padding
    assert group.page_size_bytes == 15872
    assert group.shape == torch.Size((1, 15872))
    views = flat_state_page_view(
        (conv, ssm), group.page_size_bytes
    )
    assert [nbytes for _view, nbytes in views] == [15360, 512]
    assert views[0][0].shape == (5, 15360)
    assert views[1][0].shape == (5, 512)


def test_neither_page_packed_nor_contiguous_raises():
    pages = 5
    raw = torch.empty((pages, 1280), dtype=torch.uint8)
    conv = torch.as_strided(raw, (pages, 3, 128), (384, 128, 1))
    ssm = torch.empty((pages, 2, 64), dtype=torch.float32)

    with pytest.raises(ValueError, match="neither page-packed"):
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


def test_hybrid_groups_carry_layer_names():
    kv_caches = {
        "layer.0": _state_list(),
        "layer.1": _attn_pair(),
        "layer.2": _attn_pair(),
    }
    geometries = build_group_geometries(kv_caches, torch.bfloat16, CHUNK)

    attn, rec = geometries
    assert attn.layer_names == ("layer.1", "layer.2")
    assert rec.layer_names == ("layer.0",)


def _mla_planes(nb=4, bs=16, widths=(512, 64), dtype=torch.bfloat16):
    return tuple(
        torch.empty((nb, bs, 1, w), dtype=dtype) for w in widths
    )


def test_mla_two_plane_tuple_is_flat_mla_group():
    kv_caches = {"l0": _mla_planes(), "l1": _mla_planes()}
    geometries = build_group_geometries(kv_caches, torch.bfloat16, CHUNK)

    assert len(geometries) == 1
    group = geometries[0]
    assert group.mla and not group.recurrent
    assert group.shape == torch.Size((2, 1, CHUNK, 1, 576))
    assert group.dtype is torch.bfloat16
    assert group.layer_names == ("l0", "l1")


def test_dsa_three_plane_tuple_flattens_indexer_width():
    entry = _mla_planes(widths=(512, 64, 128))
    geometries = build_group_geometries(
        {"l0": entry}, torch.bfloat16, CHUNK
    )

    group = geometries[0]
    assert group.mla
    assert group.shape == torch.Size((1, 1, CHUNK, 1, 704))


def test_v4_mixed_dtype_scale_flattens_bytes():
    latent = torch.empty((4, 16, 1, 512), dtype=torch.int8)
    scale = torch.empty((4, 16, 1, 32), dtype=torch.float16)
    geometries = build_group_geometries(
        {"l0": (latent, scale)}, torch.bfloat16, CHUNK
    )

    group = geometries[0]
    # (512 * 1 + 32 * 2) bytes, one-byte primary dtype -> 576 elements
    assert group.shape == torch.Size((1, 1, CHUNK, 1, 576))
    assert group.dtype is torch.int8


def test_v4_shared_pool_offset_views_classify_not_raise():
    pool = torch.empty((4, 16, 1, 512 + 64), dtype=torch.bfloat16)
    latent = pool[:, :, :, :512]
    rope = pool[:, :, :, 512:]

    geometries = build_group_geometries(
        {"l0": (latent, rope)}, torch.bfloat16, CHUNK
    )

    assert len(geometries) == 1
    assert geometries[0].mla
    assert geometries[0].shape == torch.Size((1, 1, CHUNK, 1, 576))


def test_one_head_kv_pair_stays_standard_attention():
    k = torch.empty((4, 16, 1, 64), dtype=torch.bfloat16)
    v = torch.empty((4, 16, 1, 64), dtype=torch.bfloat16)

    geometries = build_group_geometries(
        {"l0": (k, v)}, torch.bfloat16, CHUNK
    )

    group = geometries[0]
    assert not group.mla
    assert group.shape == torch.Size((1, 2, CHUNK, 1, 64))


def test_dense_mla_and_dsa_layers_form_two_ordered_groups():
    kv_caches = {
        "l0": _mla_planes(widths=(512, 64)),
        "l1": _mla_planes(widths=(512, 64, 128)),
        "l2": _mla_planes(widths=(512, 64, 128)),
    }
    geometries = build_group_geometries(kv_caches, torch.bfloat16, CHUNK)

    assert len(geometries) == 2
    dense, dsa = geometries
    assert dense.layer_indices == (0,)
    assert dense.shape == torch.Size((1, 1, CHUNK, 1, 576))
    assert dsa.layer_indices == (1, 2)
    assert dsa.shape == torch.Size((2, 1, CHUNK, 1, 704))


def test_latent_only_single_plane_tuple_is_mla():
    entry = (torch.empty((4, 16, 1, 512), dtype=torch.bfloat16),)
    geometries = build_group_geometries(
        {"l0": entry}, torch.bfloat16, CHUNK
    )

    assert geometries[0].mla
    assert geometries[0].shape == torch.Size((1, 1, CHUNK, 1, 512))


def test_mla_byte_total_not_itemsize_multiple_raises():
    latent = torch.empty((4, 16, 1, 512), dtype=torch.float16)
    scale = torch.empty((4, 16, 1, 1), dtype=torch.int8)

    with pytest.raises(ValueError, match="not a multiple"):
        build_group_geometries(
            {"l0": (latent, scale)}, torch.bfloat16, CHUNK
        )


def test_flat_state_page_view_raw_form():
    entry = torch.arange(5 * 1000, dtype=torch.int8).reshape(5, 1, 1, 1000)
    views = flat_state_page_view(entry, 1000)
    assert [nbytes for _view, nbytes in views] == [1000]
    view = views[0][0]

    assert view.shape == (5, 1000)
    assert view.dtype is torch.int8
    assert view[2].data_ptr() == entry[2, 0, 0].data_ptr()
    assert torch.equal(view[3], entry[3, 0, 0])


def test_flat_state_page_view_state_list_form():
    pages, page_bytes = 5, 1280
    raw = torch.arange(pages * page_bytes, dtype=torch.uint8).reshape(
        pages, page_bytes
    )
    conv = raw[:, :768].view(torch.bfloat16).view(pages, 3, 128)
    ssm = raw[:, 768:].view(torch.float32).view(pages, 2, 64)
    views = flat_state_page_view([conv, ssm], page_bytes)

    assert [nbytes for _view, nbytes in views] == [page_bytes]
    view = views[0][0]
    assert view.shape == (pages, page_bytes)
    assert view.dtype is torch.uint8
    assert torch.equal(view[1], raw[1])


def test_flat_state_page_view_contract_violations():
    entry = torch.empty((5, 1, 1, 999), dtype=torch.int8)
    with pytest.raises(ValueError, match="does not match"):
        flat_state_page_view(entry, 1000)

    raw = torch.empty((5, 1792), dtype=torch.uint8)
    conv = torch.as_strided(raw, (5, 3, 128), (1792, 128, 1))
    ssm = torch.as_strided(raw, (5, 2, 64), (1792, 64, 1), storage_offset=768)
    with pytest.raises(ValueError, match="page-packed stride"):
        flat_state_page_view([conv, ssm], 768)

    conv = torch.empty((5, 3, 2560), dtype=torch.bfloat16)
    ssm = torch.empty((5, 2, 64), dtype=torch.float32)
    with pytest.raises(ValueError, match="slice-planes content"):
        flat_state_page_view([conv, ssm], 16000)
