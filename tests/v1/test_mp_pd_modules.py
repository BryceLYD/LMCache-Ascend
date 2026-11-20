# SPDX-License-Identifier: Apache-2.0
"""Module-level tests for the MP+PD direct-transfer pipeline (commit 1-7).

CPU-only: no NPU, no LMCache server, no network. Every test exercises one
module of the branch with fakes, mirroring the runtime contracts:

  1  lmcache_v056 frozen snapshot   -> wire messages round-trip (msgspec)
  2  (rewire)                       -> import surface of the dev-shaped tree
  3  mooncake channel               -> protocol structs + factory dispatch
  4  PD backend family              -> message extension semantics
  5  ledger / engine config shim    -> coverage resolution + decision rules
  6  group geometry classifier      -> attention / MLA / mamba layouts
     (proxy is IO-heavy; its pure helpers are covered in test 5)

Run:  pytest tests/v1/test_mp_pd_modules.py -v
"""

# Standard
from types import SimpleNamespace

# Third Party
import msgspec
import pytest
import torch

# ======================================================================
# Commit 1 - frozen snapshot: PD wire messages survive an encode/decode
# round-trip exactly (the wire format is the reason the snapshot exists).
# ======================================================================

from lmcache_ascend.v1.lmcache_v056.pd_backend import (
    AllocRequest,
    AllocResponse,
    CacheEngineKey,
    PDConfig,
    PDBackend,
)
from lmcache_ascend.v1.lmcache_v056.cache_key import CacheEngineKey as KeyFromSnapshot


class TestFrozenSnapshot:
    def test_alloc_request_roundtrip(self):
        req = AllocRequest(
            keys=["m@1@0@abc123@bfloat16", "m@1@0@def456@bfloat16"],
            fmt=1,
            shape=[2, 2, 256, 2, 128],
            dtype="bfloat16",
            last_chunk_toks=100,
        )
        decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(req), type=AllocRequest)
        assert decoded.keys == req.keys
        assert decoded.shape == req.shape
        assert decoded.last_chunk_toks == 100

    def test_alloc_response_roundtrip(self):
        resp = AllocResponse(
            already_sent_indexes=[0, 2],
            remote_indexes=[7, 8, 9],
        )
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(resp), type=AllocResponse
        )
        assert decoded.already_sent_indexes == [0, 2]
        assert decoded.remote_indexes == [7, 8, 9]

    def test_cache_engine_key_identity(self):
        k1 = CacheEngineKey("m", 1, 0, 0xABCD, torch.bfloat16)
        k2 = CacheEngineKey.from_string(k1.to_string())
        assert k1 == k2
        assert hash(k1) == hash(k2)
        assert KeyFromSnapshot is CacheEngineKey

    def test_pd_config_requires_peer_for_receiver(self):
        cfg = SimpleNamespace(
            pd_role="receiver",
            pd_buffer_size=1024,
            pd_buffer_device="npu",
            pd_peer_host="10.0.0.1",
            pd_peer_init_port=[7300],
            pd_peer_alloc_port=[7400],
            pd_peer_query_port=None,
            pd_proxy_host=None,
            pd_proxy_port=None,
            pd_skip_proxy_notification=True,
        )
        meta = SimpleNamespace(worker_id=0)
        pd_cfg = PDConfig.from_cache_engine_config(cfg, meta, tp_rank=0)
        assert pd_cfg.peer_init_port == 7300
        assert pd_cfg.buffer_device == "npu:0"  # get_correct_device applied

    def test_transfer_utils_handshake_structs(self):
        from lmcache_ascend.v1.lmcache_v056.transfer_utils import (
            P2PInitSideRetMsg,
            SideMsg,
        )

        msg = msgspec.msgpack.encode(P2PInitSideRetMsg(peer_lookup_url="tcp://h:1"))
        decoded = msgspec.msgpack.decode(msg, type=SideMsg)
        assert isinstance(decoded, P2PInitSideRetMsg)
        assert decoded.peer_lookup_url == "tcp://h:1"

    def test_token_database_chunking(self):
        from lmcache_ascend.v1.lmcache_v056.token_database import ChunkedTokenDatabase

        # Mirror pd_engine's duck-typed shims (config + metadata).
        db_config = SimpleNamespace(
            chunk_size=256,
            save_unfull_chunk=True,
            remote_url=None,
            enable_pd=True,
            pre_caching_hash_algorithm="builtin",
            get_extra_config_value=lambda key, default=None: default,
        )
        db_metadata = SimpleNamespace(
            model_name="test-model",
            world_size=1,
            worker_id=0,
            kv_dtype=torch.bfloat16,
            use_mla=False,
        )
        db = ChunkedTokenDatabase(db_config, db_metadata)
        toks = list(range(600))  # 2 full chunks + a 88-token tail
        ranges = [(s, e) for s, e, _k in db.process_tokens(tokens=toks)]
        assert len(ranges) == 3
        assert (ranges[0][0], ranges[0][1]) == (0, 256)
        assert (ranges[-1][0], ranges[-1][1]) == (512, 600)


# ======================================================================
# Commit 2 - dev-shaped tree: the rewired modules import cleanly and the
# in-process surface is really gone.
# ======================================================================

class TestDevRewire:
    def test_runtime_chain_imports(self):
        import lmcache_ascend.v1.storage_backend.pd.backend  # noqa: F401
        import lmcache_ascend.v1.storage_backend.pd.messages  # noqa: F401
        import lmcache_ascend.v1.storage_backend.pd.sender_mixin  # noqa: F401
        import lmcache_ascend.v1.storage_backend.pd.receiver_mixin  # noqa: F401
        import lmcache_ascend.v1.npu_connector.npu_connectors  # noqa: F401

    def test_zmq_helpers_carried_locally(self):
        from lmcache_ascend.v1 import rpc_utils

        assert hasattr(rpc_utils, "get_zmq_context")
        assert hasattr(rpc_utils, "get_zmq_socket")

    def test_inprocess_factory_is_gone(self):
        import lmcache_ascend.v1.npu_connector as pkg

        assert not hasattr(pkg, "CreateNPUConnector")

    def test_v2_connector_class_survives(self):
        from lmcache_ascend.v1.npu_connector.npu_connectors import (  # noqa: F401
            VLLMPagedMemNPUConnectorV2,
        )

    def test_upstream_pin_tracks_dev(self):
        import lmcache_ascend

        assert lmcache_ascend.LMCACHE_UPSTREAM_TAG == "dev"


# ======================================================================
# Commit 3 - mooncake channel: protocol structs + factory dispatch.
# ======================================================================

class TestMooncakeChannel:
    def test_protocol_struct_roundtrip(self):
        from lmcache_ascend.v1.transfer_channel.mooncake_protocol import (
            MooncakeInitRequest,
            MooncakeInitResponse,
            MooncakeMsg,
        )

        req = MooncakeInitRequest(local_id="d-1", engine_name="10.0.0.1:8888")
        out = msgspec.msgpack.decode(msgspec.msgpack.encode(req), type=MooncakeMsg)
        assert isinstance(out, MooncakeInitRequest)
        assert out.engine_name == "10.0.0.1:8888"
        resp = MooncakeInitResponse(engine_name="10.0.0.2:8888")
        out = msgspec.msgpack.decode(msgspec.msgpack.encode(resp), type=MooncakeMsg)
        assert isinstance(out, MooncakeInitResponse)

    def test_factory_accepts_mooncake(self):
        import inspect

        from lmcache_ascend.v1.transfer_channel import CreateTransferChannel

        src = inspect.getsource(CreateTransferChannel)
        assert '"mooncake"' in src or "'mooncake'" in src

    def test_channel_module_imports_without_mooncake_installed(self):
        import lmcache_ascend.v1.transfer_channel.mooncake_channel  # noqa: F401


# ======================================================================
# Commit 4 - PD backend family: the extended alloc message keeps the
# legacy fields and adds the token context + per-group geometry.
# ======================================================================

from lmcache_ascend.v1.storage_backend.pd.messages import (
    AscendAllocRequest,
    AscendAllocResponse,
    PullDoneSignal,
    PullReadyNotif,
    AscendPDMsg,
)


class TestPDTransportLayer:
    def test_extended_alloc_request_defaults_legacy(self):
        req = AscendAllocRequest(
            keys=["k1"], fmt=1, shape=[2, 2, 256, 2, 128],
            dtype="bfloat16", last_chunk_toks=256,
        )
        assert req.token_ids == []          # old senders decode fine
        assert req.token_start == 0
        assert req.shapes == []             # single-shape legacy path

    def test_extended_alloc_request_hybrid_geometry(self):
        req = AscendAllocRequest(
            keys=["k1"], fmt=1, shape=[16, 2, 256, 2, 128],
            dtype="bfloat16", last_chunk_toks=100,
            token_ids=list(range(600)),
            token_start=0,
            shapes=[[16, 2, 256, 2, 128], [48, 16, 256, 128]],
            dtypes=["bfloat16", "float32"],
        )
        raw = msgspec.msgpack.encode(req)
        out = msgspec.msgpack.decode(raw, type=AscendPDMsg)
        assert isinstance(out, AscendAllocRequest)
        assert len(out.shapes) == 2 and len(out.dtypes) == 2
        assert out.token_ids[:3] == [0, 1, 2]

    def test_response_carries_uuid_refs_and_failure_flag(self):
        resp = AscendAllocResponse(
            already_sent_indexes=[],
            remote_indexes=[4, 5],
            remote_buffer_uuids=["buf-a", "buf-a"],
            alloc_failed=False,
        )
        out = msgspec.msgpack.decode(msgspec.msgpack.encode(resp), type=AscendPDMsg)
        assert isinstance(out, AscendAllocResponse)
        assert out.remote_buffer_uuids == ["buf-a", "buf-a"]

    def test_pull_mode_lifecycle_messages(self):
        notif = PullReadyNotif(
            pull_id="p1", keys=["k"], sender_buffer_uuids=["u"],
            sender_mem_indexes=[0], sender_id="s1",
            sender_done_url="tcp://h:9999",
            fmt=1, shape=[2, 2, 256, 2, 128], dtype="bfloat16",
            last_chunk_toks=256,
        )
        out = msgspec.msgpack.decode(msgspec.msgpack.encode(notif), type=AscendPDMsg)
        assert isinstance(out, PullReadyNotif) and out.pull_id == "p1"
        done = msgspec.msgpack.decode(
            msgspec.msgpack.encode(PullDoneSignal(pull_id="p1")), type=AscendPDMsg
        )
        assert isinstance(done, PullDoneSignal) and done.pull_id == "p1"

    def test_load_instruction_carries_null_block_id(self):
        # Regression: the null-slot guard's field must live on
        # the LOAD instruction -- a bulk edit once dropped it
        # into the send instruction and the decode scheduler
        # crashed constructing it. Skipped where vllm is
        # missing (the local battery); runs on the node.
        pytest.importorskip("vllm")
        # Third Party
        from lmcache_ascend.integration.vllm.pd_engine import (
            PDLoadInstruction,
            PDSendInstruction,
        )

        instr = PDLoadInstruction(
            request_id="r1",
            token_ids=[1, 2],
            block_ids=[0],
            group_block_ids={0: [0]},
            block_size=128,
            start=0,
            end=256,
            null_block_id=7,
        )
        assert instr.null_block_id == 7
        assert "null_block_id" in PDLoadInstruction.__dataclass_fields__
        assert "null_block_id" not in PDSendInstruction.__dataclass_fields__


# ======================================================================
# Commit 5 - ledger: the KV-source coverage resolution and every query
# the connector drives decisions from.
# ======================================================================

from lmcache_ascend.v1.mp_pd.ledger import KVSource, RequestLedger


class TestLedger:
    def test_disjoint_priority_resolution(self):
        led = RequestLedger("r1")
        led.update_coverage(apc_tokens=100, l1_tokens=200, pd_tokens=350)
        sources = [(s.source, s.start, s.end) for s in led.segments]
        assert sources == [
            (KVSource.APC, 0, 100),
            (KVSource.L1, 100, 200),
            (KVSource.PD_IN, 200, 350),
        ]

    def test_no_double_count_external_tokens(self):
        led = RequestLedger("r2")
        led.update_coverage(apc_tokens=300, l1_tokens=250, pd_tokens=250)
        assert led.external_tokens == 0  # APC already covers L1/PD reach

    def test_pd_receive_segments_beyond_l1(self):
        led = RequestLedger("r3")
        led.update_coverage(apc_tokens=0, l1_tokens=256, pd_tokens=768)
        segs = led.pd_receive_segments()
        assert len(segs) == 1
        assert (segs[0].start, segs[0].end) == (256, 768)

    def test_coverage_shrink_replaces_map(self):
        led = RequestLedger("r4")
        led.update_coverage(apc_tokens=0, l1_tokens=500, pd_tokens=500)
        led.update_coverage(apc_tokens=500, l1_tokens=0, pd_tokens=0)
        assert led.segments == [led.segments[0]]
        assert led.segments[0].source is KVSource.APC
        assert led.external_tokens == 0

    def test_invalid_segment_rejected(self):
        from lmcache_ascend.v1.mp_pd.ledger import Segment

        with pytest.raises(ValueError):
            Segment(10, 5, KVSource.APC)

    def test_describe_is_lossless_enough(self):
        led = RequestLedger("r5")
        led.update_coverage(apc_tokens=8, l1_tokens=16, pd_tokens=32)
        d = led.describe()
        assert "apc=8" in d and "l1=16" in d and "pd=32" in d


# ======================================================================
# Commit 6 - group geometry: every registration form classifies and
# computes the right page geometry (mirrors the 24-test CPU suite, kept
# here as the per-commit smoke).
# ======================================================================

from lmcache_ascend.v1.mp_pd.group_geometry import (
    KVGroupGeometry,
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
    """[conv, ssm] views sharing one padded page per block (slice-planes)."""
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


def _mla_tuple(blocks=2, block=16):
    """MLA/DSA plane tuple: [latent, scale] with differing widths/dtypes."""
    return (
        torch.empty((blocks, block, 1, 8), dtype=torch.bfloat16),
        torch.empty((blocks, block, 1, 1), dtype=torch.float32),
    )


class TestGroupGeometry:
    def test_dense_attention_classifies(self):
        geos = build_group_geometries(
            {"layer-0": _attn_pair()}, torch.bfloat16, CHUNK
        )
        assert len(geos) == 1
        g = geos[0]
        assert not g.recurrent and not g.mla
        assert g.shape[2] == CHUNK

    def test_hybrid_attention_plus_mamba(self):
        caches = {
            "model.layers.0.self_attn": _attn_pair(),
            "model.layers.0.mamba": _state_list(),
        }
        geos = build_group_geometries(caches, torch.bfloat16, CHUNK)
        assert len(geos) == 2
        assert any(g.recurrent for g in geos)
        assert any((not g.recurrent) and (not g.mla) for g in geos)
        # attention groups order before recurrent groups
        assert geos[0].recurrent is False and geos[1].recurrent is True

    def test_mla_plane_tuple_classifies(self):
        geos = build_group_geometries(
            {"layer-0": _mla_tuple()}, torch.bfloat16, CHUNK
        )
        assert len(geos) == 1 and geos[0].mla
        # one flat plane of the summed width: bf16 latent (8 wide = 16B)
        # + fp32 scale (1 wide = 4B) = 20B -> 10 bf16 elements
        assert geos[0].shape == torch.Size((1, 1, CHUNK, 1, 10))

    def test_mla_only_registry_does_not_fall_through(self):
        caches = {"model.layers.%d.self_attn" % i: _mla_tuple() for i in range(3)}
        geos = build_group_geometries(caches, torch.bfloat16, CHUNK)
        # Identical geometry buckets into ONE MLA group holding
        # all three layers (the pre-e7c580a bug was zero groups).
        assert len(geos) == 1 and geos[0].mla
        assert geos[0].num_layers == 3

    def test_flat_state_page_view_covers_state_list(self):
        views = flat_state_page_view(_state_list(), region_bytes=1280)
        got = sum(nbytes for _view, nbytes in views)
        assert got == 1280

    def test_page_packed_rejects_noncontiguous_inner_layout(self):
        # Inner-dim stride != 1 means an unknown layout the transfer
        # view must refuse instead of silently misreading.
        first = torch.empty_strided((5, 128), (640, 2), dtype=torch.bfloat16)
        second = torch.empty((5, 64), dtype=torch.float32)
        with pytest.raises(ValueError, match="innermost-contiguous"):
            flat_state_page_view([first, second], region_bytes=1280)

    def test_mla_pair_separate_storages_is_not_a_state_list(self):
        # An MLA-style (k, v) pair lives in separate
        # storages and must stay unclassifiable, not become a
        # bogus recurrent group.
        k = torch.empty((2, 16, 1, 512), dtype=torch.bfloat16)
        v = torch.empty((2, 16, 64), dtype=torch.bfloat16)
        geos = build_group_geometries(
            {"layer.0": (k, v)}, torch.bfloat16, CHUNK
        )
        assert geos is None

    def test_unknown_layout_returns_none(self):
        # Unclassifiable entries make the build return None (the caller
        # falls back to the legacy model-config geometry), not raise.
        geos = build_group_geometries(
            {"layer-0": torch.empty((3, 3, 3))}, torch.bfloat16, CHUNK
        )
        assert geos is None
