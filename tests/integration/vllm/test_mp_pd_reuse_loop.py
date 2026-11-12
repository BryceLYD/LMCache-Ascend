# SPDX-License-Identifier: Apache-2.0
"""Step 6 acceptance checks: the decoder-side reuse loop.

Two features close the loop that lets a repeated prompt skip the PD
re-transfer on the decoder:

* ``pd_store_through``: the worker-confirmed PD-landed prefix is
  persisted into the decoder's own LMCache server by injecting
  STORE-direction request metadata into the upstream store flow.
* ``pd_alloc_check_l1``: the receiver's alloc handler dedups incoming
  chunks against the decoder's L1 (via the L1QueryOracle), reporting
  them through the same ``already_sent_indexes`` channel as the
  receive-buffer dedup.

Like test_mp_pd_pd_leg.py these run against real upstream lmcache/vllm
code with fakes at the process boundaries (no NPU, no server, no live
P/D pair).
"""

# Standard
from types import SimpleNamespace
from typing import Optional
import importlib
import threading

# Third Party
import pytest

_PD_EXTRA = {
    "pd_buffer_size": 1024,
    "pd_peer_host": "localhost",
    "pd_peer_init_port": [7300],
    "pd_peer_alloc_port": [7400],
    "pd_proxy_host": "localhost",
    "pd_proxy_port": 7500,
    "chunk_size": 256,
}


def _import_stack():
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    pytest.importorskip("lmcache_ascend")

    mp_pd_entry = importlib.import_module(
        "lmcache_ascend.integration.vllm.lmcache_ascend_mp_pd_connector"
    )
    pd_engine_mod = importlib.import_module(
        "lmcache_ascend.integration.vllm.pd_engine"
    )
    backend_mod = importlib.import_module(
        "lmcache_ascend.v1.storage_backend.pd.backend"
    )
    sender_mod = importlib.import_module(
        "lmcache_ascend.v1.storage_backend.pd.sender_mixin"
    )
    receiver_mod = importlib.import_module(
        "lmcache_ascend.v1.storage_backend.pd.receiver_mixin"
    )
    messages_mod = importlib.import_module(
        "lmcache_ascend.v1.storage_backend.pd.messages"
    )

    return SimpleNamespace(
        conn_cls=mp_pd_entry.LMCacheAscendMPPDConnectorV1Dynamic,
        pd_engine_mod=pd_engine_mod,
        backend_mod=backend_mod,
        sender_mod=sender_mod,
        receiver_mod=receiver_mod,
        messages_mod=messages_mod,
    )


def _fake_request(req_id="r1", n_prompt=700):
    return SimpleNamespace(
        request_id=req_id,
        num_prompt_tokens=n_prompt,
        all_token_ids=list(range(n_prompt)),
        prompt_token_ids=list(range(n_prompt)),
        kv_transfer_params=None,
        cache_salt="",
        status=None,
    )


class _FakeSchedulerAdapter:
    def __init__(self):
        self.lmcache_tokens_per_chunk = 256


def _make_connector(stack, store_through=True):
    """Scheduler-side connector wired for store-through tests."""
    connector = object.__new__(stack.conn_cls)
    connector._ledgers = {}
    connector._pd_specs = {}
    connector._pd_engine = None
    connector._pd_received = {}
    connector._pd_arrived_pending = []
    connector._pd_complete_logged = set()
    connector._pd_kv_caches = None
    connector._pd_arrived = set()
    connector._pd_key_db = None
    connector._pd_chunk_size = 256
    connector._pd_load_issued = set()
    connector._pd_load_done = set()
    connector._pd_failed_blocks = set()
    connector._pd_landed = {}
    connector._pd_loaded_pending = {}
    connector._pd_store_through = store_through
    connector._pd_alloc_check_l1 = False
    connector._pd_l1_hash_algorithm = "blake3"
    connector._pd_save_policy = "both"
    connector._group_tokens_per_block = [16]
    connector.request_trackers = {}
    connector.scheduler_adapter = _FakeSchedulerAdapter()
    connector._vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_consumer", kv_connector_extra_config={}
        ),
    )
    connector.lazy_offload = False
    connector._can_store = False
    return connector


def _real_tracker(stack, request):
    # Third Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestTracker

    return LMCacheMPRequestTracker(request)


def _metadata_stub(stack):
    """A real upstream LMCacheMPConnectorMetadata (the container the
    worker's wait_for_save iterates)."""
    # Third Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPConnectorMetadata

    return LMCacheMPConnectorMetadata()


def _store_metas(metadata):
    return [m for m in metadata.requests if m.direction == "STORE"]


# =====================================================================
# Feature 1: store-through
# =====================================================================


def test_store_through_injects_meta_and_advances_num_stored():
    """landed=700, mp_chunk=256, L1 miss: one STORE meta [0,512) (full
    chunks only) and num_stored_tokens advances to 512; re-running is
    idempotent."""
    stack = _import_stack()
    connector = _make_connector(stack)
    request = _fake_request("st", 700)
    connector.request_trackers["st"] = _real_tracker(stack, request)
    tracker = connector.request_trackers["st"]
    tracker.allocated_block_ids = {0: list(range(44))}
    connector._pd_landed["st"] = 700

    metadata = _metadata_stub(stack)
    connector._pd_inject_store_through(metadata)
    metas = _store_metas(metadata)
    assert len(metas) == 1
    assert metas[0].request_id == "st"
    assert (metas[0].op.start, metas[0].op.end) == (0, 512)
    assert metas[0].op.block_ids == [list(range(32))]
    assert tracker.num_stored_tokens == 512

    # Idempotent: nothing left above the advanced watermark.
    metadata2 = _metadata_stub(stack)
    connector._pd_inject_store_through(metadata2)
    assert _store_metas(metadata2) == []
    assert tracker.num_stored_tokens == 512


def test_store_through_resumes_above_l1_hit():
    """An L1 hit already covers [0,768) (num_stored=768): the pseudo
    store starts exactly at the watermark, never re-storing the L1
    part."""
    stack = _import_stack()
    connector = _make_connector(stack)
    request = _fake_request("l1st", 1024)
    connector.request_trackers["l1st"] = _real_tracker(stack, request)
    tracker = connector.request_trackers["l1st"]
    tracker.num_stored_tokens = 768
    tracker.num_lmcache_hit_tokens = 768
    tracker.allocated_block_ids = {0: list(range(64))}
    connector._pd_landed["l1st"] = 1024

    metadata = _metadata_stub(stack)
    connector._pd_inject_store_through(metadata)
    metas = _store_metas(metadata)
    assert len(metas) == 1
    assert (metas[0].op.start, metas[0].op.end) == (768, 1024)
    assert tracker.num_stored_tokens == 1024


def test_store_through_skips_sub_chunk_and_disabled():
    stack = _import_stack()
    # landed below one chunk -> nothing
    connector = _make_connector(stack)
    request = _fake_request("tiny", 700)
    connector.request_trackers["tiny"] = _real_tracker(stack, request)
    connector.request_trackers["tiny"].allocated_block_ids = {
        0: list(range(44))
    }
    connector._pd_landed["tiny"] = 200
    metadata = _metadata_stub(stack)
    connector._pd_inject_store_through(metadata)
    assert _store_metas(metadata) == []

    # feature off -> nothing even with a landed prefix
    connector = _make_connector(stack, store_through=False)
    request = _fake_request("off", 700)
    connector.request_trackers["off"] = _real_tracker(stack, request)
    connector.request_trackers["off"].allocated_block_ids = {
        0: list(range(44))
    }
    connector._pd_landed["off"] = 700
    metadata = _metadata_stub(stack)
    connector._pd_inject_store_through(metadata)
    assert _store_metas(metadata) == []


def test_store_through_ignores_requests_without_tracker():
    stack = _import_stack()
    connector = _make_connector(stack)
    connector._pd_landed["ghost"] = 512
    metadata = _metadata_stub(stack)
    connector._pd_inject_store_through(metadata)
    assert _store_metas(metadata) == []


class _FakeKeyDB:
    def __init__(self, chunk_size=256):
        self.chunk_size = chunk_size

    def process_tokens(self, tokens=None, **kwargs):
        n = len(tokens)
        cs = self.chunk_size
        i = 0
        while i < n:
            e = min(i + cs, n)
            yield (i, e, f"key-{i // cs}")
            i = e


class _FakeObj:
    def __init__(self, name):
        self.name = name

    def ref_count_up(self):
        pass

    def ref_count_down(self):
        pass


def _make_engine(stack, present):
    """PDEngine shell with a fake backend whose _lookup answers per the
    ``present`` set of chunk keys."""
    engine = object.__new__(stack.pd_engine_mod.PDEngine)
    engine.backend = SimpleNamespace(
        _lookup=lambda key, pin=False: (
            _FakeObj(key) if key in present else None
        ),
    )
    engine.token_db = _FakeKeyDB(256)
    engine.npu_connector = SimpleNamespace(
        batched_to_gpu=lambda objs, s, e, kvcaches, slot_mapping: None
    )
    return engine


def _load_instr(stack, n_tokens=1024, start=0, end=1024):
    return stack.pd_engine_mod.PDLoadInstruction(
        request_id="r1",
        token_ids=list(range(n_tokens)),
        block_ids=list(range(n_tokens // 16)),
        block_size=16,
        start=start,
        end=end,
    )


def test_execute_pd_load_reports_consecutive_landed_prefix():
    """A gap must stop landed_end even when chunks beyond it load fine
    (store-through would otherwise persist unloaded blocks)."""
    stack = _import_stack()
    engine = _make_engine(
        stack, present={"key-0", "key-1", "key-3"}
    )
    failed, landed = engine.execute_pd_load(_load_instr(stack), {})
    assert landed == 512, "chunks 0,1 land; chunk 2 missing stops the run"
    # chunk 3 ([768,1024)) loads but must not extend landed_end
    assert set(failed) == set(range(32, 48)), "blocks of chunk 2 fail"


def test_execute_pd_load_all_present_lands_everything():
    stack = _import_stack()
    engine = _make_engine(
        stack, present={"key-0", "key-1", "key-2", "key-3"}
    )
    failed, landed = engine.execute_pd_load(_load_instr(stack), {})
    assert failed == []
    assert landed == 1024


def test_pd_loaded_worker_metadata_aggregate_keeps_max():
    stack = _import_stack()
    a = stack.pd_engine_mod.MPPDWorkerMetadata(
        completed_store_requests={}, pd_loaded={"r1": 512, "r2": 256}
    )
    b = stack.pd_engine_mod.MPPDWorkerMetadata(
        completed_store_requests={}, pd_loaded={"r1": 256, "r3": 700}
    )
    merged = a.aggregate(b)
    assert merged.pd_loaded == {"r1": 512, "r2": 256, "r3": 700}


def test_update_connector_output_records_pd_loaded_monotonically():
    stack = _import_stack()
    connector = _make_connector(stack)
    meta = stack.pd_engine_mod.MPPDWorkerMetadata(
        completed_store_requests={}, pd_loaded={"r1": 512}
    )
    connector.update_connector_output(
        SimpleNamespace(
            kv_cache_events=None, kv_connector_worker_meta=meta
        )
    )
    assert connector._pd_landed == {"r1": 512}
    # A stale smaller report must not lower the watermark.
    meta2 = stack.pd_engine_mod.MPPDWorkerMetadata(
        completed_store_requests={}, pd_loaded={"r1": 256}
    )
    connector.update_connector_output(
        SimpleNamespace(
            kv_cache_events=None, kv_connector_worker_meta=meta2
        )
    )
    assert connector._pd_landed == {"r1": 512}


def test_request_finished_drops_landed_marker():
    stack = _import_stack()
    connector = _make_connector(stack)
    connector._pd_landed["r1"] = 512

    # Third Party
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector

    request = SimpleNamespace(request_id="r1", output_token_ids=[1])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            LMCacheMPConnector,
            "request_finished",
            lambda self, req, blocks: (False, None),
        )
        connector.request_finished(request, [])
    assert "r1" not in connector._pd_landed


# =====================================================================
# Feature 2: alloc-time L1 dedup
# =====================================================================


def test_ascend_alloc_request_roundtrip():
    """The token-extended request survives the AscendPDMsg union both
    ways, and a legacy plain AllocRequest still decodes."""
    stack = _import_stack()
    # Third Party
    from lmcache.v1.storage_backend.pd_backend import AllocRequest
    import msgspec

    msg = stack.messages_mod.AscendAllocRequest(
        keys=["k0", "k1"],
        fmt=1,
        shape=[2, 256, 8, 128],
        dtype="bfloat16",
        last_chunk_toks=100,
        token_ids=list(range(512)),
        token_start=256,
    )
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(msg), type=stack.messages_mod.AscendPDMsg
    )
    assert isinstance(decoded, AllocRequest)
    assert decoded.token_ids == list(range(512))
    assert decoded.token_start == 256

    legacy = AllocRequest(
        keys=["k0"], fmt=1, shape=[1], dtype="bfloat16", last_chunk_toks=8
    )
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(legacy), type=stack.messages_mod.AscendPDMsg
    )
    assert type(decoded) is AllocRequest
    assert decoded.keys == ["k0"]


def _sender(stack):
    return object.__new__(stack.sender_mod.AscendPDSenderMixin)


def _mem_obj():
    # Third Party
    from lmcache.v1.memory_management import MemoryFormat
    import torch

    return SimpleNamespace(
        meta=SimpleNamespace(
            fmt=MemoryFormat.KV_2LTD,
            shape=torch.Size([2, 256, 8, 128]),
            dtype=torch.bfloat16,
        )
    )


def test_sender_builds_token_carrying_alloc_request():
    stack = _import_stack()
    sender = _sender(stack)
    keys = [
        SimpleNamespace(to_string=lambda i=i: f"key-{i}") for i in range(2)
    ]
    spec = SimpleNamespace(token_ids=list(range(512)), start=256)
    req = sender._get_remote_alloc_request(keys, [_mem_obj(), _mem_obj()], spec)
    assert isinstance(req, stack.messages_mod.AscendAllocRequest)
    assert req.token_ids == list(range(512))
    assert req.token_start == 256
    assert req.keys == ["key-0", "key-1"]


def test_sender_falls_back_to_plain_request_without_tokens():
    stack = _import_stack()
    sender = _sender(stack)
    keys = [SimpleNamespace(to_string=lambda: "key-0")]
    req = sender._get_remote_alloc_request(keys, [_mem_obj()], None)
    assert type(req).__name__ == "AllocRequest"
    assert not hasattr(req, "token_ids") or not req.token_ids


class _FakeOracle:
    """Answers at server-chunk granularity like L1QueryOracle."""

    def __init__(self, present, chunk_size=256):
        self._present = set(present)
        self.chunk_size = chunk_size

    def present_server_chunks(self, token_ids, start, end):
        cs = self.chunk_size
        return frozenset(
            c
            for c in range(start // cs, end // cs)
            if c in self._present
        )


def _pd_key(i):
    """A well-formed PD chunk key string (CacheEngineKey layout)."""
    return f"test-model@1@0@{i:064x}@bfloat16"


def _receiver(stack, oracle, buffer_keys=()):
    # Third Party
    from lmcache.utils import CacheEngineKey

    backend = object.__new__(stack.backend_mod.AscendPDBackend)
    backend.data = {
        CacheEngineKey.from_string(k): _FakeObj(k) for k in buffer_keys
    }
    backend.data_lock = threading.Lock()
    backend.l1_oracle = oracle
    return backend


def _alloc_msg(stack, keys, token_ids, token_start, pd_chunk=256):
    # KV_2LTD shape is [2, num_layers, num_tokens, hidden_dim]: the
    # token dim sits at index 2 (fmt=1 -> token_dim()==2), so the PD
    # chunk size must land at shape[2] for the receiver to read it.
    return stack.messages_mod.AscendAllocRequest(
        keys=list(keys),
        fmt=1,
        shape=[2, 8, pd_chunk, 128],
        dtype="bfloat16",
        last_chunk_toks=pd_chunk,
        token_ids=list(token_ids),
        token_start=token_start,
    )


def test_receiver_l1_covered_key_indexes_equal_granularity():
    """pd chunk == server chunk: one PD chunk per server chunk."""
    stack = _import_stack()
    backend = _receiver(stack, _FakeOracle({0, 2}, chunk_size=256))
    msg = _alloc_msg(
        stack,
        [_pd_key(0), _pd_key(1), _pd_key(2), _pd_key(3)],
        range(1024),
        0,
        pd_chunk=256,
    )
    assert backend._l1_covered_key_indexes(msg) == {0, 2}


def test_receiver_l1_covered_key_indexes_coarse_server_chunk():
    """pd chunk 128 inside server chunk 256: covering indexes collapse
    into the containing server chunk."""
    stack = _import_stack()
    backend = _receiver(stack, _FakeOracle({0}, chunk_size=256))
    msg = _alloc_msg(
        stack,
        [_pd_key(0), _pd_key(1), _pd_key(2), _pd_key(3)],
        range(512),
        0,
        pd_chunk=128,
    )
    assert backend._l1_covered_key_indexes(msg) == {0, 1}


def test_receiver_l1_covered_snaps_head_clipped_start():
    """A block-coverage clamp can leave token_start mid-chunk on the
    sender; the mapping must snap onto the PD chunk grid, not drift."""
    stack = _import_stack()
    backend = _receiver(stack, _FakeOracle({1}, chunk_size=256))
    # token_start=112 sits inside PD chunk [0,256); two keys represent
    # chunks [0,256) and [256,512). Server chunk 1 covers [256,512):
    # only key 1 is L1-covered.
    msg = _alloc_msg(
        stack,
        [_pd_key(0), _pd_key(1)],
        range(512),
        112,
        pd_chunk=256,
    )
    assert backend._l1_covered_key_indexes(msg) == {1}


def test_receiver_l1_covered_fails_open():
    """No oracle / no tokens / broken oracle all yield an empty set."""
    stack = _import_stack()

    class _Boom:
        chunk_size = 256

        def present_server_chunks(self, *a):
            raise RuntimeError("server unreachable")

    msg = _alloc_msg(stack, ["k0"], range(256), 0)
    assert _receiver(stack, None)._l1_covered_key_indexes(msg) == set()

    plain = SimpleNamespace(keys=["k0"], fmt=1, shape=[2, 256, 8, 128])
    boom = _receiver(stack, _Boom())
    # Missing token_ids attribute -> getattr default -> empty
    assert boom._l1_covered_key_indexes(plain) == set()
    assert boom._l1_covered_key_indexes(msg) == set()


def test_allocate_and_put_dedups_buffer_and_l1():
    """Push-mode alloc: buffer hits and L1 hits both land in
    already_sent_indexes and never allocate pages."""
    stack = _import_stack()
    # Third Party
    from lmcache.v1.memory_management import MemoryFormat

    # First Party
    from lmcache_ascend.v1.storage_backend.pd.messages import AscendAllocResponse

    oracle = _FakeOracle({1}, chunk_size=256)
    # key 0 already in the receive buffer; key 1 in L1; key 2 new.
    backend = _receiver(stack, oracle, buffer_keys={_pd_key(0)})
    allocated = []

    def _allocate(shape, dtype, fmt=MemoryFormat.KV_2LTD, **kwargs):
        obj = _FakeObj(f"alloc-{len(allocated)}")
        allocated.append(obj)
        return obj

    backend.allocate = _allocate
    backend._peer_alloc_backoff_ttl = 0.0
    backend.transfer_channel = SimpleNamespace(
        get_local_buffer_refs=lambda objs: (
            [f"uuid-{i}" for i in range(len(objs))],
            list(range(len(objs))),
        )
    )
    backend.put = lambda key, obj: None

    msg = _alloc_msg(
        stack,
        [_pd_key(0), _pd_key(1), _pd_key(2)],
        range(768),
        0,
        pd_chunk=256,
    )
    resp = backend._allocate_and_put(msg)
    assert isinstance(resp, AscendAllocResponse)
    assert not resp.alloc_failed
    assert resp.already_sent_indexes == [0, 1]
    assert len(allocated) == 1, "only the L1-missing chunk gets a page"
    assert resp.remote_buffer_uuids == ["uuid-0"]


# =====================================================================
# L1QueryOracle
# =====================================================================


class _FakeFuture:
    def __init__(self, value):
        self._value = value

    def result(self, timeout=None):
        if isinstance(self._value, Exception):
            raise self._value
        return self._value


class _FakeHasher:
    def __init__(self, chunk_size=256):
        self.chunk_size = chunk_size

    def compute_chunk_hashes(self, token_ids, start=0, end=None):
        cs = self.chunk_size
        return [f"h{c}".encode() for c in range(start // cs, end // cs)]


class _FakeReqClient:
    def __init__(self, addresses=None, error: Optional[Exception] = None):
        self.addresses = addresses
        self.error = error
        self.locked_keys = None
        self.unlocked_keys = None

    def p2p_lookup_and_lock(self, keys, layout):
        if self.error is not None:
            return _FakeFuture(self.error)
        self.locked_keys = list(keys)
        return _FakeFuture(7)

    def p2p_query_lookup_results(self, task_id):
        if self.error is not None:
            return _FakeFuture(self.error)
        return _FakeFuture(self.addresses)

    def p2p_unlock_objects(self, keys):
        self.unlocked_keys = list(keys)
        return _FakeFuture(None)


def _oracle(stack, client):
    # Third Party
    import torch

    oracle = object.__new__(stack.pd_engine_mod.L1QueryOracle)
    oracle._server_url = "tcp://127.0.0.1:8000"
    oracle._zmq_context = None
    oracle._model_name = "m"
    oracle._kv_rank = 0
    oracle._cache_salt = ""
    oracle._timeout = 0.5
    oracle._hasher = _FakeHasher(256)
    oracle._kv_shape = torch.Size([2, 256, 8, 128])
    oracle._kv_dtype = torch.bfloat16
    oracle._req_client = client
    oracle._lock = threading.Lock()
    oracle.chunk_size = 256
    return oracle


def _addr(valid):
    return SimpleNamespace(is_valid=lambda: valid)


def test_oracle_maps_found_addresses_to_chunk_indexes():
    stack = _import_stack()
    client = _FakeReqClient(addresses=[_addr(True), _addr(False), _addr(True)])
    oracle = _oracle(stack, client)
    present = oracle.present_server_chunks(list(range(768)), 0, 768)
    assert present == frozenset({0, 2})
    # Only found keys are unlocked.
    assert [k.chunk_hash for k in client.unlocked_keys] == [b"h0", b"h2"]


def test_oracle_fails_open_on_rpc_error():
    stack = _import_stack()
    oracle = _oracle(
        stack, _FakeReqClient(error=RuntimeError("server down"))
    )
    assert oracle.present_server_chunks(list(range(512)), 0, 512) == frozenset()
    assert oracle.present_server_chunks(list(range(512)), 0, 512) == frozenset()


def test_oracle_skips_partial_trailing_chunk():
    """[0, 700) with chunk 256 -> only chunks 0,1 are queryable."""
    stack = _import_stack()
    client = _FakeReqClient(addresses=[_addr(True), _addr(True)])
    oracle = _oracle(stack, client)
    present = oracle.present_server_chunks(list(range(700)), 0, 700)
    assert present == frozenset({0, 1})
    assert len(client.locked_keys) == 2
