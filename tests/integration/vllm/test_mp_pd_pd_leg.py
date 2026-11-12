# SPDX-License-Identifier: Apache-2.0
"""Step 4 acceptance checks: the PD leg of the MP+PD connector.

These tests run against the real upstream lmcache/vllm code (no NPU,
no lmcache server, no live P/D pair): the scheduler-side pieces (发货单
extraction, send-instruction computation) and the worker->scheduler
observation path (metadata aggregation, ledger ``pd_tokens``) are
exercised with fake trackers/adapters, following the repo's
``object.__new__`` pattern (see test_handle_preemptions.py).

The assembled ``PDEngine`` itself (AscendPDBackend, transfer channels)
needs NPU and a live peer; it is covered by the 1p1d run on the env
machine, not here.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import importlib

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


class _FakeSchedulerAdapter:
    """Canned lookup results over the no-op seam."""

    def __init__(self, results):
        self.results = list(results)
        self.lmcache_tokens_per_chunk = 256

    def maybe_submit_lookup_request(self, request_id, **kwargs):
        return True

    def check_lookup_result(self, request_id):
        return self.results.pop(0) if self.results else 0

    def cleanup_lookup_result(self, request_id):
        return None

    def free_lookup_locks(self, **kwargs):
        return None

    def end_session(self, request_id):
        return None


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

    # First Party
    from lmcache_ascend.v1.mp_pd import KVSource

    return SimpleNamespace(
        conn_cls=mp_pd_entry.LMCacheAscendMPPDConnectorV1Dynamic,
        pd_engine=pd_engine_mod.PDEngine,
        pd_engine_mod=pd_engine_mod,
        kv_source=KVSource,
    )


def _vllm_config(extra, kv_role):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config=dict(extra),
            kv_role=kv_role,
            engine_id="test",
        ),
        cache_config=SimpleNamespace(cache_dtype="bfloat16", block_size=16),
    )


def _fake_request(req_id="r1", n_prompt=700, with_spec=True):
    params = None
    if with_spec:
        params = {
            "disagg_spec": {
                "req_id": req_id,
                "receiver_host": "localhost",
                "receiver_init_port": [7300],
                "receiver_alloc_port": [7400],
            }
        }
    return SimpleNamespace(
        request_id=req_id,
        num_prompt_tokens=n_prompt,
        all_token_ids=list(range(n_prompt)),
        prompt_token_ids=list(range(n_prompt)),
        kv_transfer_params=params,
        cache_salt="",
        status=None,
    )


def _make_scheduler_connector(lookup_results=(0,)):
    stack = _import_stack()
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
    connector._pd_chunk_size = 0
    connector._pd_load_issued = set()
    connector._pd_load_done = set()
    connector._pd_failed_blocks = set()
    connector._pd_save_policy = "both"
    connector.request_trackers = {}
    connector.scheduler_adapter = _FakeSchedulerAdapter(
        list(lookup_results)
    )
    connector._hit_alignment_tokens = 256
    connector._reserve_last_token_for_lookup = False
    connector._connector_stats = SimpleNamespace(
        record_lookup=lambda dt: None
    )
    connector._vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_consumer", kv_connector_extra_config={}
        ),
    )
    connector.lazy_offload = False
    connector._can_store = False
    connector._kv_cache_events = []
    return connector, stack


class _FakeKeyDB:
    """Mirrors ChunkedTokenDatabase.process_tokens: yields
    (start, end, key) with string keys."""

    def __init__(self, chunk_size=256, save_tail=True):
        self.chunk_size = chunk_size
        self.save_tail = save_tail

    def process_tokens(self, tokens=None, **kwargs):
        n = len(tokens)
        cs = self.chunk_size
        end = n if self.save_tail else n - n % cs
        i = 0
        while i < end:
            e = min(i + cs, n)
            yield (i, e, f"key-{i // cs}")
            i = e


def test_extract_spec_from_kv_transfer_params():
    stack = _import_stack()
    request = _fake_request()
    spec = stack.pd_engine_mod.extract_pd_transfer_spec(request)
    assert spec is not None
    assert spec.req_id == "r1"
    assert spec.receiver_host == "localhost"
    assert spec.receiver_init_port == [7300]
    assert spec.receiver_alloc_port == [7400]
    assert spec.num_prompt_tokens == 700
    assert spec.sent_end == 0


def test_extract_spec_returns_none_when_absent_or_malformed():
    stack = _import_stack()
    plain = SimpleNamespace(
        request_id="r2", num_prompt_tokens=8, kv_transfer_params=None
    )
    assert stack.pd_engine_mod.extract_pd_transfer_spec(plain) is None
    no_params = SimpleNamespace(request_id="r3", num_prompt_tokens=8)
    assert stack.pd_engine_mod.extract_pd_transfer_spec(no_params) is None
    broken = SimpleNamespace(
        request_id="r4",
        num_prompt_tokens=8,
        kv_transfer_params={"disagg_spec": {"req_id": "r4"}},
    )
    assert stack.pd_engine_mod.extract_pd_transfer_spec(broken) is None


def test_maybe_create_gating():
    stack = _import_stack()

    # Third Party
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    assert (
        stack.pd_engine.maybe_create(
            _vllm_config({}, "kv_producer"), KVConnectorRole.WORKER
        )
        is None
    ), "no pd keys -> disabled"
    assert (
        stack.pd_engine.maybe_create(
            _vllm_config(_PD_EXTRA, "kv_producer"),
            KVConnectorRole.SCHEDULER,
        )
        is None
    ), "scheduler role -> disabled"
    assert (
        stack.pd_engine.maybe_create(
            _vllm_config(_PD_EXTRA, "kv_both"), KVConnectorRole.WORKER
        )
        is None
    ), "kv_both assembly arrives with Step 6"
    engine = stack.pd_engine.maybe_create(
        _vllm_config(_PD_EXTRA, "kv_producer"), KVConnectorRole.WORKER
    )
    assert engine is not None and engine.pd_role == "sender"
    assert engine.chunk_size == 256
    assert engine._config.transfer_channel == "hccl"
    assert engine.assembled is False


def _real_tracker(request):
    """Build a real upstream tracker from a fake request (the tests that
    drive upstream hooks need its full method surface)."""
    # Third Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestTracker

    return LMCacheMPRequestTracker(request)


def _blocks_ns():
    return SimpleNamespace(get_block_ids=lambda: (list(range(16)),))


def test_update_state_after_alloc_records_spec():
    connector, _ = _make_scheduler_connector()
    request = _fake_request()
    connector.request_trackers["r1"] = _real_tracker(request)
    connector.update_state_after_alloc(request, _blocks_ns(), 0)
    assert "r1" in connector._pd_specs
    connector.request_finished(request, [])
    assert "r1" not in connector._pd_specs


def test_send_instructions_mid_prefill_then_tail():
    connector, _ = _make_scheduler_connector()
    request = _fake_request()
    connector.request_trackers["r1"] = _real_tracker(request)
    connector.update_state_after_alloc(request, _blocks_ns(), 0)

    tracker = connector.request_trackers["r1"]
    tracker.num_scheduled_tokens = 300
    instrs = connector._pd_build_send_instructions()
    assert len(instrs) == 1
    assert (instrs[0].start, instrs[0].end) == (0, 256)
    assert instrs[0].is_last_prefill is False

    tracker.num_scheduled_tokens = 700
    tracker.allocated_block_ids = {0: list(range(44))}
    instrs = connector._pd_build_send_instructions()
    assert len(instrs) == 1
    assert (instrs[0].start, instrs[0].end) == (256, 700)
    assert instrs[0].is_last_prefill is True

    assert connector._pd_specs["r1"].sent_end == 700
    assert connector._pd_build_send_instructions() == []


def test_send_instructions_skip_multi_group():
    connector, _ = _make_scheduler_connector()
    request = _fake_request()
    connector.request_trackers["r1"] = _real_tracker(request)
    connector.update_state_after_alloc(request, _blocks_ns(), 0)
    tracker = connector.request_trackers["r1"]
    tracker.num_scheduled_tokens = 700
    tracker.allocated_block_ids = {0: [1, 2], 1: [3, 4]}
    assert connector._pd_build_send_instructions() == []


def test_send_instructions_cover_l1_hit_prefix():
    """The shipment covers the L1-loaded prefix too (it lives in the
    paged pool just like computed tokens) — otherwise a producer-side
    L1 hit leaves the decoder with a recompute hole."""
    connector, _ = _make_scheduler_connector(lookup_results=(768,))
    request = _fake_request("hole", 917)
    connector.request_trackers["hole"] = _real_tracker(request)
    connector.update_state_after_alloc(request, _blocks_ns(), 0)
    tracker = connector.request_trackers["hole"]
    tracker.num_lmcache_hit_tokens = 768
    tracker.allocated_block_ids = {0: list(range(8))}

    tracker.num_scheduled_tokens = 100
    instrs = connector._pd_build_send_instructions()
    assert len(instrs) == 1
    assert (instrs[0].start, instrs[0].end) == (0, 768)
    assert instrs[0].is_last_prefill is False

    tracker.num_scheduled_tokens = 149
    instrs = connector._pd_build_send_instructions()
    assert len(instrs) == 1
    assert (instrs[0].start, instrs[0].end) == (768, 917)
    assert instrs[0].is_last_prefill is True
    assert connector._pd_specs["hole"].last_notified is True


def test_send_instructions_notify_only_when_nothing_left_to_ship():
    """Prefill done, everything already shipped, gate never opened:
    a notify-only instruction must still be emitted (exactly once) so
    the disagg proxy lets the decode request through."""
    connector, _ = _make_scheduler_connector()
    request = _fake_request("gate", 700)
    connector.request_trackers["gate"] = _real_tracker(request)
    connector.update_state_after_alloc(request, _blocks_ns(), 0)
    tracker = connector.request_trackers["gate"]
    tracker.num_scheduled_tokens = 700
    spec = connector._pd_specs["gate"]
    spec.sent_end = 700

    instrs = connector._pd_build_send_instructions()
    assert len(instrs) == 1
    assert instrs[0].notify_only is True
    assert instrs[0].is_last_prefill is True
    assert spec.last_notified is True
    assert connector._pd_build_send_instructions() == []


def _pd_instr(stack, req_id="r1", is_last=False, notify_only=False):
    return stack.pd_engine_mod.PDSendInstruction(
        req_id=req_id,
        receiver_host="h",
        receiver_init_port=[7300],
        receiver_alloc_port=[7400],
        token_ids=[1],
        block_ids=[],
        block_size=16,
        start=0,
        end=256,
        is_last_prefill=is_last,
        notify_only=notify_only,
    )


def test_execute_send_instructions_notify_only_opens_gate():
    connector, stack = _make_scheduler_connector()
    connector._pd_engine = MagicMock()
    connector._pd_kv_caches = []
    instr = _pd_instr(stack, notify_only=True, is_last=True)
    connector._pd_execute_send_instructions(
        SimpleNamespace(pd_send_instructions=[instr])
    )
    connector._pd_engine.send_proxy_notification.assert_called_once_with("r1")
    connector._pd_engine.execute_send.assert_not_called()


def test_execute_send_instructions_failure_opens_gate():
    """A failed last-chunk shipment must still open the proxy gate: the
    decoder recomputes the missing range, but it can only start once
    the notification arrives."""
    connector, stack = _make_scheduler_connector()
    connector._pd_engine = MagicMock()
    connector._pd_engine.execute_send.side_effect = RuntimeError("boom")
    connector._pd_kv_caches = []
    instr = _pd_instr(stack, is_last=True)
    connector._pd_execute_send_instructions(
        SimpleNamespace(pd_send_instructions=[instr])
    )
    connector._pd_engine.send_proxy_notification.assert_called_once_with("r1")


def test_execute_send_instructions_mid_failure_keeps_gate():
    connector, stack = _make_scheduler_connector()
    connector._pd_engine = MagicMock()
    connector._pd_engine.execute_send.side_effect = RuntimeError("boom")
    connector._pd_kv_caches = []
    instr = _pd_instr(stack, is_last=False)
    connector._pd_execute_send_instructions(
        SimpleNamespace(pd_send_instructions=[instr])
    )
    connector._pd_engine.send_proxy_notification.assert_not_called()


def test_worker_metadata_aggregate_merges_pd_received():
    stack = _import_stack()
    a = stack.pd_engine_mod.MPPDWorkerMetadata(
        completed_store_requests={"r1": 1},
        pd_received={"r1": 512},
    )
    b = stack.pd_engine_mod.MPPDWorkerMetadata(
        completed_store_requests={"r1": 1},
        pd_received={"r2": 700},
    )
    merged = a.aggregate(b)
    assert isinstance(merged, stack.pd_engine_mod.MPPDWorkerMetadata)
    assert merged.pd_received == {"r1": 512, "r2": 700}
    assert merged.completed_store_requests == {"r1": 2}


def test_update_connector_output_records_pd_into_ledger():
    connector, stack = _make_scheduler_connector()
    ledger = connector._ledger_for("money")
    ledger.update_coverage(apc_tokens=512, l1_tokens=1024, pd_tokens=0)

    connector_output = SimpleNamespace(
        kv_cache_events=None,
        kv_connector_worker_meta=stack.pd_engine_mod.MPPDWorkerMetadata(
            completed_store_requests={},
            pd_received={"money": 1536},
        ),
    )
    connector.update_connector_output(connector_output)

    assert ledger.pd_tokens == 1536
    kv = stack.kv_source
    assert [(s.start, s.end, s.source) for s in ledger.segments] == [
        (0, 512, kv.APC),
        (512, 1024, kv.L1),
        (1024, 1536, kv.PD_IN),
    ]
    assert [(s.start, s.end) for s in ledger.pd_receive_segments()] == [
        (1024, 1536)
    ]


def _enable_pd_merge(connector, arrived):
    """Install the fake key db and arrived-key set for merge tests."""
    connector._pd_key_db = _FakeKeyDB(chunk_size=256)
    connector._pd_chunk_size = 256
    connector._pd_arrived = set(arrived)


def test_merged_lookup_pd_only_hit():
    """L1 miss + full prompt arrived: external = prompt - 1, async."""
    connector, stack = _make_scheduler_connector(lookup_results=(0,))
    _enable_pd_merge(connector, {"key-0", "key-1", "key-2"})
    request = _fake_request("pd-only", 700)
    result = connector.get_num_new_matched_tokens(request, 0)
    assert result == (699, True)
    ledger = connector._ledgers["pd-only"]
    kv = stack.kv_source
    assert [(s.start, s.end, s.source) for s in ledger.segments] == [
        (0, 700, kv.PD_IN)
    ]


def test_merged_lookup_round2_money_path():
    """APC=512 | L1=1024 | PD arrived [1024,1536): three-way merge."""
    connector, stack = _make_scheduler_connector(
        lookup_results=(4 * 256,)
    )
    _enable_pd_merge(connector, {"key-4", "key-5"})
    request = _fake_request("money", 2048)
    result = connector.get_num_new_matched_tokens(request, 512)
    assert result == (1024, True)
    ledger = connector._ledgers["money"]
    kv = stack.kv_source
    assert [(s.start, s.end, s.source) for s in ledger.segments] == [
        (0, 512, kv.APC),
        (512, 1024, kv.L1),
        (1024, 1536, kv.PD_IN),
    ]
    assert [(s.start, s.end) for s in ledger.pd_receive_segments()] == [
        (1024, 1536)
    ]


def test_merged_lookup_gap_degrades_to_recompute():
    """Producer-skip hole with no L1 cover: union walk stops at 0."""
    connector, _ = _make_scheduler_connector(lookup_results=(0,))
    _enable_pd_merge(connector, {"key-2", "key-3"})
    request = _fake_request("gap", 1024)
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)


def test_merged_lookup_pending_passes_none_through():
    connector, _ = _make_scheduler_connector(lookup_results=(None,))
    _enable_pd_merge(connector, {"key-0"})
    request = _fake_request("pend", 700)
    result = connector.get_num_new_matched_tokens(request, 0)
    assert result[0] is None


def test_merged_lookup_matches_arrived_without_spec():
    """The disagg proxy strips kv_transfer_params from decode requests,
    so arrived chunks must match by content with no spec attached."""
    connector, stack = _make_scheduler_connector(lookup_results=(0,))
    _enable_pd_merge(connector, {"key-0", "key-1", "key-2"})
    request = _fake_request("nospec", 700, with_spec=False)
    assert connector.get_num_new_matched_tokens(request, 0) == (699, True)
    ledger = connector._ledgers["nospec"]
    kv = stack.kv_source
    assert [(s.start, s.end, s.source) for s in ledger.segments] == [
        (0, 700, kv.PD_IN)
    ]


def test_pd_build_load_instructions_once():
    connector, _ = _make_scheduler_connector(lookup_results=(4 * 256,))
    _enable_pd_merge(connector, {"key-4", "key-5"})
    request = _fake_request("money", 2048)
    connector.get_num_new_matched_tokens(request, 512)
    tracker = connector.request_trackers["money"]
    tracker.num_scheduled_tokens = 1536
    tracker.allocated_block_ids = {0: list(range(96))}
    instructions = connector._pd_build_load_instructions()
    assert len(instructions) == 1
    assert (instructions[0].start, instructions[0].end) == (1024, 1536)
    assert connector._pd_build_load_instructions() == []
    connector.request_finished(request, [])
    assert "money" not in connector._pd_load_issued


def test_worker_metadata_aggregates_arrived_keys():
    stack = _import_stack()
    a = stack.pd_engine_mod.MPPDWorkerMetadata(
        completed_store_requests={},
        pd_arrived_keys=["k1"],
    )
    b = stack.pd_engine_mod.MPPDWorkerMetadata(
        completed_store_requests={},
        pd_arrived_keys=["k2"],
    )
    assert a.aggregate(b).pd_arrived_keys == ["k1", "k2"]


def test_update_connector_output_absorbs_arrived_keys():
    connector, stack = _make_scheduler_connector(lookup_results=(0,))
    meta = stack.pd_engine_mod.MPPDWorkerMetadata(
        completed_store_requests={},
        pd_arrived_keys=["key-4", "key-5"],
    )
    connector.update_connector_output(
        SimpleNamespace(
            kv_cache_events=None, kv_connector_worker_meta=meta
        )
    )
    assert connector._pd_arrived == {"key-4", "key-5"}


def test_save_policy_validation():
    stack = _import_stack()

    # Third Party
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    with pytest.raises(ValueError):
        stack.pd_engine.maybe_create(
            SimpleNamespace(
                kv_transfer_config=SimpleNamespace(
                    kv_connector_extra_config={
                        **_PD_EXTRA,
                        "pd_save_policy": "nope",
                    },
                    kv_role="kv_producer",
                ),
                cache_config=SimpleNamespace(cache_dtype="bfloat16"),
                model_config=SimpleNamespace(),
                parallel_config=SimpleNamespace(rank=0, world_size=1),
            ),
            KVConnectorRole.WORKER,
        )


def test_l1_only_policy_skips_send_instructions():
    connector, _ = _make_scheduler_connector()
    connector._pd_save_policy = "l1_only"
    request = _fake_request()
    connector.request_trackers["r1"] = _real_tracker(request)
    connector.update_state_after_alloc(request, _blocks_ns(), 0)
    tracker = connector.request_trackers["r1"]
    tracker.num_scheduled_tokens = 700
    tracker.allocated_block_ids = {0: list(range(44))}
    assert connector._pd_build_send_instructions() == []


def test_pd_only_policy_suppresses_l1_stores_and_restores():
    connector, _ = _make_scheduler_connector()
    connector._pd_save_policy = "pd_only"
    connector._pd_engine = SimpleNamespace(pd_role="sender")
    store_r1 = SimpleNamespace(
        request_id="r1", direction="STORE", op=SimpleNamespace(token_ids=[1])
    )
    store_r2 = SimpleNamespace(
        request_id="r2", direction="STORE", op=SimpleNamespace(token_ids=[2])
    )
    metadata = SimpleNamespace(
        requests=[store_r1, store_r2],
        pd_send_instructions=[
            SimpleNamespace(req_id="r1", start=0, end=256)
        ],
    )
    connector._stub_bound_metadata = metadata

    original_get = connector._get_connector_metadata

    def _get():
        return connector._stub_bound_metadata

    connector._get_connector_metadata = _get
    try:
        connector._pd_maybe_suppress_l1_stores()
        assert [m.request_id for m in metadata.requests] == ["r2"]
        connector._pd_restore_l1_stores()
        assert len(metadata.requests) == 2
    finally:
        connector._get_connector_metadata = original_get


def test_request_finished_returns_first_tok_for_disagg_prefill():
    """The disagg proxy reads first_tok from the prefill response's
    kv_transfer_params; the connector must supply it for disagg-routed
    producer requests (the prefill generates exactly the first token)."""
    stack = _import_stack()
    connector, _ = _make_scheduler_connector()
    connector._vllm_config.kv_transfer_config.kv_role = "kv_producer"
    spec = stack.pd_engine_mod.extract_pd_transfer_spec(_fake_request())
    connector._pd_specs["r1"] = spec

    request = SimpleNamespace(request_id="r1", output_token_ids=[4242])

    # Third Party
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector

    with patch.object(
        LMCacheMPConnector, "request_finished", return_value=(True, None)
    ):
        save, params = connector.request_finished(request, [])

    assert save is True
    assert params["first_tok"] == 4242
    assert "r1" not in connector._pd_specs


def test_request_finished_skips_first_tok_without_disagg_spec():
    stack = _import_stack()
    connector, _ = _make_scheduler_connector()
    connector._vllm_config.kv_transfer_config.kv_role = "kv_producer"

    request = SimpleNamespace(request_id="r9", output_token_ids=[7])

    # Third Party
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector

    with patch.object(
        LMCacheMPConnector, "request_finished", return_value=(True, None)
    ):
        _, params = connector.request_finished(request, [])

    assert params is None


def test_request_finished_skips_first_tok_on_consumer():
    stack = _import_stack()
    connector, _ = _make_scheduler_connector()
    connector._vllm_config.kv_transfer_config.kv_role = "kv_consumer"
    spec = stack.pd_engine_mod.extract_pd_transfer_spec(_fake_request())
    connector._pd_specs["r1"] = spec

    request = SimpleNamespace(request_id="r1", output_token_ids=[4242])

    # Third Party
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector

    with patch.object(
        LMCacheMPConnector, "request_finished", return_value=(True, None)
    ):
        _, params = connector.request_finished(request, [])

    assert params is None
