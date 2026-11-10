# SPDX-License-Identifier: Apache-2.0
"""Step 3 acceptance checks: ledger sync on the MP-PD connector.

The connector inherits the upstream ``LMCacheMPConnector`` (L1 leg,
import reuse). These tests pin down the only behaviour it adds in
Step 3: every lookup measurement is mirrored into the per-request
ledger as the APC / L1 coverage map, and the ledger is dropped when
the request finishes.

The connector is built via ``object.__new__`` (the repo pattern, see
test_handle_preemptions.py) because the real ``__init__`` performs a
live MP-server handshake; the scheduler adapter is faked so the full
upstream lookup pipeline runs against canned results.
"""

# Standard
from types import SimpleNamespace
import importlib

# Third Party
import pytest

_TOKENS_PER_CHUNK = 256
_HIT_ALIGNMENT = 256


def _import_stack():
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    pytest.importorskip("lmcache_ascend")

    # Third Party
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
    from vllm.v1.request import RequestStatus

    mp_pd_entry = importlib.import_module(
        "lmcache_ascend.integration.vllm.lmcache_ascend_mp_pd_connector"
    )

    # First Party
    from lmcache_ascend.v1.mp_pd import KVSource, Segment

    return SimpleNamespace(
        base_cls=LMCacheMPConnector,
        conn_cls=mp_pd_entry.LMCacheAscendMPPDConnectorV1Dynamic,
        kv_source=KVSource,
        segment=Segment,
        request_status=RequestStatus,
    )


class _FakeSchedulerAdapter:
    """Canned lookup results; the rest of the seam is no-op."""

    def __init__(self, results):
        self.results = list(results)
        self.lmcache_tokens_per_chunk = _TOKENS_PER_CHUNK

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


class _FakeRequest:
    def __init__(self, request_id="req-1", num_tokens=4096):
        self.request_id = request_id
        self.all_token_ids = list(range(num_tokens))
        self.num_prompt_tokens = num_tokens
        self.prompt_token_ids = list(range(num_tokens))
        self.cache_salt = ""
        self.status = _import_stack().request_status.FINISHED_STOPPED


def _make_connector(lookup_results):
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
    connector.scheduler_adapter = _FakeSchedulerAdapter(lookup_results)
    connector._hit_alignment_tokens = _HIT_ALIGNMENT
    connector._reserve_last_token_for_lookup = False
    connector._connector_stats = SimpleNamespace(
        record_lookup=lambda dt: None
    )
    connector.lazy_offload = False
    connector._can_store = False
    connector._kv_cache_events = []
    return connector


def test_connector_inherits_upstream_mp_connector():
    stack = _import_stack()
    assert issubclass(stack.conn_cls, stack.base_cls)


def test_pending_lookup_records_zero_l1():
    connector = _make_connector(lookup_results=[None])
    request = _FakeRequest()

    result = connector.get_num_new_matched_tokens(request, 0)

    assert result == (None, True)
    ledger = connector._ledgers[request.request_id]
    assert (ledger.apc_tokens, ledger.l1_tokens, ledger.pd_tokens) == (
        0,
        0,
        0,
    )


def test_l1_hit_records_full_ledger_coverage():
    stack = _import_stack()
    connector = _make_connector(lookup_results=[4 * _TOKENS_PER_CHUNK])
    request = _FakeRequest()

    result = connector.get_num_new_matched_tokens(request, 0)

    assert result == (4 * _TOKENS_PER_CHUNK, True)
    ledger = connector._ledgers[request.request_id]
    assert ledger.l1_tokens == 4 * _TOKENS_PER_CHUNK
    assert ledger.external_tokens == 4 * _TOKENS_PER_CHUNK
    assert ledger.segments == [
        stack.segment(0, 4 * _TOKENS_PER_CHUNK, stack.kv_source.L1)
    ]


def test_round2_money_path_splits_apc_and_l1():
    """APC=512, L1=1024: ledger must own the disjoint two-segment map."""
    stack = _import_stack()
    connector = _make_connector(lookup_results=[4 * _TOKENS_PER_CHUNK])
    request = _FakeRequest()

    result = connector.get_num_new_matched_tokens(request, 2 * _HIT_ALIGNMENT)

    assert result == (4 * _TOKENS_PER_CHUNK - 2 * _HIT_ALIGNMENT, True)
    ledger = connector._ledgers[request.request_id]
    assert (ledger.apc_tokens, ledger.l1_tokens) == (
        2 * _HIT_ALIGNMENT,
        4 * _TOKENS_PER_CHUNK,
    )
    assert ledger.segments == [
        stack.segment(0, 2 * _HIT_ALIGNMENT, stack.kv_source.APC),
        stack.segment(2 * _HIT_ALIGNMENT, 4 * _TOKENS_PER_CHUNK, stack.kv_source.L1),
    ]


def test_l1_miss_keeps_ledger_at_zero():
    connector = _make_connector(lookup_results=[0])
    request = _FakeRequest()

    result = connector.get_num_new_matched_tokens(request, 0)

    assert result == (0, False)
    ledger = connector._ledgers[request.request_id]
    assert ledger.segments == []
    assert ledger.external_tokens == 0


def test_request_finished_drops_ledger():
    connector = _make_connector(lookup_results=[4 * _TOKENS_PER_CHUNK])
    request = _FakeRequest()
    connector.get_num_new_matched_tokens(request, 0)
    assert connector._ledgers

    result = connector.request_finished(request, [1, 2, 3])

    assert result == (False, None)
    assert connector._ledgers == {}


def test_ledger_sync_is_idempotent_across_polls():
    connector = _make_connector(
        lookup_results=[4 * _TOKENS_PER_CHUNK, 4 * _TOKENS_PER_CHUNK]
    )
    request = _FakeRequest()

    first = connector.get_num_new_matched_tokens(request, 0)
    second = connector.get_num_new_matched_tokens(request, 0)

    assert first == second
    ledger = connector._ledgers[request.request_id]
    assert ledger.l1_tokens == 4 * _TOKENS_PER_CHUNK
    assert len(connector._ledgers) == 1
