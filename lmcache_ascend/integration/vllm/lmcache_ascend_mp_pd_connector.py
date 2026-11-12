# SPDX-License-Identifier: Apache-2.0
"""Dynamic registration entry for the monolithic MP+PD connector.

Usage (``--kv-transfer-config``)::

    {
      "kv_connector": "LMCacheAscendMPPDConnectorV1Dynamic",
      "kv_role": "kv_both",
      "kv_connector_module_path":
          "lmcache_ascend.integration.vllm.lmcache_ascend_mp_pd_connector"
    }

vLLM resolves the class by exact ``getattr()`` on
``kv_connector_module_path``, so the class name must match
``kv_connector`` exactly (see examples/disagg_prefill/hccs/README.md).

Step status:

* Step 3: the L1 leg is the upstream ``LMCacheMPConnector``
  (import reuse, no fork -- key design decision 1). Every MP-cache
  hook -- ZMQ lookup, RETRIEVE (H2D), STORE (D2H), lookup locks,
  preemption handling -- is inherited from upstream. Lookup results
  are mirrored into the per-request ledger
  (``lmcache_ascend.v1.mp_pd``) as the APC / L1 coverage, so the
  segmented-source decisions of Step 5 can be driven from it.
* Step 4 (current): the PD leg rides alongside as an explicitly
  assembled ``AscendPDBackend`` capability surface (key design
  decision 2 -- see ``pd_engine.py``). The P side pushes the
  chunk-aligned range computed each step (wait_for_save); the D side
  observes how much of the prompt has arrived in the receiver buffer
  and reports it to the scheduler, where it fills the ledger's
  ``pd_tokens``. Scheduling semantics are still upstream's: the PD
  observation is pure bookkeeping until the Step-5 merge, so D keeps
  recomputing what it has not loaded (correct, wasteful-by-design).
* Step 5 will replace the scheduler-side lookup pipeline with the
  ledger-driven APC -> L1 -> PD_IN merge and the mixed load path.
"""

# Standard
from typing import TYPE_CHECKING, Any, Optional

# Third Party
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPWorkerMetadata
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.logger import init_logger
from vllm.v1.outputs import KVConnectorOutput

# First Party
from lmcache_ascend.integration.vllm.pd_engine import (
    MPPDWorkerMetadata,
    PDEngine,
    PDLoadInstruction,
    PDSendInstruction,
    PDTransferSpec,
    build_pd_chunk_db,
    extract_pd_transfer_spec,
)
from lmcache_ascend.v1.mp_pd import RequestLedger
import lmcache_ascend  # noqa: F401

if TYPE_CHECKING:
    # Third Party
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


class LMCacheAscendMPPDConnectorV1Dynamic(LMCacheMPConnector):
    """Monolithic connector combining MP cache and PD transfer.

    One connector serves both roles that MultiConnector would
    otherwise assign to competing children: L1 retrieval from the
    standalone lmcache server and PD transfer over hixl/hccl, unified
    by the per-request ledger. The ``kv_role`` producer/consumer
    semantics still follow vLLM's ``kv-transfer-config``.

    Step 3/4 overrides stay observational wherever possible: L1
    measurements and PD arrivals are mirrored into the ledger, and the
    PD engine moves data without changing any scheduling decision.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional[Any] = None,
    ) -> None:
        if kv_cache_config is not None:
            super().__init__(vllm_config, role, kv_cache_config)
        else:
            super().__init__(vllm_config, role)
        # Scheduler-side per-request ledger (APC / L1 / PD_IN coverage
        # map). Created on both roles for attribute symmetry; only the
        # scheduler half is ever written by the lookup pipeline.
        self._ledgers: dict[str, RequestLedger] = {}
        # Scheduler-side 发货单 bookkeeping (request_id -> spec).
        self._pd_specs: dict[str, PDTransferSpec] = {}
        # Worker-side PD engine (None on the scheduler half and when PD
        # is not configured).
        self._pd_engine: Optional[PDEngine] = PDEngine.maybe_create(
            vllm_config, role
        )
        # Worker-side receive observations awaiting the next
        # build_connector_worker_meta call.
        self._pd_received: dict[str, int] = {}
        self._pd_arrived_pending: list = []
        self._pd_complete_logged: set[str] = set()
        self._pd_kv_caches: Optional[dict] = None
        # Step 6: producer-side save policy (both / pd_only / l1_only).
        kv_transfer_config = vllm_config.kv_transfer_config
        extra = (
            kv_transfer_config.kv_connector_extra_config or {}
            if kv_transfer_config is not None
            else {}
        )
        self._pd_save_policy = str(extra.get("pd_save_policy", "both")).lower()
        if self._pd_save_policy not in PDEngine.VALID_SAVE_POLICIES:
            raise ValueError(
                "pd_save_policy must be one of "
                f"{PDEngine.VALID_SAVE_POLICIES}, got "
                f"{extra.get('pd_save_policy')!r}"
            )
        # Step 5: scheduler-side PD union-walk state.
        self._pd_arrived: set = set()
        self._pd_key_db = None
        self._pd_chunk_size = 0
        self._pd_load_issued: set[str] = set()
        # Step 5: worker-side PD load completion / failure state.
        self._pd_load_done: set[str] = set()
        self._pd_failed_blocks: set[int] = set()
        if role == KVConnectorRole.SCHEDULER:
            kv_transfer_config = vllm_config.kv_transfer_config
            extra = (
                kv_transfer_config.kv_connector_extra_config or {}
                if kv_transfer_config is not None
                else {}
            )
            if "pd_buffer_size" in extra:
                self._pd_key_db, self._pd_chunk_size = build_pd_chunk_db(
                    vllm_config, extra
                )
                mp_chunk = self.scheduler_adapter.lmcache_tokens_per_chunk
                if (
                    self._pd_chunk_size <= 0
                    or mp_chunk % self._pd_chunk_size != 0
                ):
                    raise ValueError(
                        f"PD chunk_size ({self._pd_chunk_size}) must "
                        f"divide the LMCache MP chunk size ({mp_chunk}); "
                        "a mismatch misaligns PD chunk keys with the L1 "
                        "lookup boundaries."
                    )

    # ==============================
    # Worker-side methods
    # ==============================

    def register_kv_caches(self, kv_caches: dict):
        super().register_kv_caches(kv_caches)
        if self._pd_engine is not None:
            # vLLM hands a {layer: tensor} dict; the NPU connector expects
            # the per-layer tensor list (the same conversion the
            # in-process line performs in vllm_v1_adapter.py).
            self._pd_kv_caches = list(kv_caches.values())
            self._pd_engine.assemble(kv_caches)

    def wait_for_save(self):
        self._pd_maybe_suppress_l1_stores()
        try:
            super().wait_for_save()
        finally:
            self._pd_restore_l1_stores()
        if self._pd_engine is None or self._pd_kv_caches is None:
            return
        self._pd_execute_send_instructions(self._get_connector_metadata())

    def _pd_execute_send_instructions(self, metadata) -> None:
        for instr in getattr(metadata, "pd_send_instructions", None) or []:
            try:
                if getattr(instr, "notify_only", False):
                    self._pd_engine.send_proxy_notification(instr.req_id)
                    continue
                self._pd_engine.execute_send(instr, self._pd_kv_caches)
            except Exception:
                logger.exception(
                    "PD send failed for req %s [%d,%d); the receiver will "
                    "recompute the missing range.",
                    instr.req_id,
                    instr.start,
                    instr.end,
                )
                if instr.is_last_prefill:
                    # The proxy gates the decode request on this
                    # notification; without it the client would hang even
                    # though the receiver would have recomputed the
                    # missing range. Open the gate no matter what.
                    try:
                        self._pd_engine.send_proxy_notification(instr.req_id)
                    except Exception:
                        logger.exception(
                            "Proxy notification for req %s failed; the "
                            "disagg proxy will keep waiting.",
                            instr.req_id,
                        )

    def _pd_maybe_suppress_l1_stores(self) -> None:
        """pd_only policy: skip the L1 store for PD-routed requests.

        Only the producer suppresses: its D2H store bandwidth is what
        the policy saves. The decoder keeps storing to fill the
        warehouse. The suppression happens worker-side (the tracker's
        store bookkeeping still advances, so no stale chunks are
        re-emitted later).
        """
        engine = self._pd_engine
        if (
            engine is None
            or engine.pd_role != "sender"
            or self._pd_save_policy != "pd_only"
        ):
            return
        metadata = self._get_connector_metadata()
        pd_reqs = {
            instr.req_id
            for instr in getattr(metadata, "pd_send_instructions", None)
            or []
        }
        if not pd_reqs:
            return
        original = metadata.requests
        metadata.requests = [
            m
            for m in original
            if not (
                getattr(m, "direction", None) == "STORE"
                and m.request_id in pd_reqs
            )
        ]
        self._pd_suppressed_requests = original
        logger.debug(
            "pd_only: suppressed %d L1 store ops for PD-routed requests",
            len(original) - len(metadata.requests),
        )

    def _pd_restore_l1_stores(self) -> None:
        original = getattr(self, "_pd_suppressed_requests", None)
        if original is None:
            return
        self._get_connector_metadata().requests = original
        self._pd_suppressed_requests = None

    def start_load_kv(self, forward_context, **kwargs: Any) -> None:
        super().start_load_kv(forward_context, **kwargs)
        engine = self._pd_engine
        if engine is None or engine.pd_role != "receiver":
            return
        self._pd_arrived_pending.extend(engine.drain_arrived_keys())
        metadata = self._get_connector_metadata()
        for req_meta in getattr(metadata, "requests", None) or []:
            req_id = req_meta.request_id
            if req_id in self._pd_complete_logged:
                continue
            token_ids = getattr(req_meta.op, "token_ids", None)
            if not token_ids:
                continue
            received = engine.received_prefix_tokens(token_ids)
            if received > 0:
                self._pd_received[req_id] = received
            if received >= engine.expected_prefix_tokens(len(token_ids)):
                logger.info(
                    "PD transfer complete for req %s (%d tokens arrived)",
                    req_id,
                    received,
                )
                self._pd_complete_logged.add(req_id)
                self._pd_received[req_id] = received
        kvcaches = self._pd_kv_caches
        if kvcaches is None:
            return
        for instr in getattr(metadata, "pd_load_instructions", None) or []:
            try:
                failed = engine.execute_pd_load(instr, kvcaches)
                if failed:
                    self._pd_failed_blocks.update(failed)
                self._pd_load_done.add(instr.request_id)
            except Exception:
                logger.exception(
                    "PD load failed for req %s [%d,%d); marking its "
                    "blocks for recompute",
                    instr.request_id,
                    instr.start,
                    instr.end,
                )
                self._pd_load_done.add(instr.request_id)
                for pos in range(instr.start, instr.end):
                    idx = pos // instr.block_size
                    if 0 <= idx < len(instr.block_ids):
                        self._pd_failed_blocks.add(instr.block_ids[idx])

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        sending, recving = super().get_finished(finished_req_ids)
        if self._pd_load_done:
            done = self._pd_load_done
            self._pd_load_done = set()
            recving = set(recving or set()) | done
        return sending, recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        errors = super().get_block_ids_with_load_errors()
        if self._pd_failed_blocks:
            errors = set(errors or set()) | self._pd_failed_blocks
            self._pd_failed_blocks = set()
        return errors

    def build_connector_worker_meta(self):
        metadata = super().build_connector_worker_meta()
        engine = self._pd_engine
        if engine is None or engine.pd_role != "receiver":
            return metadata
        if (
            metadata is None
            and not self._pd_received
            and not self._pd_arrived_pending
        ):
            return None
        base = metadata
        if base is None:
            base = LMCacheMPWorkerMetadata(
                completed_store_requests={}, failed_store_requests=set()
            )
        out = MPPDWorkerMetadata(
            completed_store_requests=base.completed_store_requests,
            failed_store_requests=base.failed_store_requests,
            kv_events=getattr(base, "kv_events", None),
        )
        out.pd_received = self._pd_received
        out.pd_arrived_keys = self._pd_arrived_pending
        self._pd_received = {}
        self._pd_arrived_pending = []
        return out

    def shutdown(self):
        result = super().shutdown()
        if self._pd_engine is not None:
            self._pd_engine.close()
        return result

    # ==============================
    # Scheduler-side methods
    # ==============================

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        super().update_state_after_alloc(
            request, blocks, num_external_tokens
        )
        spec = extract_pd_transfer_spec(request)
        if spec is not None:
            self._pd_specs[request.request_id] = spec

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> "KVConnectorMetadata":
        metadata = super().build_connector_meta(scheduler_output)
        kv_role = getattr(
            getattr(self._vllm_config, "kv_transfer_config", None),
            "kv_role",
            None,
        )
        if kv_role == "kv_producer":
            instructions = self._pd_build_send_instructions()
            if instructions:
                metadata.pd_send_instructions = instructions
        else:
            load_instructions = self._pd_build_load_instructions()
            if load_instructions:
                metadata.pd_load_instructions = load_instructions
        return metadata

    def update_connector_output(self, connector_output: KVConnectorOutput):
        super().update_connector_output(connector_output)
        meta = getattr(connector_output, "kv_connector_worker_meta", None)
        if not isinstance(meta, MPPDWorkerMetadata):
            return
        if meta.pd_arrived_keys:
            self._pd_arrived.update(meta.pd_arrived_keys)
        if not meta.pd_received:
            return
        for req_id, pd_tokens in meta.pd_received.items():
            ledger = self._ledger_for(req_id)
            ledger.update_coverage(
                apc_tokens=ledger.apc_tokens,
                l1_tokens=ledger.l1_tokens,
                pd_tokens=pd_tokens,
            )
            logger.debug(
                "MP-PD ledger after PD report: %s", ledger.describe()
            )

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        # The 发货单 may first appear here, before any allocation.
        if request.request_id not in self._pd_specs:
            spec = extract_pd_transfer_spec(request)
            if spec is not None:
                self._pd_specs[request.request_id] = spec
        result = super().get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        if self._pd_key_db is None or result[0] is None:
            # PD not configured, or the L1 lookup is still pending:
            # upstream semantics unchanged (None retried by vLLM).
            self._sync_ledger_from_tracker(request, num_computed_tokens)
            return result
        trackers = getattr(self, "request_trackers", None) or {}
        tracker = trackers.get(request.request_id)
        l1_tokens = tracker.num_lmcache_hit_tokens if tracker else 0
        pd_tokens = self._pd_union_prefix(request, l1_tokens)
        ledger = self._ledger_for(request.request_id)
        ledger.update_coverage(num_computed_tokens, l1_tokens, pd_tokens)
        external = ledger.external_tokens
        # Mirror upstream's full-prompt-hit adjustment: the last token
        # is always recomputed so num_new_tokens stays positive.
        if max(l1_tokens, pd_tokens) >= len(request.all_token_ids):
            external = max(0, external - 1)
        logger.debug(
            "MP-PD merged lookup: %s -> external=%d",
            ledger.describe(),
            external,
        )
        return external, external > 0

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        spec = self._pd_specs.get(request.request_id)
        result = super().request_finished(request, block_ids)
        self._ledgers.pop(request.request_id, None)
        self._pd_specs.pop(request.request_id, None)
        self._pd_load_issued.discard(request.request_id)
        # Disagg proxy handshake: the proxy reads first_tok from the
        # prefill response's kv_transfer_params to build the decode
        # request (prefill generates the first token, decode continues
        # from the second). vLLM natively echoes the connector's
        # request_finished params into the response, but the MP
        # connector super-chain never produces first_tok — fill it here
        # for disagg-routed prefill requests.
        kv_role = getattr(
            getattr(self._vllm_config, "kv_transfer_config", None),
            "kv_role",
            None,
        )
        if (
            spec is not None
            and kv_role == "kv_producer"
            and isinstance(result, tuple)
        ):
            output_ids = getattr(request, "output_token_ids", None) or []
            if output_ids:
                params = dict(result[1] or {})
                params.setdefault("first_tok", output_ids[0])
                result = (result[0], params)
        return result

    # ==============================
    # Helpers
    # ==============================

    def _ledger_for(self, request_id: str) -> RequestLedger:
        ledger = self._ledgers.get(request_id)
        if ledger is None:
            ledger = RequestLedger(request_id=request_id)
            self._ledgers[request_id] = ledger
        return ledger

    def _pd_build_send_instructions(self) -> list:
        """Compute this step's PD send ranges (发货单 execution orders).

        The covered reach is the L1-loaded prefix plus the newly
        computed tokens (both live in the paged pool, so both can be
        shipped): without the L1 part, a producer-side cache hit would
        leave the decoder with a hole it can only fill by recomputing.

        When the prefill phase is over but nothing is left to ship
        (e.g. everything was already sent), a notify-only instruction
        is emitted so the proxy gate still opens; without it the client
        would hang forever even though the decoder would have
        recomputed the missing range.
        """
        if not self._pd_specs:
            return []
        if self._pd_save_policy == "l1_only":
            return []
        trackers = getattr(self, "request_trackers", None) or {}
        chunk = self.scheduler_adapter.lmcache_tokens_per_chunk
        block_size = self._vllm_config.cache_config.block_size

        def _notify_only(spec, tracker, block_size):
            return PDSendInstruction(
                req_id=spec.req_id,
                receiver_host=spec.receiver_host,
                receiver_init_port=spec.receiver_init_port,
                receiver_alloc_port=spec.receiver_alloc_port,
                token_ids=tracker.get_token_ids(),
                block_ids=[],
                block_size=block_size,
                start=spec.sent_end,
                end=spec.sent_end,
                is_last_prefill=True,
                notify_only=True,
            )

        instructions: list = []
        for req_id, spec in list(self._pd_specs.items()):
            tracker = trackers.get(req_id)
            if tracker is None:
                continue
            handled = min(
                tracker.num_lmcache_hit_tokens + tracker.num_scheduled_tokens,
                spec.num_prompt_tokens,
            )
            if handled <= spec.sent_end:
                if (
                    handled >= spec.num_prompt_tokens
                    and not spec.last_notified
                ):
                    instructions.append(
                        _notify_only(spec, tracker, block_size)
                    )
                    spec.last_notified = True
                continue
            if handled < spec.num_prompt_tokens:
                end = handled // chunk * chunk
            else:
                end = spec.num_prompt_tokens
            block_ids = self._flatten_single_group_block_ids(
                tracker.allocated_block_ids
            )
            if block_ids is None:
                logger.warning(
                    "PD leg: req %s has %d kv cache groups; only single "
                    "group is supported, PD send skipped",
                    req_id,
                    len(tracker.allocated_block_ids),
                )
                if (
                    handled >= spec.num_prompt_tokens
                    and not spec.last_notified
                ):
                    instructions.append(
                        _notify_only(spec, tracker, block_size)
                    )
                    spec.last_notified = True
                continue
            end = min(end, len(block_ids) * block_size)
            if end <= spec.sent_end:
                if (
                    handled >= spec.num_prompt_tokens
                    and not spec.last_notified
                ):
                    instructions.append(
                        _notify_only(spec, tracker, block_size)
                    )
                    spec.last_notified = True
                continue
            is_last = end >= spec.num_prompt_tokens
            instructions.append(
                PDSendInstruction(
                    req_id=spec.req_id,
                    receiver_host=spec.receiver_host,
                    receiver_init_port=spec.receiver_init_port,
                    receiver_alloc_port=spec.receiver_alloc_port,
                    token_ids=tracker.get_token_ids(),
                    block_ids=block_ids,
                    block_size=block_size,
                    start=spec.sent_end,
                    end=end,
                    is_last_prefill=is_last,
                )
            )
            spec.sent_end = end
            if is_last:
                spec.last_notified = True
        return instructions

    @staticmethod
    def _flatten_single_group_block_ids(allocated: dict) -> Optional[list]:
        groups = list(allocated.values())
        if len(groups) != 1:
            return None
        return list(groups[0])

    def _pd_union_prefix(self, request: "Request", l1_tokens: int) -> int:
        """How far the union of (L1 hit + arrived PD chunks) reaches.

        The producer only sends chunks beyond its own skip, so the PD
        buffer is generally NOT a prefix from token 0. The walk starts
        at the L1 reach (which covers the hole below the producer's
        skip) and counts consecutive arrived chunks; the ledger's
        ``pd_tokens`` is then this union end, which makes the PD_IN
        segment exactly the arrived range beyond L1.

        Spec-free by design: the disagg proxy strips
        ``kv_transfer_params`` from the decode request, so arrivals are
        matched by content — the chunk keys derived from this request's
        own tokens are exactly the keys the producer transferred.
        """
        if self._pd_key_db is None:
            return 0
        token_ids = request.all_token_ids
        chunk = self._pd_chunk_size
        pos = l1_tokens // chunk * chunk
        reached = pos
        for start, end, key in self._pd_key_db.process_tokens(
            tokens=token_ids
        ):
            if start < pos:
                continue
            key_str = key if isinstance(key, str) else key.to_string()
            if key_str in self._pd_arrived:
                reached = end
            else:
                break
        return reached

    def _pd_build_load_instructions(self) -> list:
        """Worker orders to land PD-arrived ranges into the paged pool.

        Emitted once per request, for the ledger's PD_IN segment beyond
        the L1 reach, using the tracker's allocated blocks. A chunk that
        went missing by load time surfaces as a load error and vLLM
        recomputes it, so no retry is issued here.

        Spec-free by design (see ``_pd_union_prefix``): the decode
        request never carries ``disagg_spec`` — ledgers with a PD_IN
        segment are the driver.
        """
        if self._pd_key_db is None:
            return []
        trackers = getattr(self, "request_trackers", None) or {}
        block_size = self._vllm_config.cache_config.block_size
        instructions: list = []
        for req_id, ledger in self._ledgers.items():
            if req_id in self._pd_load_issued:
                continue
            segments = ledger.pd_receive_segments()
            if not segments:
                continue
            tracker = trackers.get(req_id)
            if tracker is None:
                continue
            block_ids = self._flatten_single_group_block_ids(
                tracker.allocated_block_ids
            )
            if block_ids is None:
                continue
            end = segments[-1].end
            start = segments[0].start
            if end > len(block_ids) * block_size:
                continue
            instructions.append(
                PDLoadInstruction(
                    request_id=req_id,
                    token_ids=tracker.get_token_ids(),
                    block_ids=block_ids,
                    block_size=block_size,
                    start=start,
                    end=end,
                )
            )
            self._pd_load_issued.add(req_id)
        return instructions

    def _sync_ledger_from_tracker(
        self, request: "Request", num_computed_tokens: int
    ) -> None:
        """Mirror the latest lookup measurement into the request ledger.

        The upstream tracker already holds exactly the two prefix
        coverages the ledger needs: ``num_vllm_hit_tokens`` is the
        APC reach (alignment-floored) and ``num_lmcache_hit_tokens``
        is the L1 lookup reach. ``pd_tokens`` is filled separately by
        the worker's PD receive report (update_connector_output).

        A pending lookup (upstream returned ``None``) always finds the
        tracker with a zeroed L1 reach -- every upstream path that
        returns ``None`` either has not resolved the lookup yet or has
        just reset the tracker for a re-lookup -- so syncing on every
        call is safe and idempotent.
        """
        trackers = getattr(self, "request_trackers", None)
        if trackers is None:
            return
        tracker = trackers.get(request.request_id)
        if tracker is None:
            return
        ledger = self._ledger_for(request.request_id)
        ledger.update_coverage(
            apc_tokens=num_computed_tokens,
            l1_tokens=tracker.num_lmcache_hit_tokens,
            pd_tokens=ledger.pd_tokens,
        )
        logger.debug("MP-PD ledger after lookup: %s", ledger.describe())
