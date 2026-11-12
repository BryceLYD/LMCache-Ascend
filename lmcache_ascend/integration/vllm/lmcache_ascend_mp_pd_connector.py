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
* Step 5 replaced the scheduler-side lookup pipeline with the
  ledger-driven APC -> L1 -> PD_IN merge and the mixed load path.
* Step 6 (current) closes the reuse loop on the decoder:
  ``pd_store_through`` persists the worker-confirmed PD-landed prefix
  into the decoder's own LMCache server by injecting STORE-direction
  request metadata into the upstream store flow (upstream itself never
  stores from the consumer), and ``pd_alloc_check_l1`` extends the
  receiver's alloc-time dedup from the receive buffer to the decoder's
  L1 so a repeated prompt is not re-transmitted by the producer. Both
  features fail open to the pre-Step-6 behavior.
"""

# Standard
from typing import TYPE_CHECKING, Any, Optional

# Third Party
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPRequestMetadata,
    LMCacheMPWorkerMetadata,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp
from lmcache.v1.multiprocess.group_view import slice_block_ids_per_group
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.logger import init_logger
from vllm.v1.outputs import KVConnectorOutput
import torch

# First Party
from lmcache_ascend.integration.vllm.pd_engine import (
    L1QueryOracle,
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
        # Step 6: decoder-side store-through (persist the PD-landed
        # prefix into the decoder's own LMCache server) and the
        # alloc-time L1 dedup (skip re-transmission of chunks the
        # decoder can serve from its own L1). Both degrade to today's
        # behavior when off or when their plumbing errors.
        self._pd_store_through = bool(extra.get("pd_store_through", True))
        self._pd_alloc_check_l1 = bool(extra.get("pd_alloc_check_l1", True))
        # Must match the server fleet's hashing (the same key the MP
        # adapters resolve; default "blake3" on both sides).
        self._pd_l1_hash_algorithm = str(
            extra.get("lmcache.mp.hash_algorithm", "blake3")
        )
        # Step 5: scheduler-side PD union-walk state.
        self._pd_arrived: set = set()
        self._pd_key_db = None
        self._pd_chunk_size = 0
        self._pd_load_issued: set[str] = set()
        # Step 6: scheduler-side confirmed-landed prefix per request
        # (worker-reported; the store-through source of truth).
        self._pd_landed: dict[str, int] = {}
        # Step 5: worker-side PD load completion / failure state.
        self._pd_load_done: set[str] = set()
        self._pd_failed_blocks: set[int] = set()
        # Step 6: worker-side landed reports awaiting the next
        # build_connector_worker_meta call.
        self._pd_loaded_pending: dict[str, int] = {}
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
            self._pd_maybe_attach_l1_oracle()

    def _pd_maybe_attach_l1_oracle(self) -> None:
        """Build the L1 residency oracle and hand it to the PD backend.

        Receiver (decoder) worker only: the alloc handler will dedup
        incoming chunks against the decoder's OWN LMCache server, which
        is the same server the scheduler-side lookup queries -- so an
        alloc-time hit implies a scheduler lookup hit and the ledger
        resolves the range to L1 (no transmission hole). Any wiring
        failure just leaves the dedup at buffer level.
        """
        engine = self._pd_engine
        if (
            engine is None
            or engine.pd_role != "receiver"
            or not self._pd_alloc_check_l1
        ):
            return
        adapter = getattr(self, "worker_adapter", None)
        server_url = getattr(adapter, "_server_url", None)
        mp_chunk = getattr(adapter, "lmcache_tokens_per_chunk", None)
        if not server_url or not mp_chunk:
            logger.warning(
                "MP+PD: no MP server info on the worker adapter; the "
                "alloc-time L1 dedup stays disabled"
            )
            return
        try:
            # Third Party
            from lmcache.v1.distributed.api import ObjectKey
            from lmcache.v1.rpc_utils import get_zmq_context

            # Mirror the worker-specific store-key derivation exactly
            # (see ipc_key_to_object_keys): the adapter's worker_id /
            # world_size already carry the kv-rank semantics (MLA / DCP
            # aware), so the oracle queries the same shard the decoder
            # itself stores under.
            kv_rank = ObjectKey.ComputeKVRank(
                world_size=adapter.world_size,
                global_rank=adapter.worker_id,
                local_world_size=adapter.world_size,
                local_rank=adapter.worker_id,
            )
            metadata = engine.metadata
            shape = list(metadata.kv_shape)
            token_dim = engine.fmt.token_dim() if engine.fmt else 2
            shape[token_dim] = mp_chunk
            oracle = L1QueryOracle(
                server_url,
                get_zmq_context(use_asyncio=False),
                model_name=metadata.model_name,
                kv_rank=kv_rank,
                chunk_size=mp_chunk,
                hash_algorithm=self._pd_l1_hash_algorithm,
                kv_shape=torch.Size(shape),
                kv_dtype=metadata.kv_dtype,
            )
            engine.attach_l1_oracle(oracle)
        except Exception:
            logger.exception(
                "MP+PD: L1 oracle wiring failed; alloc dedup stays at "
                "buffer level"
            )

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
                failed, landed_end = engine.execute_pd_load(
                    instr, kvcaches
                )
                if failed:
                    self._pd_failed_blocks.update(failed)
                if landed_end > self._pd_loaded_pending.get(
                    instr.request_id, 0
                ):
                    self._pd_loaded_pending[instr.request_id] = landed_end
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
        self._pd_drain_consumer_store_tracking(finished_req_ids)
        return sending, recving

    def _pd_drain_consumer_store_tracking(
        self, finished_req_ids: set[str]
    ) -> None:
        """Drop store-through completions the adapter can never report.

        Upstream consumers pass an empty engine-finished set to the
        worker adapter (``_can_store`` is producer-only), so completed
        store ids would sit in ``finished_stores`` forever. Harmless
        for upstream (consumers never store); with store-through the
        decoder does store, so bound the tracking by the request
        lifetime: once the engine reports the request finished, its
        store id is dead weight (the id would never be returned
        anyway). Producer behavior is untouched.
        """
        engine = self._pd_engine
        adapter = getattr(self, "worker_adapter", None)
        if (
            not finished_req_ids
            or engine is None
            or engine.pd_role != "receiver"
            or adapter is None
        ):
            return
        for req_id in finished_req_ids:
            adapter.finished_stores.discard(req_id)
            adapter.previously_finished.discard(req_id)

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
            and not self._pd_loaded_pending
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
        )
        out.pd_received = self._pd_received
        out.pd_arrived_keys = self._pd_arrived_pending
        out.pd_loaded = self._pd_loaded_pending
        self._pd_received = {}
        self._pd_arrived_pending = []
        self._pd_loaded_pending = {}
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
            self._pd_inject_store_through(metadata)
        return metadata

    def update_connector_output(self, connector_output: KVConnectorOutput):
        super().update_connector_output(connector_output)
        meta = getattr(connector_output, "kv_connector_worker_meta", None)
        if not isinstance(meta, MPPDWorkerMetadata):
            return
        if meta.pd_arrived_keys:
            self._pd_arrived.update(meta.pd_arrived_keys)
        for req_id, landed in (getattr(meta, "pd_loaded", None) or {}).items():
            if landed > self._pd_landed.get(req_id, 0):
                self._pd_landed[req_id] = landed
                logger.debug(
                    "MP+PD store-through: req %s landed prefix is now %d "
                    "tokens",
                    req_id,
                    landed,
                )
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
        self._pd_landed.pop(request.request_id, None)
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

        ``l1_only`` ships nothing through PD, but the proxy gate must
        still open once the prefill phase is over (the decoder
        recomputes everything), so notify-only instructions are emitted
        for it too. The same gate guarantee applies whenever the
        prefill phase ends with nothing left to ship.

        Ranges are aligned to the PD chunk size (the unit the chunk
        keys and the token database are built from).
        """
        if not self._pd_specs:
            return []
        trackers = getattr(self, "request_trackers", None) or {}
        chunk = self._pd_chunk_size or (
            self.scheduler_adapter.lmcache_tokens_per_chunk
        )
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
            ship = self._pd_save_policy != "l1_only"
            if handled <= spec.sent_end or not ship:
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

    def _pd_inject_store_through(self, metadata) -> None:
        """Persist the PD-landed prefix into the decoder's L1 (Step 6).

        Upstream never stores from the decoder: ``_can_store`` is
        producer-only and ``GetStoreMetadata`` bounds the store range
        by computed tokens, which excludes the PD-loaded prefix. So a
        repeated prompt finds an empty decoder cache and forces a full
        PD re-transfer.

        The fix rides upstream's own store data path: for each request
        whose worker-confirmed landed prefix (``pd_loaded`` -- the
        leading run of D2D loads that succeeded) reaches past
        ``num_stored_tokens``, emit one extra STORE-direction request
        metadata block covering the missing full server chunks and
        advance ``num_stored_tokens`` to keep upstream's "already
        stored" invariant. The worker's ``wait_for_save`` executes it
        like any store (paged-pool gather -> D2H -> the decoder's
        server); the blocks still belong to the running request, so
        the data cannot have been evicted underneath the store.

        Emitted after ``super().build_connector_meta`` -- on v0.5.6 the
        decoder generates no regular stores (``_can_store`` false), so
        the pseudo store is the only one and no overlap exists.
        """
        if not self._pd_store_through or not self._pd_landed:
            return
        mp_chunk = self.scheduler_adapter.lmcache_tokens_per_chunk
        group_tokens_per_block = getattr(self, "_group_tokens_per_block", None)
        if not mp_chunk or not group_tokens_per_block:
            return
        trackers = getattr(self, "request_trackers", None) or {}
        for req_id in list(self._pd_landed):
            tracker = trackers.get(req_id)
            if tracker is None:
                continue
            landed = self._pd_landed[req_id]
            # Only complete server chunks are storable (upstream
            # invariant: num_stored_tokens stays a multiple of
            # lmcache_tokens_per_chunk); the PD tail beyond the last
            # full server chunk waits for the regular computed store
            # or is not stored (documented: set the PD chunk size
            # equal to the server chunk size for full coverage).
            store_end = min(
                landed // mp_chunk * mp_chunk,
                len(tracker.all_token_ids),
                tracker.num_prompt_tokens,
            )
            start = tracker.num_stored_tokens
            num_chunks = (store_end - start) // mp_chunk
            if num_chunks <= 0:
                continue
            end = start + num_chunks * mp_chunk
            try:
                block_ids = slice_block_ids_per_group(
                    tracker.allocated_block_ids,
                    group_tokens_per_block,
                    start,
                    end,
                )
            except ValueError:
                logger.warning(
                    "MP+PD store-through: req %s range [%d,%d) does not "
                    "align to a KV group; skipping this step",
                    req_id,
                    start,
                    end,
                )
                continue
            store_meta = LMCacheMPRequestMetadata(
                request_id=req_id,
                direction="STORE",
                op=LoadStoreOp(
                    token_ids=tracker.get_token_ids(),
                    block_ids=block_ids,
                    start=start,
                    end=end,
                ),
                cache_salt=tracker.cache_salt,
                request_configs=tracker.request_configs,
            )
            if getattr(self, "lazy_offload", False):
                self._lazy_offload_manager.add_store_candidate(store_meta)
            else:
                metadata.add_request_metadata(store_meta)
            tracker.increase_num_stored_tokens(end - start)
            logger.info(
                "MP+PD store-through: storing req %s [%d,%d) from the "
                "PD-landed prefix",
                req_id,
                start,
                end,
            )
            # The num_stored_tokens advance makes re-emission idempotent.
            # The landed marker itself stays: if upstream ever resets
            # num_stored_tokens (failed-load bypass), the re-store would
            # persist recomputed -- content-identical -- data, which is
            # redundant but correct. request_finished drops the marker.

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
