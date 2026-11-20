# MP+PD Direct Transfer on lmcache dev — Port Design

This document describes the `mp-pd-dev` branch: the MP+PD disaggregated
prefill pipeline (prefiller pushes KV chunks straight to the decoder over
an Ascend transfer channel) rebuilt on top of **lmcache dev** instead of
the removed in-process v1 runtime. It covers what moved where, what was
frozen here and why, which runtime contracts were verified, and what is
deliberately not ported.

For the feature itself (proxy, roles, store-through, ledger), see the
commit series starting from `[Feature] Add MP+PD unified KV-source
ledger` on `mp-pd-clean` (PR #362). This doc only covers the dev port.

## 1. Motivation

The PD pipeline was originally built against lmcache v0.5.6, where it
inherited from the in-process `PDBackend` (`lmcache/v1/storage_backend/
pd_backend.py`). Upstream lmcache dev:

- moved the whole in-process v1 runtime into `lmcache/inprocess/`
  (v0.5.7), then **deleted it** (#6254) — `pd_backend`, `metadata`,
  `token_database`, `config`, the GPU connector implementations, and the
  transfer-channel base are gone;
- has **no native P→D direct transfer** (the supported disagg path stores
  through the shared MP server, 1p1d only), so there is nothing to
  migrate onto;
- is the only line the upstream lmcache-ascend `main` tracks (nightly
  910B3 CI), and the only line that contains the mamba layout fix #5702
  that hybrid models require.

Staying on v0.5.6 was therefore a dead end, and rewriting the transport
from scratch would discard a stress-tested implementation. The chosen
route: **freeze the removed v0.5.6 modules and re-point everything that
merely moved.**

## 2. Architecture after the port

```
vLLM (vllm-ascend v0.23)
  └─ LMCacheAscendMPPDConnectorV1Dynamic        (ours, scheduler+worker)
       ├─ inherits LMCacheMPConnector           (lmcache dev, live)
       ├─ pd_engine                              (ours)
       │    ├─ group_geometry / ledger           (ours, pure python)
       │    ├─ VLLMPagedMemNPUConnectorV2        (ours; to_gpu/from_gpu
       │    │     call lmcache_ascend.c_ops kernels directly)
       │    └─ AscendPDBackend                   (ours: mixins + config shim)
       │         ├─ PDBackend / PDConfig / PD wire msgs
       │         │      (frozen v0.5.6 lmcache_v056/pd_backend.py)
       │         └─ hccl / hixl / mooncake channels (ours)
       │                └─ BaseTransferChannel + init handshake msgs
       │                     (frozen v0.5.6 lmcache_v056/channel_abstract.py)
       └─ scheduler-side arrival set / dedup    (ours)
```

Three kinds of code now exist side by side:

| Kind | Meaning | Rule |
| :--- | :--- | :--- |
| **lmcache dev, live** | modules that merely moved (`common.kv_format`, `common.token_hasher`, `client.transport`, `common.group_view`, `distributed.api`, `memory_management`, `rpc_utils.get_ip`) | import from their new dev paths — never freeze a copy |
| **frozen v0.5.6** | modules dev deleted that the PD wire protocol depends on | live under `lmcache_ascend/v1/lmcache_v056/`, unchanged in behavior |
| **fork-native** | everything the fork already owned (connectors, mixins, engine, channels, geometry) | unchanged except import rewiring |

## 3. The `lmcache_v056` package

Ground rules (also in `lmcache_v056/__init__.py`):

1. Every file is copied from the LMCache **v0.5.6** tag noted in its
   header and is adjusted **only** for import paths.
2. No new features in the frozen copies; new code targets dev APIs.
3. When dev grows a native replacement for a module in this package, swap the
   fork imports and delete the frozen copy in the same change.
4. Files are trimmed to the surface this repo's runtime actually
   consumes; dropped classes are listed in the file header. The frozen
   snapshot exists to serve this pipeline, not to preserve upstream.

Inventory (origin → adjustments):

| Frozen file | Origin (v0.5.6) | Why frozen here | Adjustments |
| :--- | :--- | :--- | :--- |
| `pd_backend.py` | `v1/storage_backend/pd_backend.py` | PD wire protocol, alloc loops, push path | channel factory + zmq helpers → fork-native; config annotation dropped (duck-typed shim) |
| `channel_abstract.py` | `v1/transfer_channel/abstract.py` | channel base class the Ascend channels extend | `transfer_utils` → in-package sibling |
| `transfer_utils.py` | `v1/transfer_channel/transfer_utils.py` | init handshake msg types | none |
| `cache_key.py` | `utils.py` (`CacheEngineKey` family) | PD chunk-key wire format; dev replaced it with `ObjectKey` hashing | imports dtype maps from dev `lmcache.utils` |
| `metadata.py` | `v1/metadata.py` | per-engine metadata the backend/connector consume | none |
| `token_database.py` | `v1/token_database.py` | chunk boundary logic for the PD leg | `CacheEngineKey` → in-package; config annotation dropped |
| `paged_cpu_gpu_memory_allocator.py` | `v1/memory_allocators/...` | receiver staging pages | paged tensor allocator → in-package sibling |
| `paged_tensor_memory_allocator.py` | `v1/memory_allocators/...` | ditto | stats monitor → no-op stub |
| `gpu_connectors.py` | `v1/gpu_connector/gpu_connectors.py` | `GPUConnectorInterface` + `VLLMPagedMemGPUConnectorV2`, the parent of our NPU V2 connector | trimmed to those two classes (V3 / layerwise / SGLang / TRT-LLM dropped: they served the removed in-process runtime); kv_format → dev `common.kv_format` |
| `gpu_connector_utils.py` | `v1/gpu_connector/utils.py` | geometry facade used by the above | trimmed to the consumed functions; kv_format → dev `common.kv_format` |
| `abstract_backend.py` | — (new stub) | `PDBackend` never calls the base; an empty class keeps the shape without the deleted storage-manager tree | new file |
| `observability.py` | — (new stub) | dev removed `lmcache.observability`; stats become no-ops | new file |

Fork-side additions that carry frozen-copy semantics:

- `lmcache_ascend/v1/rpc_utils.py` gains `get_zmq_context` /
  `get_zmq_socket` (copied verbatim from v0.5.6; dev removed them).
- `pd_engine._build_config_shim()` already produced a `SimpleNamespace`
  config; the frozen backend reads only `pd_*` fields, so no
  `LMCacheEngineConfig` is needed anywhere on this branch.

## 4. Import migration map

| v0.5.6 import | Now |
| :--- | :--- |
| `lmcache.v1.storage_backend.pd_backend` | `lmcache_ascend.v1.lmcache_v056.pd_backend` |
| `lmcache.v1.metadata` | `..lmcache_v056.metadata` |
| `lmcache.v1.token_database` | `..lmcache_v056.token_database` |
| `lmcache.utils.CacheEngineKey` | `..lmcache_v056.cache_key` |
| `lmcache.v1.transfer_channel.abstract/.transfer_utils` | `..lmcache_v056.channel_abstract / .transfer_utils` |
| `lmcache.v1.memory_allocators.paged_cpu_gpu_memory_allocator` | `..lmcache_v056.paged_cpu_gpu_memory_allocator` |
| `lmcache.v1.gpu_connector.gpu_connectors / .utils` | `..lmcache_v056.gpu_connectors / .gpu_connector_utils` |
| `lmcache.v1.multiprocess.token_hasher` | `lmcache.v1.common.token_hasher` (dev, live) |
| `lmcache.v1.multiprocess.transport.factory.RequestClientFactory` | `lmcache.v1.client.transport.factory` (dev, live) |
| `lmcache.v1.multiprocess.group_view.slice_block_ids_per_group` | same path — dev keeps a re-export shim to `common.group_view` (signature-compatible, 4 positional args) |
| `lmcache.v1.gpu_connector.kv_format.*` | `lmcache.v1.common.kv_format.*` (dev, live) |
| `lmcache.v1.rpc_utils.get_zmq_*` | `lmcache_ascend.v1.rpc_utils` (fork) |
| `lmcache.v1.config.LMCacheEngineConfig` | dropped; duck-typed config shim |

## 5. Runtime contracts verified (beyond import resolution)

Module/symbol resolution is necessary but not sufficient; these
behavioral contracts were checked against dev sources:

1. **`LoadStoreOp` field rename** (found in review, fixed): dev packs
   token IDs as `token_bytes` (`common.token_codec`). The store-through
   leg now builds
   `LoadStoreOp(token_bytes=pack_token_ids(tracker.get_token_ids()), ...)`
   — explicit packing, independent of the tracker's internal packing
   timing. `block_ids` stays per-engine-group (`list[list[int]]`), which
   is what both v0.5.6 and dev expect.
2. **`LMCacheMPRequestTracker`** keeps every member the connector reads
   (`all_token_ids`, `num_prompt_tokens`, `allocated_block_ids`,
   `num_stored_tokens`, `num_lmcache_hit_tokens`, `num_scheduled_tokens`,
   `get_token_ids`, `increase_num_stored_tokens`, `cache_salt`,
   `request_configs`).
3. **`LMCacheMPConnectorMetadata.requests`** and
   `add_request_metadata` still exist; the connector's append fallback
   is valid.
4. **`build_connector_worker_meta` semantics changed upstream** (it now
   reports lazy-offload results and returns `None` when disabled). Our
   override already handles `None` from `super()` and rebuilds a base
   metadata with defaults, so it composes with dev.
5. **`MPPDWorkerMetadata.aggregate`**: dev's
   `LMCacheMPWorkerMetadata.aggregate` exists and returns the same two
   fields our merge copies; the TP-merge contract is unchanged.
6. **Connector override signatures**: all 11 overridden methods match
   the dev base class (parameter names, order).
7. **`MemoryObj` / `MemoryObjMetadata` / `MemoryFormat.KV_2LTD`** and
   the ref-count methods (`ref_count_up/down`, `get_ref_count`,
   `meta.address`) are unchanged in dev.
8. **Kernel dispatch is self-contained**: our NPU V2 connector's
   `to_gpu`/`from_gpu`/`batched_*` call `lmcache_ascend.c_ops` kernels
   directly (never the frozen parent's `device_ops` path). Where
   `device_ops` is used at all, dev's `bind_native` rebinds per-symbol
   from our `ops.py`, so the Ascend kernels (v0.5.6-compatible
   signatures) override the torch fallbacks.
9. **Plane-geometry patch target**: `make_page_buffer_shape_desc` is
   called (lazily) by dev's `KVLayerGroupsManager` from
   `lmcache.v1.common.utils` — exactly where `shape_desc.py` installs
   the MLA/DSA plane-geometry fill. The frozen v0.5.6 copy of that
   function is dead code on this path (zero call sites).
10. **`KVLayerGroupsManager(kv_caches, engine_kv_formats=...)`** and
    `GPUMemoryAllocator(size, device=...)` constructor calls are
    signature-compatible with dev.
11. **`RequestClientFactory.create(server_url)`** /
    `TokenHasher(chunk_size=, hash_algorithm=)` — dev signatures match
    our call sites.

The hybrid state-page leg adds pipeline-internal contracts of its own:

12. **Hybrid state pages ride the last boundary only**: the live
    recurrent state a send instruction transfers is the runner's
    state after the block covering the instruction's LAST boundary,
    so only that boundary's combined object carries a state page.
    Earlier boundaries' blocks may already be retired or reused by
    the time the gather runs, so their pages would read garbage --
    even another request's state. The receiver scatters the page
    only when the instruction landed fully
    (``landed_end == instr.end``); on a gap, scattering is
    suppressed and vLLM's native hybrid recompute regenerates the
    state beyond the gap.
13. **Recurrent block tables index with the recurrent group's own
    tokens-per-block** (``get_group_tokens_per_block``, resolved
    once at assemble). Attention-group block sizes never index a
    recurrent table: the two may differ, which would misplace
    state pages.
14. **Restore-point null slots fail closed**: vLLM pads retired or
    skipped block-table slots with one shared null block. When the
    restore-point state slot is the null block, the whole load
    instruction fails (all blocks failed, landed end reset) so vLLM
    recomputes the range instead of decoding from lost state.
15. **Receiver staging pages have a bounded lifecycle**: chunks are
    released once their load lands, and never-consumed chunks
    (timeout recomputes, preemptions, one-shot prompts) are evicted
    oldest-first when receiver allocation comes up short. A removed
    key self-reports through the arrived-keys diff, so the
    scheduler's arrival walks stay truthful.
16. **Geometry discrimination is strict**: state lists have a
    uniform rank; a page-packed form (block stride spanning a padded
    page) implies one backing pool, so tensors in separate storages
    cannot be page-packed; page-packed views must be
    innermost-contiguous; slice-planes states must be dense
    contiguous slices. MLA-style ``(k, v)`` pairs stay
    unclassifiable instead of becoming bogus recurrent groups.

## 6. Deliberately not ported

- **In-process storage manager** (`CreateStorageBackends` /
  `AscendStorageManager` / `AscendP2PBackend` launcher): the v0.5.6
  branches keep it; on this branch `storage_backend/__init__.py` is a
  docstring-only placeholder. The P2P backend test
  (`tests/v1/storage_backend/test_ascend_p2p_backend.py`) is in-process
  only and does not run here.
- **The in-process NPU connector family** (buffer/paged layerwise,
  SGLang, CacheBlend variant) and its `CreateNPUConnector` factory:
  they served the in-process v1 runtime that lmcache dev removed,
  nothing on this branch calls them (`npu_connector/__init__.py` is
  docstring-only), and the frozen snapshot no longer carries their
  parent classes. They remain on the v0.5.6-era branches;
  `tests/v1/test_npu_connector.py` keeps only the V2 benchmark.
- **CacheBlend** and non-vLLM engines: unsupported, same as upstream.

## 7. Limitations and deferred work

- Frozen code no longer receives upstream fixes; the removal
  rule (§3) is the exit path. Watch dev's `distributed/transfer_channel`
  and any native PD server-side work as future replacements.
- `isort` could not run on the Windows workstation; import blocks were
  hand-ordered to the repo convention. Re-run `isort` (black profile)
  on the node before sending anything upstream.
- Static verification only (py_compile, pyflakes undefined-name
  sweep, module+symbol closure against the pinned dev tree,
  signature diffs). Runtime bring-up on the nodes is the next
  gate (§8).
- dev moves ~40 commits/day. The branch is verified against **LMCache
  dev @ `61faa902` (2026-11-16)**; re-pin and re-run the closure checks
  before node deployment.

## 8. Version matrix and bring-up

| Component | Version |
| :--- | :--- |
| lmcache | dev @ `61faa902` (contains #5702 — no cherry-pick needed) |
| lmcache-ascend | `mp-pd-dev` branch |
| vllm-ascend | v0.23.0 (same image the upstream nightly CI uses) |
| vllm | whatever vllm-ascend v0.23 ships |

Node bring-up order: checkout lmcache dev pin on both nodes → reinstall →
checkout `mp-pd-dev` → rebuild (`c_ops`) → start decoder with
`LMCACHE_ASCEND_DUMP_MAMBA=1` → verify the three startup log lines
(group classification, dedup state, no "hybrid grouping unavailable") →
single-request closed loop → invariance diff vs bare-vllm → lm-eval →
`multi_round_qa` stress (10 users × 8 rounds).

Note: the Qwen3.8 startup crash that blocked the v0.5.6 stack
("Mamba conv state must view the page base") does not exist on this
stack — the fix is already in dev.
