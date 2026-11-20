# SPDX-License-Identifier: Apache-2.0
"""NPU KV-cache connectors.

On lmcache dev this package carries only the paged-memory V2 connector
(:class:`~lmcache_ascend.v1.npu_connector.npu_connectors.VLLMPagedMemNPUConnectorV2`)
used by the MP+PD engine. The in-process connector family and its
CreateNPUConnector factory served the in-process v1 runtime that
lmcache dev removed; they are not importable on this branch and were
dropped. They remain maintained on the v0.5.6-era branches.
"""
