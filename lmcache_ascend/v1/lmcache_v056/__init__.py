# SPDX-License-Identifier: Apache-2.0
"""Upstream LMCache v0.5.6 modules kept as a frozen snapshot for the
MP+PD direct-transfer chain.

LMCache dev removed the in-process tree (`lmcache/v1/storage_backend`,
`lmcache/v1/metadata`, `lmcache/v1/token_database`, the GPU connector
implementations, and the transfer-channel base) in the v0.5.7 cycle
(upstream #6254). The Ascend PD direct-transfer pipeline still needs
those proven building blocks, so they live here as a frozen snapshot,
unchanged in behavior.

Ground rules for this package:

* Every file is copied from the LMCache v0.5.6 tag noted in its header
  and is adjusted ONLY for import paths (dead upstream modules ->
  in-package siblings, moved-but-alive modules -> their new lmcache dev
  locations) - plus one documented trim: files are cut down to the
  surface this repo's runtime actually consumes, with the dropped
  classes listed in the file header (e.g. the in-process-only connector
  classes inside gpu_connectors.py).
* Do not add new features here. New code should target lmcache dev APIs;
  when dev grows a native replacement for a module in this package, swap
  the fork imports and delete the frozen copy in the same change.
"""

# This package intentionally has no exports; import the submodules directly.
