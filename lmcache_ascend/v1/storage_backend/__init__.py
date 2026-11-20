# SPDX-License-Identifier: Apache-2.0
"""Ascend storage backends.

On lmcache dev this package carries only the MP+PD direct-transfer
surface: :mod:`.pd` (the Ascend PD backend family) and :mod:`.utils`.
The in-process storage-manager launcher (``CreateStorageBackends`` /
``AscendStorageManager`` / the P2P plugin) used the in-process v1 runtime
that lmcache dev removed, so it is not importable on this branch and has
been dropped here; it remains maintained on the v0.5.6-era branches.
"""
