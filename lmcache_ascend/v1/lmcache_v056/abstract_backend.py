# SPDX-License-Identifier: Apache-2.0
"""Minimal stand-in for the removed upstream AllocatorBackendInterface.

Frozen v0.5.6 ``PDBackend`` subclasses this name but never calls into the
base (its ``__init__`` is fully self-contained), so an empty base keeps the
inheritance shape without dragging in the in-process storage-manager tree
that lmcache dev deleted.
"""


class AllocatorBackendInterface:
    pass
