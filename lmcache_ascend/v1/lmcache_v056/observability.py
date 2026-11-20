# SPDX-License-Identifier: Apache-2.0
"""No-op stand-in for the removed upstream ``lmcache.observability``.

The frozen v0.5.6 paged allocator reports usage stats through
``LMCStatsMonitor``; lmcache dev removed that module (stats moved into the
MP server). Metrics are optional: this stub accepts every call and drops
the data, keeping the frozen allocator behavior otherwise identical.
"""


class LMCStatsMonitor:
    _instance = None

    @classmethod
    def GetOrCreate(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __getattr__(self, name):
        # Any update_*/record_* call becomes a no-op.
        def _noop(*args, **kwargs):
            return None
        return _noop
