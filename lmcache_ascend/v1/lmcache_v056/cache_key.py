# SPDX-License-Identifier: Apache-2.0
# Frozen copy of LMCache v0.5.6 `lmcache/utils.py` (CacheEngineKey and
# LayerCacheEngineKey). lmcache dev replaced this key family with
# ObjectKey hashing for MP mode; the PD direct-transfer wire protocol and
# the prefix-hash ledger still speak CacheEngineKey strings.

# Standard
from dataclasses import dataclass, field
from typing import List, Optional

# Third Party
import torch

# First Party
from lmcache.utils import STR_DTYPE_TO_TORCH_DTYPE, TORCH_DTYPE_TO_STR_DTYPE

@dataclass(slots=True)
class CacheEngineKey:
    model_name: str
    world_size: int
    worker_id: int
    chunk_hash: int
    dtype: torch.dtype
    request_configs: Optional[dict] = field(default_factory=dict)
    tags: Optional[tuple] = field(init=False, default=None)
    _dtype_str: str = field(init=False, default="")

    def __post_init__(self):
        tag_list = None
        if self.request_configs is not None:
            for k, v in self.request_configs.items():
                if k.startswith("lmcache.tag."):
                    if tag_list is None:
                        tag_list = []
                    tag_list.append((k[len("lmcache.tag.") :], v))
        if self.dtype not in TORCH_DTYPE_TO_STR_DTYPE:
            raise ValueError(f"Unsupported dtype in CacheEngineKey: {self.dtype}")
        self._dtype_str = TORCH_DTYPE_TO_STR_DTYPE[self.dtype]
        # use tuple to save tags
        self.tags = None if tag_list is None else tuple(tag_list)

    def __hash__(self):
        return hash(
            (
                self.model_name,
                self.world_size,
                self.worker_id,
                self.chunk_hash,
                self._dtype_str,
                self.tags,
            )
        )

    def __eq__(self, other):
        if type(self) is type(other):
            return (
                self.model_name == other.model_name
                and self.world_size == other.world_size
                and self.worker_id == other.worker_id
                and self.chunk_hash == other.chunk_hash
                and self.dtype == other.dtype
                and self.tags == other.tags
            )

        return False

    def to_string(self):
        s = (
            f"{self.model_name}@{self.world_size}"
            f"@{self.worker_id}@{self.chunk_hash_hex}@{self._dtype_str}"
        )
        if self.tags is not None and len(self.tags) != 0:
            tags = [f"{k}%{v}" for k, v in self.tags]
            s += "@" + "@".join(tags)
        return s

    def split_layers(self, num_layers: int) -> List["LayerCacheEngineKey"]:
        """Split the key into multiple keys for each layer"""
        keys = []
        for layer_id in range(num_layers):
            keys.append(
                LayerCacheEngineKey(
                    model_name=self.model_name,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    chunk_hash=self.chunk_hash,
                    dtype=self.dtype,
                    request_configs=self.request_configs,
                    layer_id=layer_id,
                )
            )
        return keys

    def get_first_layer(self) -> "LayerCacheEngineKey":
        """Return the key for the first layer"""
        key = LayerCacheEngineKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=self.worker_id,
            chunk_hash=self.chunk_hash,
            dtype=self.dtype,
            request_configs=self.request_configs,
            layer_id=0,
        )
        return key

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        if len(parts) < 5:
            raise ValueError(f"Invalid key string: {s}")
        request_configs = None
        if len(parts) >= 6:
            request_configs = {}
            for kv in parts[5:]:
                kvs = kv.split("%", 1)
                if len(kvs) != 2:
                    raise ValueError(f"Invalid key string: {s}")
                request_configs["lmcache.tag." + kvs[0]] = kvs[1]
        return CacheEngineKey(
            model_name=parts[0],
            world_size=int(parts[1]),
            worker_id=int(parts[2]),
            chunk_hash=int(parts[3], 16),
            dtype=STR_DTYPE_TO_TORCH_DTYPE[parts[4]],
            request_configs=request_configs,
        )

    def to_dict(self):
        # Note(Kuntai): this is used for serializing CacheEngineKey via msgpack.
        msg = {
            "__type__": "CacheEngineKey",
            "model_name": self.model_name,
            "world_size": self.world_size,
            "worker_id": self.worker_id,
            "chunk_hash": self.chunk_hash,
            "dtype": self._dtype_str,
        }
        if self.request_configs is not None and len(self.request_configs) != 0:
            msg["request_configs"] = [
                f"{k}%{v}" for k, v in self.request_configs.items()
            ]
        return msg

    @staticmethod
    def from_dict(d):
        request_configs = None
        if request_configs_list := d.get("request_configs"):
            request_configs = {}
            for kv in request_configs_list:
                kvs = kv.split("%", 1)
                if len(kvs) != 2:
                    raise ValueError(f"Invalid key dict: {d}")
                request_configs[kvs[0]] = kvs[1]
        return CacheEngineKey(
            model_name=d["model_name"],
            world_size=d["world_size"],
            worker_id=d["worker_id"],
            chunk_hash=d["chunk_hash"],
            dtype=STR_DTYPE_TO_TORCH_DTYPE[d["dtype"]],
            request_configs=request_configs,
        )

    def with_new_worker_id(self, new_worker_id: int) -> "CacheEngineKey":
        # Reconstruct the cache engine key with new worker id
        return CacheEngineKey(
            self.model_name,
            world_size=self.world_size,
            worker_id=new_worker_id,
            chunk_hash=self.chunk_hash,
            dtype=self.dtype,
            request_configs=self.request_configs,
        )

    @property
    def chunk_hash_hex(self) -> str:
        if isinstance(self.chunk_hash, bytes):
            return self.chunk_hash.hex()
        return f"{self.chunk_hash:x}"


@dataclass(slots=True)
class LayerCacheEngineKey(CacheEngineKey):
    """A key for the layer cache engine"""

    layer_id: int = 0

    def __hash__(self):
        return hash(
            (
                self.model_name,
                self.world_size,
                self.worker_id,
                self.chunk_hash,
                self._dtype_str,
                self.tags,
                self.layer_id,
            )
        )

    def __eq__(self, other):
        if super(LayerCacheEngineKey, self).__eq__(other):
            return self.layer_id == other.layer_id

        return False

    def to_string(self):
        s = (
            f"{self.model_name}@{self.world_size}"
            f"@{self.worker_id}@{self.chunk_hash_hex}@{self._dtype_str}@{self.layer_id}"
        )
        if self.tags is not None and len(self.tags) != 0:
            tags = [f"{k}%{v}" for k, v in self.tags]
            s += "@" + "@".join(tags)
        return s

    def split_layers(self, num_layers: int) -> List["LayerCacheEngineKey"]:
        """Split the key into multiple keys for each layer"""
        keys = []
        for layer_id in range(num_layers):
            keys.append(
                LayerCacheEngineKey(
                    model_name=self.model_name,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    chunk_hash=self.chunk_hash,
                    dtype=self.dtype,
                    request_configs=self.request_configs,
                    layer_id=layer_id,
                )
            )
        return keys

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        if len(parts) < 6:
            raise ValueError(f"Invalid key string: {s}")
        request_configs = None
        if len(parts) >= 7:
            request_configs = {}
            for kv in parts[6:]:
                kvs = kv.split("%", 1)
                if len(kvs) != 2:
                    raise ValueError(f"Invalid key string: {s}")
                request_configs["lmcache.tag." + kvs[0]] = kvs[1]
        return LayerCacheEngineKey(
            model_name=parts[0],
            world_size=int(parts[1]),
            worker_id=int(parts[2]),
            chunk_hash=int(parts[3], 16),
            dtype=STR_DTYPE_TO_TORCH_DTYPE[parts[4]],
            request_configs=request_configs,
            layer_id=int(parts[5]),
        )
