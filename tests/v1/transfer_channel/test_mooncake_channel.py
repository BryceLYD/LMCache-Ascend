# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Mooncake TE transfer channel.

The real ``mooncake`` package is replaced by a fake module injected into
``sys.modules``; transfers and handshake run over real localhost ZMQ
sockets so the channel's protocol logic is exercised end to end without
an NPU.
"""

# Standard
from types import SimpleNamespace
import asyncio
import socket
import sys
import time
import types

# Third Party
import pytest
import torch

# First Party
from lmcache_ascend.v1.transfer_channel import CreateTransferChannel
from lmcache_ascend.v1.transfer_channel.buffer_config import BufferConfig, BufferType
from lmcache_ascend.v1.transfer_channel.mooncake_channel import (
    MooncakeChannel,
    _check_status,
    _load_mooncake,
    _resolve_opcodes,
)

PAGE = 1024
BUF_A = 0x7F0000000000
BUF_B = 0x7F0000000100


class _FakeTransferOpcode:
    READ = "READ"
    WRITE = "WRITE"


class FakeMooncakeEngine:
    """Records every engine call; returns success unless fail_status set."""

    instances: list = []

    def __init__(self):
        self.initialize_calls = []
        self.rpc_port = 23456
        self.registered = []
        self.unregistered = []
        self.transfers = []
        self.notify_args = []
        self.fail_status = 0
        FakeMooncakeEngine.instances.append(self)

    def initialize(
        self, local_hostname, metadata_server, protocol, device_name
    ):
        self.initialize_calls.append(
            (local_hostname, metadata_server, protocol, device_name)
        )
        return 0

    def get_rpc_port(self):
        return self.rpc_port

    def register_memory(self, addr, length):
        self.registered.append((addr, length))
        return 0

    def unregister_memory(self, addr, length):
        self.unregistered.append((addr, length))
        return 0

    def batch_transfer_sync(
        self, target_hostname, buffers, peer_buffer_addresses, lengths, opcode,
        notify=None,
    ):
        self.transfers.append(
            (
                target_hostname,
                list(buffers),
                list(peer_buffer_addresses),
                list(lengths),
                opcode,
            )
        )
        self.notify_args.append(notify)
        return self.fail_status


class _FakeMemObj:
    """Duck-typed MemoryObj: the channel only reads data_ptr/meta.address."""

    def __init__(self, data_ptr, address):
        self.data_ptr = data_ptr
        self.meta = SimpleNamespace(address=address)


@pytest.fixture
def fake_mooncake(monkeypatch):
    FakeMooncakeEngine.instances = []
    fake = types.SimpleNamespace(
        TransferEngine=FakeMooncakeEngine, TransferOpcode=_FakeTransferOpcode
    )
    monkeypatch.setitem(sys.modules, "mooncake", fake)
    return fake


def _patch_npu(monkeypatch):
    monkeypatch.setattr(torch.npu, "current_device", lambda: 0, raising=False)
    monkeypatch.setattr(torch.npu, "set_device", lambda dev: None, raising=False)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _buffer_config(ptr: int, pages: int = 8) -> BufferConfig:
    return BufferConfig(
        ptr=ptr,
        size=pages * PAGE,
        device_id=-1,
        device_type=BufferType.CPU,
        align_bytes=PAGE,
    )


def _make_channel(peer_init_url: str, ptr: int = BUF_A) -> MooncakeChannel:
    return MooncakeChannel(
        async_mode=False,
        buffers=[_buffer_config(ptr)],
        role="both",
        tp_rank=0,
        peer_init_url=peer_init_url,
    )


def _wait_server_ready(chan: MooncakeChannel, timeout: float = 5.0) -> None:
    """The REP socket is bound inside the init-loop thread; the attribute
    is only assigned after the socket is created and bound."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if getattr(chan, "init_side_channel", None) is not None:
            return
        time.sleep(0.01)
    raise TimeoutError("channel init loop did not bind its side channel")


# ---------------------------------------------------------------------------
# Factory / helpers
# ---------------------------------------------------------------------------


def test_factory_rejects_unknown_channel_types():
    with pytest.raises(AssertionError, match="Unsupported channel type"):
        CreateTransferChannel(
            channel_type="bogus",
            async_mode=False,
            role="both",
            buffer_ptr=1,
            buffer_size=PAGE,
            align_bytes=PAGE,
            tp_rank=0,
            peer_init_url="127.0.0.1:1",
            buffer_type=["cpu"],
        )


def test_check_status_variants():
    _check_status(None, "x")
    _check_status(0, "x")
    _check_status(True, "x")
    with pytest.raises(RuntimeError, match="failed with status 3"):
        _check_status(3, "x")


def test_load_mooncake_resolves_transfer_engine_submodule(monkeypatch):
    """Canonical mooncake-transfer-engine layout: TransferEngine lives in
    the mooncake.transfer_engine submodule, not at the package top."""
    top = types.ModuleType("mooncake")
    te = types.SimpleNamespace(
        TransferEngine=FakeMooncakeEngine, TransferOpcode=_FakeTransferOpcode
    )
    monkeypatch.setitem(sys.modules, "mooncake", top)
    monkeypatch.setitem(sys.modules, "mooncake.transfer_engine", te)
    assert _load_mooncake() is te


def test_load_mooncake_resolves_repackaged_engine_module(monkeypatch):
    """vllm-ascend images ship a repackaged tree where the pybind module
    is mooncake.engine (engine.so) without transfer_engine.py."""
    top = types.ModuleType("mooncake")
    eng = types.SimpleNamespace(
        TransferEngine=FakeMooncakeEngine, TransferOpcode=_FakeTransferOpcode
    )
    monkeypatch.setitem(sys.modules, "mooncake", top)
    monkeypatch.setitem(sys.modules, "mooncake.engine", eng)
    assert _load_mooncake() is eng


def test_load_mooncake_error_lists_visible_symbols(monkeypatch):
    """A mooncake package without the transfer-engine API must fail with
    actionable diagnostics (origin + visible symbols)."""
    top = types.ModuleType("mooncake")
    monkeypatch.setitem(sys.modules, "mooncake", top)
    with pytest.raises(ImportError, match="transfer-engine API"):
        _load_mooncake()


# ---------------------------------------------------------------------------
# Construction / registration
# ---------------------------------------------------------------------------


def test_channel_registers_buffers_and_exposes_engine_name(
    fake_mooncake, monkeypatch
):
    _patch_npu(monkeypatch)
    chan = _make_channel(peer_init_url=f"127.0.0.1:{_free_port()}")
    try:
        assert len(chan.engine.initialize_calls) == 1
        local_hostname, metadata_server, protocol, device_name = (
            chan.engine.initialize_calls[0]
        )
        assert local_hostname.endswith(":0")  # port 0 = let Mooncake pick
        assert metadata_server == "P2PHANDSHAKE"
        assert protocol == "ascend"
        assert device_name == ""
        assert chan.engine_name.endswith(":23456")
        assert chan.engine.registered == [(BUF_A, 8 * PAGE)]
        assert chan.mem_handles[0].local_buffer_addrs[2] == BUF_A + 2 * PAGE
    finally:
        chan.close()


def test_close_unregisters_buffers(fake_mooncake, monkeypatch):
    _patch_npu(monkeypatch)
    chan = _make_channel(peer_init_url=f"127.0.0.1:{_free_port()}")
    chan.close()
    assert chan.engine.unregistered == [(BUF_A, 8 * PAGE)]


def test_close_unregisters_single_argument_layout(monkeypatch):
    """The repackaged build's unregister_memory takes the address only."""
    class _OneArgEngine(FakeMooncakeEngine):
        def unregister_memory(self, addr):
            self.unregistered.append((addr, None))
            return 0

    fake = types.SimpleNamespace(
        TransferEngine=_OneArgEngine, TransferOpcode=_FakeTransferOpcode
    )
    monkeypatch.setitem(sys.modules, "mooncake", fake)
    _patch_npu(monkeypatch)
    chan = _make_channel(peer_init_url=f"127.0.0.1:{_free_port()}")
    chan.close()
    assert chan.engine.unregistered == [(BUF_A, None)]


def test_timeout_misconfiguration_warns(fake_mooncake, monkeypatch, caplog):
    _patch_npu(monkeypatch)
    monkeypatch.setenv("MC_TRANSFER_TIMEOUT", "5")
    monkeypatch.setenv("ASCEND_TRANSFER_TIMEOUT", "10000")
    chan = _make_channel(peer_init_url=f"127.0.0.1:{_free_port()}")
    chan.close()
    assert any(
        "should be strictly greater than" in rec.message
        for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# Handshake (two real channels over localhost ZMQ)
# ---------------------------------------------------------------------------


def _make_peer_pair(fake_mooncake, monkeypatch):
    _patch_npu(monkeypatch)
    chan_a = _make_channel(
        peer_init_url=f"127.0.0.1:{_free_port()}", ptr=BUF_A
    )
    chan_b = _make_channel(
        peer_init_url=f"127.0.0.1:{_free_port()}", ptr=BUF_B
    )
    _wait_server_ready(chan_a)
    _wait_server_ready(chan_b)
    return chan_a, chan_b


def test_lazy_init_handshake_exchanges_engines_and_buffers(
    fake_mooncake, monkeypatch
):
    chan_a, chan_b = _make_peer_pair(fake_mooncake, monkeypatch)
    try:
        chan_a.lazy_init_peer_connection(
            local_id="A",
            peer_id="B",
            peer_init_url=chan_b.peer_init_url,
        )

        # Client side sees the server's engine name and buffer table.
        assert chan_a.remote_engine_dict["B"] == chan_b.engine_name
        handle_b = chan_a.remote_peer_buffers["B"].get_handle_by_uuid(
            chan_b.mem_handles[0].uuid
        )
        assert handle_b.buffer_ptr == BUF_B
        assert handle_b.page_size == PAGE

        # Server side recorded the client symmetrically.
        assert chan_b.remote_engine_dict["A"] == chan_a.engine_name
        handle_a = chan_b.remote_peer_buffers["A"].get_handle_by_uuid(
            chan_a.mem_handles[0].uuid
        )
        assert handle_a.buffer_ptr == BUF_A
    finally:
        chan_a.close()
        chan_b.close()


# ---------------------------------------------------------------------------
# Data transfer
# ---------------------------------------------------------------------------


def _transfer_spec(chan_b) -> dict:
    return {
        "receiver_id": "B",
        "remote_buffer_uuids": [chan_b.mem_handles[0].uuid],
        "remote_mem_indexes": [3],
    }


def test_batched_write_resolves_addrs_and_write_opcode(
    fake_mooncake, monkeypatch
):
    chan_a, chan_b = _make_peer_pair(fake_mooncake, monkeypatch)
    try:
        chan_a.lazy_init_peer_connection(
            local_id="A", peer_id="B", peer_init_url=chan_b.peer_init_url
        )
        obj = _FakeMemObj(BUF_A, 2)
        ret = chan_a.batched_write([obj], _transfer_spec(chan_b))

        assert ret == 1
        assert len(chan_a.engine.transfers) == 1
        target, local, remote, lengths, opcode = chan_a.engine.transfers[0]
        assert target == chan_b.engine_name
        assert local == [BUF_A + 2 * PAGE]
        assert remote == [BUF_B + 3 * PAGE]
        assert lengths == [PAGE]
        assert opcode == "WRITE"
    finally:
        chan_a.close()
        chan_b.close()


def test_batched_read_uses_read_opcode(fake_mooncake, monkeypatch):
    chan_a, chan_b = _make_peer_pair(fake_mooncake, monkeypatch)
    try:
        chan_a.lazy_init_peer_connection(
            local_id="A", peer_id="B", peer_init_url=chan_b.peer_init_url
        )
        obj = _FakeMemObj(BUF_A, 5)
        ret = chan_a.batched_read([obj], _transfer_spec(chan_b))

        assert ret == 1
        target, local, remote, lengths, opcode = chan_a.engine.transfers[0]
        assert opcode == "READ"
        assert local == [BUF_A + 5 * PAGE]
        assert remote == [BUF_B + 3 * PAGE]
    finally:
        chan_a.close()
        chan_b.close()


def test_async_batched_read_delegates_to_sync(fake_mooncake, monkeypatch):
    """Pull-delay (ProxyMemoryObj) awaits async_batched_read; it must run
    the same READ transfer as the sync path."""
    chan_a, chan_b = _make_peer_pair(fake_mooncake, monkeypatch)
    try:
        chan_a.lazy_init_peer_connection(
            local_id="A", peer_id="B", peer_init_url=chan_b.peer_init_url
        )
        obj = _FakeMemObj(BUF_A, 5)
        ret = asyncio.run(
            chan_a.async_batched_read([obj], _transfer_spec(chan_b))
        )

        assert ret == 1
        assert len(chan_a.engine.transfers) == 1
        target, local, remote, lengths, opcode = chan_a.engine.transfers[0]
        assert opcode == "READ"
        assert target == chan_b.engine_name
        assert local == [BUF_A + 5 * PAGE]
        assert remote == [BUF_B + 3 * PAGE]
    finally:
        chan_a.close()
        chan_b.close()


def test_opcode_prefix_spelling_resolves(monkeypatch):
    """Repackaged NPU builds may spell the enum members OP_WRITE/OP_READ;
    the channel must resolve either spelling and pass the right value."""
    class _PrefixedOpcode:
        OP_WRITE = "OP_W"
        OP_READ = "OP_R"

    fake = types.SimpleNamespace(
        TransferEngine=FakeMooncakeEngine, TransferOpcode=_PrefixedOpcode
    )
    monkeypatch.setitem(sys.modules, "mooncake", fake)
    _patch_npu(monkeypatch)
    chan_a = _make_channel(peer_init_url=f"127.0.0.1:{_free_port()}")
    chan_b = _make_channel(peer_init_url=f"127.0.0.1:{_free_port()}")
    _wait_server_ready(chan_b)
    try:
        chan_a.lazy_init_peer_connection(
            local_id="A", peer_id="B", peer_init_url=chan_b.peer_init_url
        )
        chan_a.batched_write([_FakeMemObj(BUF_A, 0)], _transfer_spec(chan_b))
        assert chan_a.engine.transfers[0][4] == "OP_W"
        chan_a.batched_read([_FakeMemObj(BUF_A, 1)], _transfer_spec(chan_b))
        assert chan_a.engine.transfers[1][4] == "OP_R"
    finally:
        chan_a.close()
        chan_b.close()


def test_resolve_opcodes_unknown_spelling_lists_members():
    class _WeirdOpcode:
        kWrite = 1

    fake = types.SimpleNamespace(TransferOpcode=_WeirdOpcode)
    with pytest.raises(AttributeError, match="match no known"):
        _resolve_opcodes(fake)


def test_notify_argument_layout(monkeypatch):
    """The repackaged build requires a trailing TransferNotify argument;
    the canonical layout (no TransferNotify) keeps the 5-arg call."""
    class _Notify:
        NONE = "no-notify"

    fake = types.SimpleNamespace(
        TransferEngine=FakeMooncakeEngine,
        TransferOpcode=_FakeTransferOpcode,
        TransferNotify=_Notify,
    )
    monkeypatch.setitem(sys.modules, "mooncake", fake)
    _patch_npu(monkeypatch)
    chan_a = _make_channel(peer_init_url=f"127.0.0.1:{_free_port()}")
    chan_b = _make_channel(peer_init_url=f"127.0.0.1:{_free_port()}")
    _wait_server_ready(chan_b)
    try:
        chan_a.lazy_init_peer_connection(
            local_id="A", peer_id="B", peer_init_url=chan_b.peer_init_url
        )
        chan_a.batched_write([_FakeMemObj(BUF_A, 0)], _transfer_spec(chan_b))
        assert chan_a.engine.notify_args == ["no-notify"]
    finally:
        chan_a.close()
        chan_b.close()


def test_canonical_layout_keeps_five_arg_call(fake_mooncake, monkeypatch):
    """The fixture module has no TransferNotify: the call must stay
    five-argument (notify recorded as None)."""
    chan_a, chan_b = _make_peer_pair(fake_mooncake, monkeypatch)
    try:
        chan_a.lazy_init_peer_connection(
            local_id="A", peer_id="B", peer_init_url=chan_b.peer_init_url
        )
        chan_a.batched_write([_FakeMemObj(BUF_A, 0)], _transfer_spec(chan_b))
        assert chan_a.engine.notify_args == [None]
    finally:
        chan_a.close()
        chan_b.close()


def test_notify_struct_layout_default_instance(monkeypatch):
    """The vllm-ascend repackaged tree ships TransferNotify as a struct
    (msg/name fields, no enum members): the no-op is a default-constructed
    instance passed as the sixth argument."""
    class _NotifyStruct:
        def __init__(self):
            self.msg = ""
            self.name = ""

    fake = types.SimpleNamespace(
        TransferEngine=FakeMooncakeEngine,
        TransferOpcode=_FakeTransferOpcode,
        TransferNotify=_NotifyStruct,
    )
    monkeypatch.setitem(sys.modules, "mooncake", fake)
    _patch_npu(monkeypatch)
    chan_a = _make_channel(peer_init_url=f"127.0.0.1:{_free_port()}")
    chan_b = _make_channel(peer_init_url=f"127.0.0.1:{_free_port()}")
    _wait_server_ready(chan_b)
    try:
        chan_a.lazy_init_peer_connection(
            local_id="A", peer_id="B", peer_init_url=chan_b.peer_init_url
        )
        chan_a.batched_write([_FakeMemObj(BUF_A, 0)], _transfer_spec(chan_b))
        recorded = chan_a.engine.notify_args[0]
        assert isinstance(recorded, _NotifyStruct)
        assert recorded.msg == "" and recorded.name == ""
    finally:
        chan_a.close()
        chan_b.close()


def test_async_batched_write_raises(fake_mooncake, monkeypatch):
    """The upstream ABC declares async_batched_write abstract; the channel
    overrides it to stay instantiable but keeps the failure explicit, so
    P2P KV sharing stays on the hccl/hixl channels."""
    chan_a, chan_b = _make_peer_pair(fake_mooncake, monkeypatch)
    try:
        chan_a.lazy_init_peer_connection(
            local_id="A", peer_id="B", peer_init_url=chan_b.peer_init_url
        )
        with pytest.raises(NotImplementedError, match="async_batched_write"):
            asyncio.run(
                chan_a.async_batched_write(
                    [_FakeMemObj(BUF_A, 0)], _transfer_spec(chan_b)
                )
            )
        assert chan_a.engine.transfers == []
    finally:
        chan_a.close()
        chan_b.close()


def test_batched_write_failure_raises_runtime_error(fake_mooncake, monkeypatch):
    chan_a, chan_b = _make_peer_pair(fake_mooncake, monkeypatch)
    try:
        chan_a.lazy_init_peer_connection(
            local_id="A", peer_id="B", peer_init_url=chan_b.peer_init_url
        )
        chan_a.engine.fail_status = 1
        with pytest.raises(RuntimeError, match="batched_write"):
            chan_a.batched_write([_FakeMemObj(BUF_A, 0)], _transfer_spec(chan_b))
    finally:
        chan_a.close()
        chan_b.close()


def test_raw_bytes_rejected(fake_mooncake, monkeypatch):
    chan_a, chan_b = _make_peer_pair(fake_mooncake, monkeypatch)
    try:
        chan_a.lazy_init_peer_connection(
            local_id="A", peer_id="B", peer_init_url=chan_b.peer_init_url
        )
        with pytest.raises(NotImplementedError, match="raw bytes"):
            chan_a.batched_write([b"raw"], _transfer_spec(chan_b))
    finally:
        chan_a.close()
        chan_b.close()
