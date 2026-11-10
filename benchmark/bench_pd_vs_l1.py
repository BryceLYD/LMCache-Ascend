# SPDX-License-Identifier: Apache-2.0
"""Benchmark PD (D2D, NPU-to-NPU) vs L1 cache pull (H2D, host-to-NPU).

Measures the two KV delivery paths that the MP+PD connector ledger can
choose from, using the production code paths of this repo:

- ``d2d``: the real transfer channel stack (CreateTransferChannel ->
  MooncakeChannel/HcclChannel) — same transport the PD transfer leg
  uses. Sender and receiver run as two processes (two NPUs on one
  machine, or two machines), so the topology mirrors production.
- ``h2d``: pinned host pool -> NPU page-stream copies — the copy floor
  of the L1 cache pull path (the full L1 pull additionally includes
  lookup + IPC, so treat this number as the best case for L1).

Usage (single machine, two NPUs):

    # terminal 1 (receiver, e.g. npu:1)
    python benchmark/bench_pd_vs_l1.py d2d --role receiver --device 1

    # terminal 2 (sender, e.g. npu:0)
    python benchmark/bench_pd_vs_l1.py d2d --role sender --device 0 \
        --peer-host 127.0.0.1

    # L1 H2D leg (single process)
    python benchmark/bench_pd_vs_l1.py h2d --device 0

Cross-node PD: run the receiver on node B and the sender on node A
with ``--peer-host <nodeB_ip>``.

Defaults mirror a real 1p1d deployment: 64 pages x 36 MiB = 2.25 GiB
per iteration (the pd_buffer_size used by the hccs/1p1d examples).

Notes:
- If a transfer aborts with a Mooncake timeout, raise
  ASCEND_TRANSFER_TIMEOUT (ms) / MC_TRANSFER_TIMEOUT (s) or lower
  --pages.
- Mooncake TE also ships its own perf tool
  (transfer_engine_ascend_direct_perf) as an independent cross-check.
"""

# Standard
from types import SimpleNamespace
import argparse
import dataclasses
import time

# Third Party
import torch
import torch_npu  # noqa: F401
import zmq

# First Party
from lmcache_ascend.v1.transfer_channel import CreateTransferChannel
from lmcache_ascend.v1.transfer_channel.buffer_config import BufferType

MB = 1024 * 1024


def _fmt_bw(total_bytes: int, elapsed_s: float) -> str:
    gbps = total_bytes / elapsed_s / 1024**3 if elapsed_s > 0 else 0.0
    return f"{gbps:8.2f} GB/s"


def _report(tag: str, total_bytes: int, elapsed_s: float, iters: int) -> None:
    per_iter = elapsed_s / iters
    print(
        f"[{tag}] total={total_bytes / 1024**3:.2f} GiB in "
        f"{elapsed_s:.2f}s -> {_fmt_bw(total_bytes, elapsed_s)} | "
        f"{per_iter * 1000:.1f} ms/iter ({iters} iters)"
    )


def _make_buffer(device: int, total_bytes: int, fill: bool) -> torch.Tensor:
    torch.npu.set_device(device)
    buf = torch.empty(total_bytes, dtype=torch.uint8, device=f"npu:{device}")
    if fill:
        buf.fill_(0x5A)
        torch.npu.synchronize()
    return buf


def _make_channel(channel: str, role: str, buf: torch.Tensor, page_bytes: int,
                  init_url: str):
    try:
        return CreateTransferChannel(
            channel_type=channel,
            async_mode=False,
            role=role,
            buffer_ptr=buf.data_ptr(),
            buffer_size=buf.numel(),
            buffer_type="npu",
            align_bytes=page_bytes,
            tp_rank=0,
            peer_init_url=init_url,
        )
    except ImportError as e:
        raise SystemExit(
            f"channel '{channel}' is unavailable in this environment: {e}\n"
            "hint: build mooncake-transfer-engine with -DUSE_ASCEND_DIRECT=ON "
            "for --channel mooncake, or use --channel hccl"
        )


def _make_objects(channel: str, buf: torch.Tensor, pages: int) -> list:
    """One object per page. MooncakeChannel resolves (data_ptr, page idx)
    via duck typing; hccl/hixl channels require real MemoryObj."""
    page_bytes = buf.numel() // pages
    if channel == "mooncake":
        return [
            SimpleNamespace(
                data_ptr=buf.data_ptr() + k * page_bytes,
                meta=SimpleNamespace(address=k),
            )
            for k in range(pages)
        ]

    # Third Party
    from lmcache.v1.cache_engine import MemoryFormat
    from lmcache.v1.memory_management import MemoryObj, MemoryObjMetadata

    fields = {f.name for f in dataclasses.fields(MemoryObjMetadata)}
    kwargs = {
        "fmt": MemoryFormat.KV_2LTD,
        "dtype": buf.dtype,
        "shape": torch.Size([page_bytes]),
        "physical_size": torch.Size([page_bytes]),
        "address": k,
        "use_gpu": True,
        "pin_allocator": False,
    }
    kwargs = {k: v for k, v in kwargs.items() if k in fields}
    objs = []
    for k in range(pages):
        meta = MemoryObjMetadata(**{**kwargs, "address": k})
        view = buf.narrow(0, k * page_bytes, page_bytes)
        try:
            objs.append(MemoryObj(view, meta))
        except TypeError:
            objs.append(MemoryObj(view, meta, "GPU"))
    return objs


def _transfer_spec(peer_id: str, uuid: str, pages: int) -> dict:
    return {
        "receiver_id": peer_id,
        "remote_buffer_uuids": [uuid] * pages,
        "remote_mem_indexes": list(range(pages)),
    }


def run_d2d(args) -> None:
    page_bytes = args.page_size_mb * MB
    total_bytes = args.pages * page_bytes
    buf = _make_buffer(args.device, total_bytes, fill=args.role == "sender")

    if args.role == "receiver":
        chan = _make_channel(args.channel, "receiver", buf, page_bytes,
                             f"{args.bind_ip}:{args.init_port}")
        ctx = zmq.Context()
        done = ctx.socket(zmq.PULL)
        done.bind(f"tcp://{args.bind_ip}:{args.done_port}")
        print(f"[receiver] waiting on {args.bind_ip}:{args.init_port} "
              f"(channel={args.channel}, device=npu:{args.device})")
        t0 = time.perf_counter()
        done.recv()
        _report("receiver", args.iters * total_bytes,
                time.perf_counter() - t0, args.iters)
        done.close(0)
        ctx.term()
        chan.close()
        return

    # sender
    chan = _make_channel(args.channel, "sender", buf, page_bytes,
                         "127.0.0.1:0")
    peer_id = "bench_recv"
    chan.lazy_init_peer_connection(
        local_id="bench_sender",
        peer_id=peer_id,
        peer_init_url=f"{args.peer_host}:{args.init_port}",
    )
    handle = chan.remote_peer_buffers[peer_id].handles[0]
    recv_pages = handle.buffer_size // page_bytes
    assert args.pages <= recv_pages, (
        f"receiver buffer too small: {recv_pages} pages < {args.pages}"
    )
    objs = _make_objects(args.channel, buf, args.pages)
    spec = _transfer_spec(peer_id, handle.uuid, args.pages)

    ctx = zmq.Context()
    done = ctx.socket(zmq.PUSH)
    done.connect(f"tcp://{args.peer_host}:{args.done_port}")

    for _ in range(args.warmup):
        chan.batched_write(objs, spec)
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.iters):
        chan.batched_write(objs, spec)
    elapsed = time.perf_counter() - t0
    torch.npu.synchronize()

    _report("sender", args.iters * total_bytes, elapsed, args.iters)
    done.send(b"DONE")
    done.close(0)
    ctx.term()
    chan.close()


def run_h2d(args) -> None:
    page_bytes = args.page_size_mb * MB
    total_bytes = args.pages * page_bytes
    torch.npu.set_device(args.device)

    try:
        src = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
        pinned = "pinned"
    except Exception as e:
        print(f"[h2d] pin_memory unavailable ({e}); falling back to pageable")
        src = torch.empty(total_bytes, dtype=torch.uint8)
        pinned = "pageable"
    src.fill_(0xAB)
    dst = torch.empty(total_bytes, dtype=torch.uint8,
                      device=f"npu:{args.device}")

    for _ in range(args.warmup):
        dst.copy_(src, non_blocking=True)
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.iters):
        dst.copy_(src, non_blocking=True)
    torch.npu.synchronize()
    _report(f"h2d bulk ({pinned})", args.iters * total_bytes,
            time.perf_counter() - t0, args.iters)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        for k in range(args.pages):
            off = k * page_bytes
            dst.narrow(0, off, page_bytes).copy_(
                src.narrow(0, off, page_bytes), non_blocking=True
            )
    torch.npu.synchronize()
    _report(f"h2d paged ({pinned}, {args.pages} pages/iter)",
            args.iters * total_bytes, time.perf_counter() - t0, args.iters)

    print(
        "[hint] the L1 pull path pays this H2D copy + lookup + IPC; "
        "compare against the d2d leg with the same --page-size-mb/--pages"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark PD D2D vs L1 H2D KV delivery paths"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--page-size-mb", type=int, default=36,
                        help="KV page size in MiB (default: 36, the 1p1d/hccs "
                        "example page)")
    common.add_argument("--pages", type=int, default=64,
                        help="pages per iteration (default: 64 = 2.25 GiB)")
    common.add_argument("--iters", type=int, default=20)
    common.add_argument("--warmup", type=int, default=3)
    common.add_argument("--device", type=int, default=0)
    common.add_argument("--channel", choices=["mooncake", "hccl"],
                        default="mooncake")

    d2d = sub.add_parser("d2d", parents=[common],
                         help="PD leg: NPU-to-NPU over the transfer channel")
    d2d.add_argument("--role", choices=["sender", "receiver"], required=True)
    d2d.add_argument("--peer-host", default="127.0.0.1",
                     help="receiver address (sender side); use the remote "
                     "node IP for cross-node runs")
    d2d.add_argument("--init-port", type=int, default=7321)
    d2d.add_argument("--done-port", type=int, default=7322)
    d2d.add_argument("--bind-ip", default="0.0.0.0",
                     help="receiver bind address for ZMQ sockets")
    d2d.set_defaults(func=run_d2d)

    h2d = sub.add_parser("h2d", parents=[common],
                         help="L1 leg: pinned host pool -> NPU copies")
    h2d.set_defaults(func=run_h2d)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
