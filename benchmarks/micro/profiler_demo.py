import argparse
import json
import math
import os
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, Dataset


class SlowCPUDataset(Dataset):
    """Synthetic dataset with controllable CPU work and optional fake I/O latency."""

    def __init__(self, n: int = 50000, image_size: int = 224, cpu_work: int = 2000, io_sleep_ms: int = 0):
        self.n = n
        self.image_size = image_size
        self.cpu_work = cpu_work
        self.io_sleep_ms = io_sleep_ms

    def __len__(self) -> int:
        return self.n

    def _heavy_cpu_augment(self, x: torch.Tensor) -> torch.Tensor:
        acc = 0.0
        for i in range(self.cpu_work):
            acc += math.sin(i * 0.001) * math.cos(i * 0.002)
        x = x * (1.0 + (acc % 1e-3))
        return x

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.io_sleep_ms > 0:
            time.sleep(self.io_sleep_ms / 1000.0)
        x = torch.randn(3, self.image_size, self.image_size)
        y = int(torch.randint(0, 1000, (1,)).item())
        x = self._heavy_cpu_augment(x)
        return x, y


class TinyOpsBlock(nn.Module):
    """Builds many tiny ops to amplify kernel launch overhead/fragmentation."""

    def __init__(self, channels: int = 256):
        super().__init__()
        self.ln = nn.LayerNorm(channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(30):
            x = x + 0.01 * torch.tanh(x)
            x = x * torch.sigmoid(x)
            x = self.ln(x)
            x = self.proj(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, channels_last: bool = False):
        super().__init__()
        self.channels_last = channels_last
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.head = nn.Linear(256, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.mean(dim=[2, 3])
        x = self.head(x)
        return x


def _autocast_context(enabled: bool):
    if enabled:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return torch.amp.autocast(device_type="cuda", enabled=False)


def run_step(
    model: nn.Module,
    block: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    scenario: str,
    device: torch.device,
    amp: bool = False,
) -> torch.Tensor:
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device=device, dtype=torch.long, non_blocking=True)

    if scenario == "memory_churn":
        tmp = []
        for _ in range(20):
            tmp.append(torch.randn_like(x))
        x = x + sum(t.mean() for t in tmp) * 0.001

    with _autocast_context(amp):
        logits = model(x)
        if scenario == "tiny_ops":
            feats = logits[:, :256]
            feats = block(feats)
            logits = torch.cat([feats, logits[:, 256:]], dim=1)
        loss = F.cross_entropy(logits, y)

    if scenario == "sync":
        _ = float(loss.item())
    return loss


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This demo is intended for CUDA profiling. Please run on a CUDA-capable machine.")

    if args.scenario == "dataloader":
        cpu_work = 4000
        io_sleep_ms = 2
    else:
        cpu_work = 200
        io_sleep_ms = 0

    ds = SlowCPUDataset(
        n=args.dataset_size,
        image_size=args.image_size,
        cpu_work=cpu_work,
        io_sleep_ms=io_sleep_ms,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        drop_last=True,
    )

    model = SimpleCNN(channels_last=args.channels_last).to(device)
    block = TinyOpsBlock(channels=256).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(block.parameters()), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    iterator = iter(loader)
    for _ in range(args.warmup_steps):
        batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss = run_step(model, block, batch, args.scenario, device, amp=args.amp)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    os.makedirs(args.trace_dir, exist_ok=True)
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=args.with_stack,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.trace_dir),
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=args.profile_steps, repeat=1),
    )

    step_times: List[float] = []
    data_wait_times: List[float] = []
    end_prev = time.time()

    with prof:
        for step, batch in enumerate(loader):
            if step >= (1 + 1 + args.profile_steps):
                break

            now = time.time()
            data_wait_times.append(now - end_prev)

            torch.cuda.synchronize()
            t0 = time.time()

            optimizer.zero_grad(set_to_none=True)
            with record_function("train_step"):
                loss = run_step(model, block, batch, args.scenario, device, amp=args.amp)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            torch.cuda.synchronize()
            t1 = time.time()
            step_times.append(t1 - t0)

            prof.step()
            end_prev = time.time()

    stats: Dict[str, object] = {
        "scenario": args.scenario,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory),
        "amp": bool(args.amp),
        "avg_step_time_s": _safe_mean(step_times),
        "avg_dataloader_wait_s": _safe_mean(data_wait_times),
        "trace_dir": args.trace_dir,
    }

    print("\n==== Quick stats ====")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print("\n==== Top ops by self CUDA time ====")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=args.row_limit))
    print(f"\nTrace saved to: {args.trace_dir}")
    print("Open with: tensorboard --logdir <trace_dir> (Profiler tab)")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Summary saved to: {args.output_json}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU profiler sandbox demo for bottleneck analysis.")
    p.add_argument(
        "--scenario",
        type=str,
        default="dataloader",
        choices=["dataloader", "sync", "tiny_ops", "memory_churn"],
        help="Bottleneck scenario to emulate.",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--dataset_size", type=int, default=100000)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--channels_last", action="store_true")
    p.add_argument("--with_stack", action="store_true")
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--profile_steps", type=int, default=8)
    p.add_argument("--row_limit", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trace_dir", type=str, default="results/traces/profiler_demo")
    p.add_argument("--output_json", type=str, default="results/metrics_demo.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
