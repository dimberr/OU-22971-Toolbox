"""Train the manually partitioned two-stage pipeline.

Launch with torchrun:
  torchrun --standalone --nproc_per_node=4 train_pipeline.py \\
    --steps 3 --local-batch-size 2 --profiler --output-dir artifacts/batch2
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function
from torchvision import datasets, transforms

from setup_comm_groups import build_comm_groups, print_comm_structure
from setup_model_stages import (
    BOUNDARY_SHAPE,
    EMBEDDING_DIM,
    assert_replicas_aligned,
    assert_stage_shapes,
    broadcast_module_state,
    build_and_verify_optimizer,
    build_owned_stage,
)


TEMPERATURE = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--profiler", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baseline"))
    return parser.parse_args()


def build_augmentation() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]
    )


def prepare_views(
    dataset: datasets.FakeData,
    augmentation: transforms.Compose,
    step: int,
    rank: int,
    local_batch_size: int,
    seed: int,
    pair_count: int,
) -> torch.Tensor:
    torch.manual_seed(seed + 1_000 * step + rank)
    replica_id = rank // 2
    images = []
    for local_index in range(local_batch_size):
        global_index = (
            (step * local_batch_size + local_index) * pair_count + replica_id
        ) % len(dataset)
        image, _ = dataset[global_index]
        images.append(image)
    first_views = [augmentation(image) for image in images]
    second_views = [augmentation(image) for image in images]
    return torch.stack(first_views + second_views)


def gather_global_embeddings(
    local_embeddings: torch.Tensor,
    stage1_group: dist.ProcessGroup,
    rank: int,
) -> torch.Tensor:
    gathered = [
        torch.empty_like(local_embeddings)
        for _ in range(dist.get_world_size(stage1_group))
    ]
    dist.all_gather(gathered, local_embeddings.detach(), group=stage1_group)
    local_group_rank = rank // 2
    gathered[local_group_rank] = local_embeddings
    return torch.cat(gathered)


def approximate_contrastive_loss(
    local_embeddings: torch.Tensor,
    global_embeddings: torch.Tensor,
    rank: int,
    local_batch_size: int,
) -> torch.Tensor:
    local_group_rank = rank // 2
    local_normalized = F.normalize(local_embeddings, dim=1)
    global_normalized = F.normalize(global_embeddings, dim=1)
    logits = local_normalized @ global_normalized.T / TEMPERATURE

    local_view_count = 2 * local_batch_size
    offset = local_group_rank * local_view_count
    local_indices = torch.arange(local_view_count)
    self_indices = local_indices + offset
    positive_local_indices = torch.where(
        local_indices < local_batch_size,
        local_indices + local_batch_size,
        local_indices - local_batch_size,
    )
    targets = positive_local_indices + offset
    self_mask = F.one_hot(
        self_indices,
        num_classes=global_embeddings.shape[0],
    ).bool()
    logits = logits.masked_fill(self_mask, torch.finfo(logits.dtype).min)
    return F.cross_entropy(logits, targets)


def synchronize_gradients(
    module: torch.nn.Module,
    group: dist.ProcessGroup,
) -> None:
    group_size = dist.get_world_size(group)
    for parameter in module.parameters():
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("Encountered non-finite parameter gradient")
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=group)
        parameter.grad.div_(group_size)


def assert_parameter_replicas_aligned(
    module: torch.nn.Module,
    group: dist.ProcessGroup,
) -> None:
    sample = next(module.parameters()).detach().flatten()[:16].clone()
    gathered = [torch.empty_like(sample) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, sample, group=group)
    if not all(torch.allclose(gathered[0], value) for value in gathered[1:]):
        raise RuntimeError("Stage parameter replicas diverged after optimizer step")


def collect_step_metrics(
    elapsed: float,
    local_loss: float,
    world_size: int,
) -> tuple[float, float, list[float]]:
    elapsed_tensor = torch.tensor([elapsed], dtype=torch.float64)
    rank_times = [torch.zeros_like(elapsed_tensor) for _ in range(world_size)]
    dist.all_gather(rank_times, elapsed_tensor)

    max_time = elapsed_tensor.clone()
    dist.all_reduce(max_time, op=dist.ReduceOp.MAX)
    loss_tensor = torch.tensor([local_loss], dtype=torch.float64)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    mean_loss = loss_tensor.item() / (world_size // 2)
    return max_time.item(), mean_loss, [value.item() for value in rank_times]


def write_outputs(
    args: argparse.Namespace,
    world_size: int,
    metrics: list[dict[str, float | int]],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset_size": args.dataset_size,
        "seed": args.seed,
        "local_batch_size": args.local_batch_size,
        "global_batch_size": args.local_batch_size * (world_size // 2),
        "steps": args.steps,
        "world_size": world_size,
        "profiler": args.profiler,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    with (args.output_dir / "metrics.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)


def run_stage0_step(
    module: torch.nn.Module,
    dataset: datasets.FakeData,
    augmentation: transforms.Compose,
    step: int,
    rank: int,
    args: argparse.Namespace,
    world_size: int,
    pair_group: dist.ProcessGroup,
    stage_group: dist.ProcessGroup,
) -> float:
    """Run Stage 0: views -> forward -> send X -> receive dL/dX -> backward."""
    with record_function("prepare_views"):
        views = prepare_views(
            dataset,
            augmentation,
            step,
            rank,
            args.local_batch_size,
            args.seed,
            world_size // 2,
        )
    with record_function("stage0_forward"):
        boundary = module(views)
    expected_shape = (2 * args.local_batch_size, *BOUNDARY_SHAPE)
    if tuple(boundary.shape) != expected_shape:
        raise RuntimeError(f"Boundary shape {tuple(boundary.shape)} != {expected_shape}")

    with record_function("send_boundary"):
        dist.send(
            boundary.detach().contiguous(),
            dst=rank + 1,
            group=pair_group,
        )
    returned_gradient = torch.empty_like(boundary)
    with record_function("recv_boundary_grad"):
        dist.recv(returned_gradient, src=rank + 1, group=pair_group)
    if not torch.isfinite(returned_gradient).all():
        raise RuntimeError("Encountered non-finite boundary gradient")

    with record_function("stage0_backward"):
        boundary.backward(returned_gradient)
    with record_function("grad_sync_stage0"):
        synchronize_gradients(module, stage_group)
    return 0.0


def run_stage1_step(
    module: torch.nn.Module,
    rank: int,
    args: argparse.Namespace,
    pair_group: dist.ProcessGroup,
    stage_group: dist.ProcessGroup,
) -> float:
    """Run Stage 1: receive X -> loss/backward -> send dL/dX -> sync gradients."""
    boundary = torch.empty(2 * args.local_batch_size, *BOUNDARY_SHAPE)
    with record_function("recv_boundary"):
        dist.recv(boundary, src=rank - 1, group=pair_group)
    boundary.requires_grad_()

    with record_function("stage1_forward"):
        embeddings = module(boundary)
    expected_shape = (2 * args.local_batch_size, EMBEDDING_DIM)
    if tuple(embeddings.shape) != expected_shape:
        raise RuntimeError(f"Embedding shape {tuple(embeddings.shape)} != {expected_shape}")
    if not torch.isfinite(embeddings).all():
        raise RuntimeError("Encountered non-finite embeddings")

    with record_function("gather_embeddings"):
        global_embeddings = gather_global_embeddings(
            embeddings,
            stage_group,
            rank,
        )
    with record_function("loss_calculation"):
        loss = approximate_contrastive_loss(
            embeddings,
            global_embeddings,
            rank,
            args.local_batch_size,
        )
    if not torch.isfinite(loss):
        raise RuntimeError("Encountered non-finite loss")
    loss.backward()
    if boundary.grad is None:
        raise RuntimeError("Boundary gradient was not calculated")

    with record_function("send_boundary_grad"):
        dist.send(
            boundary.grad.detach().contiguous(),
            dst=rank - 1,
            group=pair_group,
        )
    with record_function("grad_sync_stage1"):
        synchronize_gradients(module, stage_group)
    return loss.item()


def train(args: argparse.Namespace) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    groups = build_comm_groups(world_size)
    print_comm_structure(rank, world_size, groups)
    stage_id, module = build_owned_stage(rank)

    stage_group = (
        groups["stage0_group"] if stage_id == 0 else groups["stage1_group"]
    )
    source_rank = 0 if stage_id == 0 else 1
    assert isinstance(stage_group, dist.ProcessGroup)
    broadcast_module_state(module, source_rank, stage_group)
    assert_replicas_aligned(module, stage_group)
    assert_stage_shapes(stage_id, module)
    optimizer = build_and_verify_optimizer(module)
    module.train()

    pair = groups["pair_groups"][rank // 2]
    pair_group = pair["group"]
    assert isinstance(pair_group, dist.ProcessGroup)

    dataset = None
    augmentation = None
    if stage_id == 0:
        dataset = datasets.FakeData(
            size=args.dataset_size,
            image_size=(3, 224, 224),
            num_classes=1_000,
            random_offset=args.seed,
        )
        augmentation = build_augmentation()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    profiler_context = (
        profile(activities=[ProfilerActivity.CPU])
        if args.profiler
        else nullcontext()
    )
    metrics: list[dict[str, float | int]] = []

    with profiler_context as active_profiler:
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()

            if stage_id == 0:
                assert dataset is not None
                assert augmentation is not None
                local_loss = run_stage0_step(
                    module,
                    dataset,
                    augmentation,
                    step,
                    rank,
                    args,
                    world_size,
                    pair_group,
                    stage_group,
                )
            else:
                local_loss = run_stage1_step(
                    module,
                    rank,
                    args,
                    pair_group,
                    stage_group,
                )

            with record_function("optimizer_step"):
                optimizer.step()
            assert_parameter_replicas_aligned(module, stage_group)

            elapsed = time.perf_counter() - started
            max_time, mean_loss, rank_times = collect_step_metrics(
                elapsed,
                local_loss,
                world_size,
            )
            global_batch_size = args.local_batch_size * (world_size // 2)
            images_per_second = global_batch_size / max_time
            if rank == 0:
                row: dict[str, float | int] = {
                    "step": step,
                    "local_batch_size": args.local_batch_size,
                    "global_batch_size": global_batch_size,
                    "images_per_second": images_per_second,
                    "step_time_s": max_time,
                    "loss": mean_loss,
                }
                for rank_id, rank_time in enumerate(rank_times):
                    row[f"rank_{rank_id}_step_time_s"] = rank_time
                metrics.append(row)
                print(
                    f"step={step} loss={mean_loss:.4f} "
                    f"images/s={images_per_second:.2f} "
                    f"step_time={max_time:.4f}s",
                    flush=True,
                )
            if args.profiler:
                active_profiler.step()

    if args.profiler:
        active_profiler.export_chrome_trace(
            str(args.output_dir / f"trace_rank{rank}.json")
        )
    dist.barrier()
    if rank == 0:
        write_outputs(args, world_size, metrics)


def main() -> None:
    args = parse_args()
    dist.init_process_group(backend="gloo")
    try:
        train(args)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
