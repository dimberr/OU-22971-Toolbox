"""Split ResNet18 into two stages and align stage replicas.

Launch with torchrun:
  torchrun --standalone --nproc_per_node=4 setup_model_stages.py
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn
from torchvision.models import resnet18

from setup_comm_groups import build_comm_groups


EMBEDDING_DIM = 128
BOUNDARY_SHAPE = (128, 28, 28)
LEARNING_RATE = 0.01


def build_stage0(model: nn.Module) -> nn.Module:
    return nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
    )


def build_stage1(model: nn.Module, embedding_dim: int) -> nn.Module:
    return nn.Sequential(
        model.layer3,
        model.layer4,
        model.avgpool,
        nn.Flatten(start_dim=1),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, embedding_dim),
    )


def build_owned_stage(rank: int) -> tuple[int, nn.Module]:
    torch.manual_seed(1_000 + rank)
    model = resnet18(weights=None)
    if rank % 2 == 0:
        return 0, build_stage0(model)
    return 1, build_stage1(model, EMBEDDING_DIM)


def broadcast_module_state(
    module: nn.Module,
    source_rank: int,
    group: dist.ProcessGroup,
) -> None:
    """Copy one rank's complete module state to its stage replicas.

    Every member of ``group`` calls each broadcast in the same order.

    Parameters are trainable tensors updated later by SGD, for example
    ``conv1.weight`` or ``Linear.weight``. Buffers are persistent non-trainable
    state such as BatchNorm ``running_mean`` and ``running_var``. Both must be
    aligned so replicas start identical.
    """
    for parameter in module.parameters():
        dist.broadcast(parameter.data, src=source_rank, group=group)
    for buffer in module.buffers():
        dist.broadcast(buffer, src=source_rank, group=group)


def assert_replicas_aligned(
    module: nn.Module,
    group: dist.ProcessGroup,
) -> None:
    """Smoke-test that every replica in a stage group has identical state.

    ``state_dict()`` contains both parameters and persistent buffers. Each rank
    sums its state into one checksum, ``all_gather`` collects one checksum from
    every group member, and the comparison fails if any checksum differs. This
    is a lightweight initialization check, not a collision-proof equality test.
    """
    checksum = torch.tensor(
        [sum(tensor.detach().double().sum().item() for tensor in module.state_dict().values())],
        dtype=torch.float64,
    )
    gathered = [torch.zeros_like(checksum) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, checksum, group=group)
    if not all(torch.equal(gathered[0], value) for value in gathered[1:]):
        raise RuntimeError(f"Stage replicas differ: {[value.item() for value in gathered]}")


def assert_stage_shapes(stage_id: int, module: nn.Module) -> None:
    """Run an inference-only probe and verify the stage's output contract.

    ``eval()`` prevents BatchNorm from changing its running statistics during
    this check. ``no_grad()`` avoids building an autograd graph that is not
    needed for shape validation.
    """
    module.eval()
    with torch.no_grad():
        if stage_id == 0:
            output = module(torch.zeros(1, 3, 224, 224))
            expected = (1, *BOUNDARY_SHAPE)
        else:
            output = module(torch.zeros(1, *BOUNDARY_SHAPE))
            expected = (1, EMBEDDING_DIM)
    if tuple(output.shape) != expected:
        raise RuntimeError(
            f"Stage {stage_id} output shape {tuple(output.shape)} != {expected}"
        )


def build_and_verify_optimizer(module: nn.Module) -> torch.optim.Optimizer:
    """Create local SGD and verify it references exactly this stage's parameters.

    The optimizer stores references to trainable tensors such as
    ``conv1.weight``. It does not copy those tensors, include BatchNorm buffers,
    or reference parameters owned by the other pipeline stage.
    """
    optimizer = torch.optim.SGD(module.parameters(), lr=LEARNING_RATE)
    module_parameter_ids = {id(parameter) for parameter in module.parameters()}
    optimizer_parameter_ids = {
        id(parameter)
        for parameter_group in optimizer.param_groups
        for parameter in parameter_group["params"]
    }
    if optimizer_parameter_ids != module_parameter_ids:
        raise RuntimeError("Optimizer does not own exactly the local stage parameters")
    return optimizer


def setup_and_verify() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    groups = build_comm_groups(world_size)
    stage_id, module = build_owned_stage(rank)

    if stage_id == 0:
        group = groups["stage0_group"]
        source_rank = 0
    else:
        group = groups["stage1_group"]
        source_rank = 1

    assert isinstance(group, dist.ProcessGroup)
    broadcast_module_state(module, source_rank, group)
    assert_replicas_aligned(module, group)
    assert_stage_shapes(stage_id, module)
    optimizer = build_and_verify_optimizer(module)

    parameter_count = sum(parameter.numel() for parameter in module.parameters())
    print(
        f"rank={rank} stage={stage_id} source={source_rank} "
        f"parameters={parameter_count} aligned=yes shapes=ok "
        f"optimizer={optimizer.__class__.__name__} optimizer_ownership=ok",
        flush=True,
    )
    dist.barrier()


def main() -> None:
    dist.init_process_group(backend="gloo")
    try:
        setup_and_verify()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
