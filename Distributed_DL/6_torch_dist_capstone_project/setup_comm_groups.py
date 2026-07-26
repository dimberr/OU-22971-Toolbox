"""Create and print the capstone communication groups.

Launch with torchrun:
  torchrun --standalone --nproc_per_node=4 setup_comm_groups.py
"""

from __future__ import annotations

import torch.distributed as dist


def build_comm_groups(world_size: int) -> dict[str, object]:
    if world_size < 4 or world_size % 2 != 0:
        raise SystemExit(
            f"Need even world_size >= 4, got world_size={world_size}. "
            "Launch with nproc_per_node=4."
        )

    # Every rank must create groups in this same order. Membership decides who
    # may communicate later; creation itself is a world-wide coordinated call.
    pair_groups = []
    for pair_id in range(world_size // 2):
        members = [2 * pair_id, 2 * pair_id + 1]
        pair_groups.append(
            {
                "pair_id": pair_id,
                "members": members,
                "group": dist.new_group(ranks=members),
            }
        )

    stage0_members = list(range(0, world_size, 2))
    stage1_members = list(range(1, world_size, 2))
    stage0_group = dist.new_group(ranks=stage0_members)
    stage1_group = dist.new_group(ranks=stage1_members)

    return {
        "world_members": list(range(world_size)),
        "pair_groups": pair_groups,
        "stage0_members": stage0_members,
        "stage0_group": stage0_group,
        "stage1_members": stage1_members,
        "stage1_group": stage1_group,
    }


def print_comm_structure(rank: int, world_size: int, groups: dict[str, object]) -> None:
    if rank != 0:
        return

    print("=== communication structure ===")
    print(f"world_group members: {groups['world_members']}")
    print(f"world_size: {world_size}")
    for pair in groups["pair_groups"]:
        print(f"pair_group({pair['pair_id']}) members: {pair['members']}")
    print(f"stage0_group members: {groups['stage0_members']}")
    print(f"stage1_group members: {groups['stage1_members']}")
    print("=== end communication structure ===")


def main() -> None:
    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        groups = build_comm_groups(world_size)
        print_comm_structure(rank, world_size, groups)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
