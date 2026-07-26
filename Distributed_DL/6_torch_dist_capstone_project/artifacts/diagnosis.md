# Manual batch-size diagnosis (DevContainer)

The first step of each run is treated as warmup. Using the mean of steps 1 and
2:

| local_batch_size | global_batch_size | warm images/s | warm step time |
|---|---|---|---|
| 2 | 4 | 3.47 | 1.15 s |
| 4 | 8 | 3.68 | 2.18 s |
| 8 | 16 | 3.72 | 4.31 s |

Selected configuration: local batch size 8, because it maximizes measured
`images/s`. The gain from 4 to 8 is only about +1%, so returns are diminishing.

## Waiting regions from batch-8 mean spans

- Rank 0 `stage0_forward`: ~725 ms
- Rank 0 `recv_boundary_grad`: ~1707 ms
- Rank 0 `stage0_backward`: ~1839 ms
- Rank 1 `recv_boundary`: ~792 ms
- Rank 1 `stage1_forward`: ~411 ms

Interpretation:

1. Rank 1 blocks first in `recv_boundary` while Stage 0 prepares views and
   computes the boundary activation.
2. Rank 0 later blocks in `recv_boundary_grad` while Stage 1 finishes forward,
   gather, loss, and backward.
3. Explicit send spans stay near 1–2 ms. Waiting and stage imbalance dominate.
4. Larger batches lengthen both useful compute and blocked receive time almost
   proportionally, so throughput barely rises.

## Decision

Choose local batch size 8 for the maximum observed `images/s`. The practical
systems finding is the persistent Stage-0-heavy pipeline bubble, not a large
batch-size win under four-way CPU contention in the container.
