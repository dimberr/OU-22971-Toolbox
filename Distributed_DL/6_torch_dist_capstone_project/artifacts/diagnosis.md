# Manual batch-size diagnosis (DevContainer rerun)

The first step of each run is treated as warmup. Using the mean of steps 1 and
2 from the DevContainer rerun:

- Local batch 2, global batch 4: 3.47 images/s
- Local batch 4, global batch 8: 3.68 images/s
- Change: about +6%

Absolute throughput is much lower than the earlier host run because four CPU
processes contend inside the container. The important systems observation is
still visible in the traces.

## Process separation check

The four ranks are separate OS processes. Example PIDs from the batch-2 traces:

- rank 0: pid 46929, Stage 0 spans only
- rank 1: pid 46930, Stage 1 spans only
- rank 2: pid 46931, Stage 0 spans only
- rank 3: pid 46932, Stage 1 spans only

## Waiting regions from batch-4 mean spans

- Rank 0 `stage0_forward`: ~358 ms
- Rank 0 `recv_boundary_grad`: ~868 ms
- Rank 0 `stage0_backward`: ~933 ms
- Rank 1 `recv_boundary`: ~393 ms
- Rank 1 `stage1_forward`: ~209 ms

Interpretation:

1. Rank 1 blocks first in `recv_boundary` while Stage 0 prepares views and
   computes the boundary activation.
2. Rank 0 later blocks in `recv_boundary_grad` while Stage 1 finishes forward,
   gather, loss, and backward.
3. Explicit send spans stay under 1 ms. Waiting and stage imbalance dominate.

## Batch-size decision

Doubling local batch size roughly doubled step time, so images/s barely moved.
Batch 4 is still acceptable as the follow-up configuration, but the evidence
shows that on this contended CPU container a larger batch does not buy much
throughput. The main systems finding is the pipeline bubble, not a large
throughput win.
