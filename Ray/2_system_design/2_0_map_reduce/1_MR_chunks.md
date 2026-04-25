# Ray Unit 2 - MR Chunks on the Cluster



## Setup

1. Make sure the Docker Ray cluster from the previous unit is already running.

2. The Ray Dashboard / Jobs API should be exposed at:

```text
http://127.0.0.1:8265
```

3. Run the commands below from `Ray/2_system_design/2_0_map_reduce`.

---
## `MR_chunks.py`

The script mirrors the notebook's final MapReduce version:

- each mapper emits one completion token (its batch_id) plus one bucket per reducer
- the driver uses `ray.wait()` to route ready bucket refs in mapper completion order
- every batch of ready buckets is collapsed by a remote partial reduce
- each reducer stream ends with one small final merge

This job version also reads its seed corpus from a bundled text file, `mr_job_docs.txt`.

---
## Submit the job

### 1. Short synchronous run


```powershell
ray job submit --address http://127.0.0.1:8265 --working-dir . -- python MR_chunks.py --address auto --output-file /workspace/mr_chunks_output.json
```

If the synchronous run succeeds, you will see the script's console output directly in your terminal because the Jobs CLI streams the logs back to the client.

### 2. Longer asynchronous run

For a longer run that is easier to observe in the logs, submit it asynchronously:

```powershell
ray job submit --no-wait --address http://127.0.0.1:8265 --working-dir . -- python MR_chunks.py --address auto --repeat 80 --reduce-batch-size 12 --straggler-delay-s 1.5 --reduce-delay-per-bucket-s 0.03 --output-file /workspace/mr_chunks_output.json
```

### Useful script flags

- `--repeat` expands the synthetic corpus
- `--docs-per-chunk` controls mapper fan-out
- `--reduce-batch-size` controls when each reducer stream launches a partial reduce
- `--straggler-delay-s` makes the last mapper noticeably slow
- `--reduce-delay-per-bucket-s` makes reducer batching visible in the logs

### Notes

- `--address http://127.0.0.1:8265` points the Jobs CLI at the Dashboard / Jobs API endpoint.
- `--address auto` inside the Python script tells `ray.init()` to connect to the already-running cluster.
- The entrypoint script runs on the head node by default.

---


## Check job status and logs

When you use `--no-wait`, the submit command returns a Ray job ID. Then you can inspect the run from PowerShell:

```powershell
ray job status <JOB_ID> --address http://127.0.0.1:8265
ray job logs <JOB_ID> --address http://127.0.0.1:8265
```

Use `ray job status` to confirm that the job moved from `RUNNING` to `SUCCEEDED`.
Use `ray job logs` to retrieve the driver-side output again after submission.

verify in the logs:

- the driver is running from a packaged working directory on the head container
- `mr_job_docs.txt` was loaded from that packaged directory
- the global count still matches a direct `Counter(flatten_tokens(...))` check
- the artifact file was written to `/workspace/mr_chunks_output.json`

---

## Inspect the packaged working directory

### Why `--working-dir .` matters

Because this job reads a bundled text file, `ray job submit --working-dir .` gives us a clear view into job submission mechanics: it packages the current folder and ships it to the driver process on the head container.

The most reliable thing to inspect is the unpacked runtime-env directory on the head container.
Ray's runtime-env code names working-directory packages like `_ray_pkg_<hash>.zip`, downloads them to the node, unpacks them under `runtime_resources/working_dir_files`, and removes the temporary zip after extraction.

After a job runs, inspect the unpacked package on the head container:

```powershell
docker exec ray-head bash -lc "ls -1 /tmp/ray/session_latest/runtime_resources/working_dir_files"
```

Then inspect the contents of the unpacked package directories:

```powershell
docker exec ray-head bash -lc "find /tmp/ray/session_latest/runtime_resources/working_dir_files -maxdepth 2 -type f | sort"
```

You should see this unit's files there, including `MR_chunks.py` and `mr_job_docs.txt`.

---

## Retrieve the output artifact

The script writes a small JSON summary artifact to:

```text
/workspace/mr_chunks_output.json
```

In the Docker cluster from the setup chapter, `/workspace` on the head container is bind-mounted to `Ray/1_cluster_setup/head_workspace` in the toolbox repo.
So after the run finishes, the same file should be available on the host as:

```text
repo_root/Ray/1_cluster_setup/head_workspace/mr_chunks_output.json
```

The artifact includes:

- driver host and packaged working-directory paths
- the bundled docs file path that was read on the driver
- the run configuration
- mapper completion order
- one summary block per reducer
- the global top words
- a final correctness check against a direct count
