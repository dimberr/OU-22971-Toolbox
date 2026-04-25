from pathlib import Path
import socket

import ray


def announce_task(task_name: str, label: str, amount: float, i: int) -> str:
    host = socket.gethostname()
    msg = (
        f"{task_name} requested {amount:g} units of resource '{label}' "
        f"for task {i} and ran on {host}"
    )
    print(msg, flush=True)
    return msg


# Demo-only instrumentation: these are arbitrary user-defined resource labels.
# We attach them in docker-compose.yml so placement is easy to see in the smoke test.
@ray.remote(resources={"dragon balls": 7.0})
def make_wish(i: int) -> str:
    return announce_task("make_wish", "dragon balls", 7.0, i)


# This label exists only on the worker containers because we chose to advertise it there.
@ray.remote(resources={"spice melange": 0.01})
def hyperspace_jump(i: int) -> str:
    return announce_task("hyperspace_jump", "spice melange",0.01, i)


if __name__ == "__main__":
    ray.init(address="auto")

    print(f"Driver connected from {socket.gethostname()}", flush=True)
    results = ray.get(
        [
            make_wish.remote(0),
            make_wish.remote(1),
            hyperspace_jump.remote(2),
            hyperspace_jump.remote(3),
        ]
    )
    print("Results:", results, flush=True)

    out_path = Path("/workspace/smoke_test_output.txt")
    out_path.write_text(
        "Ray smoke test succeeded\n" + "\n".join(results) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote artifact to {out_path}", flush=True)
