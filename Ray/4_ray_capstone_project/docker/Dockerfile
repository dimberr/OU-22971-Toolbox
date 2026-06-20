# Official Ray image (Ray 2.55.1 + dashboard/job server preinstalled, py313).
# No official py314 image exists; the cluster runs 3.13, which is behaviorally
# equivalent here (the code uses no 3.14-only features). We add only pandas;
# pyarrow ships with the Ray image, so we don't repin it.
FROM rayproject/ray:2.55.1-py313-cpu

RUN pip install --no-cache-dir pandas==2.3.3

WORKDIR /app
