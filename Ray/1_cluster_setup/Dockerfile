# Base Linux image with Miniconda already installed.
FROM continuumio/miniconda3:26.1.1-1

# All project files inside the container will live here.
WORKDIR /workspace

# Compose builds from the Ray part root, so this copies the part-local env spec.
COPY environment.yml /tmp/environment.yml

# Create the full course environment inside the image.
# Every container built from this image will have the same Python stack.
RUN conda env create -f /tmp/environment.yml && conda clean -afy
