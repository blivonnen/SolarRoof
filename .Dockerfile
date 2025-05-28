# ---------------------------------------------------------------------------
# Custom training image for OVHcloud AI Training
# Docs: https://help.ovhcloud.com/csm/en-gb-public-cloud-ai-training-build-use-custom-image
# ---------------------------------------------------------------------------

# FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04
FROM nvcr.io/nvidia/tensorflow:24.02-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive

# 1️⃣  System Python + pip (≈ 55 MB)
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#         python3 python3-pip python3-distutils ca-certificates && \
#     rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
ENV HOME=/workspace

# 2️⃣  Copy source & minimal data sample
COPY main.py         /workspace/
COPY requirements.txt /workspace/
COPY roof-segmentation-control-net/data/train-00000-of-00001.parquet \
     /workspace/roof-segmentation-control-net/data/train-00000-of-00001.parquet

# 3️⃣  Install Python deps
#     TensorFlow wheel already bundles the matching CUDA/CuDNN libraries.
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4️⃣  Ensure files are writeable by OVH unprivileged uid (42420)
RUN chown -R 42420:42420 /workspace

# 5️⃣  Default command (override in job YAML if needed)
CMD ["python3", "main.py"]

# ovhai data store add s3 my-os https://s3.bhs.io.cloud.ovh.net/ bhs aa247a0c5c5e450187e5a25dd103df8c 871a6013aa88418b8a7af5a9edb1627f


# ovhai job run --name solar-roof-tedious-kamerlingh-onnes --flavor ai1-le-1-gpu -v blivonnen-s3@my-os:/model-output:rw blivonnen/solar-roof:latest