FROM nvcr.io/nvidia/cuda:12.9.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PYTORCH_VERSION=2.8.0
ENV PYTHONPATH=/workspace/src

# -------------------------------------------------------------
# System setup
# -------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git git-lfs libglib2.0-0 tmux htop xvfb \
    python3 python3-pip python-is-python3 unzip curl \
    && rm -rf /var/lib/apt/lists/*

# Unblock system package restrictions
RUN rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED

# -------------------------------------------------------------
# Install protobuf compiler
# -------------------------------------------------------------
RUN curl -LO https://github.com/protocolbuffers/protobuf/releases/download/v32.1/protoc-32.1-linux-x86_64.zip \
    && unzip protoc-32.1-linux-x86_64.zip -d /usr/local \
    && rm protoc-32.1-linux-x86_64.zip
ENV PATH="/usr/local/bin:$PATH"

# -------------------------------------------------------------
# Install Python dependencies
# -------------------------------------------------------------
# Install PyTorch from the correct index (prebuilt CUDA 12.9 wheels)
RUN pip install "torch~=2.8" "torchvision~=0.23" --index-url https://download.pytorch.org/whl/cu129
# Install remaining dependencies
RUN pip install \
    "pytorch-lightning~=2.5" \
    "torchmetrics~=1.8" \
    "hydra-core~=1.3" \
    "hydra-colorlog~=1.2" \
    "pydantic~=2.12" \
    "protobuf~=6.32" \
    "mcap~=1.3" \
    "mcap-protobuf-support~=0.5" \
    "mypy-protobuf~=3.6" \
    "pytest~=8.4" \
    "av~=16.0" \
    "scipy" \
    "pandas" \
    "debugpy" \
    "opencv-python" \
    "tqdm" \
    "numba" \
    "matplotlib" \
    "ipython" \

# -------------------------------------------------------------
# Copy the source code
# -------------------------------------------------------------
WORKDIR /workspace
COPY . .
RUN chmod 777 /workspace
RUN git config --global --add safe.directory /workspace

CMD ["/bin/bash"]
