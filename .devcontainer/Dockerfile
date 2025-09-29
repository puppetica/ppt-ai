FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

RUN apt-get update && apt-get install -y git

ENV PYTHONPATH=/workspace/src

RUN pip install --no-cache-dir \
    pytorch-lightning==2.2.4 \
    torchmetrics~=1.4 \
    hydra-core~=1.3 \
    hydra_colorlog~=1.2 \
    pydantic~=2.11 \
    matplotlib~=3.10 \
    tensorboard~=2.20 \
    pytest~=8.4 \
    debugpy \
    ipython

WORKDIR /workspace
CMD ["/bin/bash"]

RUN git config --global --add safe.directory /workspace
