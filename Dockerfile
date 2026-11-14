# GTA6 Clipping Machine: optional container image.
# CPU by default. For an NVIDIA GPU, build with a CUDA PyTorch index, for example:
#   docker compose build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
# (pick the CUDA version that matches your driver on https://pytorch.org/get-started/).

FROM python:3.12-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    GCM_DATA_DIR=/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch first, from the chosen index, so pip does not pull the default CUDA wheels.
RUN pip install torch torchvision --index-url "${TORCH_INDEX_URL}"

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

COPY configs ./configs
COPY .streamlit ./.streamlit
ENV GCM_CONFIG_DIR=/app/configs

RUN useradd --create-home --uid 1000 clipper \
    && mkdir -p /data /models \
    && chown -R clipper:clipper /data /models /app
USER clipper

VOLUME ["/data", "/models"]
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"

# Inside the container Streamlit must listen on all interfaces; docker-compose.yml publishes
# the port on 127.0.0.1 only, so the app is still reachable from this computer alone.
CMD ["clipping-machine", "run", "--host", "0.0.0.0", "--port", "8501"]
