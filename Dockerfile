# Instruct-to-Act — builds on top of DreamerV3's environment setup.
#
# Build:
#   docker build -t instruct-to-act .
#
# Run (see run_docker.sh for the full command):
#   docker run --gpus all --rm \
#     -v /data/terran/instruct_to_act/seed0:/logdir \
#     -e OPENAI_API_KEY=$OPENAI_API_KEY \
#     instruct-to-act \
#     --config configs/base.yaml configs/minecraft.yaml \
#     --logdir /logdir --seed 0

# ── Base: same foundation as DreamerV3 ────────────────────────────────────────
FROM ghcr.io/nvidia/driver:7c5f8932-550.144.03-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco
RUN apt-get update && apt-get install -y \
  ffmpeg git vim curl software-properties-common grep \
  libglew-dev x11-xserver-utils xvfb wget \
  && apt-get clean

# Python 3.11
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_ROOT_USER_ACTION=ignore
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.11-dev python3.11-venv && apt-get clean
RUN python3.11 -m venv /venv --upgrade-deps
ENV PATH="/venv/bin:$PATH"
RUN pip install -U pip setuptools

# ── DreamerV3 environments (verbatim from dreamerv3/Dockerfile) ───────────────
RUN wget -O - https://gist.githubusercontent.com/danijar/ca6ab917188d2e081a8253b3ca5c36d3/raw/install-dmlab.sh | sh
RUN pip install ale_py==0.9.0 autorom[accept-rom-license]==0.6.1
RUN pip install procgen_mirror
RUN pip install crafter
RUN pip install dm_control
RUN pip install memory_maze
ENV MUJOCO_GL=egl
RUN apt-get update && apt-get install -y openjdk-8-jdk && apt-get clean
# Patched MineRL wheel that removes gym<0.20 upper bound and skips Gradle build
RUN pip install https://github.com/danijar/minerl/releases/download/v0.4.4-patched/minerl_mirror-0.4.4-cp311-cp311-linux_x86_64.whl
RUN chown -R 1000:root /venv/lib/python3.11/site-packages/minerl

# ── DreamerV3 Python deps ──────────────────────────────────────────────────────
RUN pip install jax[cuda12]==0.4.33
COPY dreamerv3/requirements.txt /tmp/dreamerv3_requirements.txt
RUN pip install -r /tmp/dreamerv3_requirements.txt

# ── Instruct-to-Act additional deps ───────────────────────────────────────────
RUN pip install \
  sentence-transformers>=2.2.0 \
  openai>=1.0.0 \
  anthropic>=0.20.0 \
  scipy>=1.11.0 \
  matplotlib>=3.7.0 \
  pandas>=2.0.0 \
  scikit-learn>=1.3.0 \
  wandb>=0.15.0 \
  pillow>=10.0.0 \
  imageio>=2.31.0

# ── Project source ────────────────────────────────────────────────────────────
RUN mkdir -p /app
WORKDIR /app
COPY . .
RUN chown -R 1000:root /app

# ── Runtime env ───────────────────────────────────────────────────────────────
# Disable JAX preallocation to avoid OOM when loading large checkpoints.
# The BFC allocator + full preallocation (default) causes shard_fn recompilation
# OOM on the 800M model. platform allocator uses cudaMalloc directly.
ENV XLA_PYTHON_CLIENT_ALLOCATOR=platform
ENV PYTHONPATH="/app:/app/dreamerv3:${PYTHONPATH}"

ENTRYPOINT ["sh", "entrypoint.sh"]
