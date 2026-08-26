FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    MAS_FORCE_DISABLE_TORCHVISION=1

# System packages: git for VCS operations, build-essential for any source builds,
# ca-certificates for HTTPS (PyPI + Hugging Face Hub downloads).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Keep pip current; project dependencies are installed from requirements.txt
# during the environment `install` step (after the repository is checked out).
RUN python -m pip install --upgrade pip
