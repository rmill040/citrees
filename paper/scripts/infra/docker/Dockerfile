# citrees Docker image with R/partykit support
# Pre-built image for fast worker startup on AWS spot instances

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies + R
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    openssh-client \
    rsync \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# R packages for partykit baselines
RUN R -e 'install.packages(c("partykit", "libcoin", "mvtnorm"), repos="https://cloud.r-project.org")'

# uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY . .

# Ensure uv installs into a predictable venv and Ray is on PATH.
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install with paper dependencies
RUN uv sync --frozen --group paper

# Fix Numba/dcor LLVM cache issue - force fresh compilation at runtime
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

ENV R_HOME=/usr/lib/R
# S3_BUCKET is set at runtime via cluster.yaml (citrees-{account_id})

CMD ["uv", "run", "python", "-c", "print('citrees container ready')"]
