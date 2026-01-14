# citrees Docker image for distributed experiments
#
# Build:   docker build -t citrees:latest .
# Server:  docker run -e TABLE_NAME=... -e S3_BUCKET=... citrees:latest server
# Worker:  docker run -e URL=... -e TABLE_NAME=... -e S3_BUCKET=... citrees:latest worker

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# R packages (partykit for validation)
RUN R -e 'install.packages(c("partykit", "libcoin", "mvtnorm"), repos="https://cloud.r-project.org")'

# uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy source
COPY . .

# Install all dependencies including paper/bench groups
RUN uv sync --frozen --group paper

# Environment variables (override at runtime)
ENV R_HOME=/usr/lib/R
ENV TABLE_NAME=""
ENV S3_BUCKET=""
ENV AWS_DEFAULT_REGION="us-east-1"
ENV URL=""

# Entrypoint
COPY paper/scripts/docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["help"]
