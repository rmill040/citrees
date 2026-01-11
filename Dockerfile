# citrees Docker image with R/partykit support
# For local validation and distributed experiments

FROM python:3.12-slim

# Prevent interactive prompts during package installation
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

# Install R packages (partykit and dependencies)
RUN R -e 'install.packages(c("partykit", "libcoin", "mvtnorm"), repos="https://cloud.r-project.org")'

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy all source files (needed for editable install)
COPY . .

# Install dependencies and package
RUN uv sync --frozen

# Set R_HOME for rpy2
ENV R_HOME=/usr/lib/R

# Default command
CMD ["uv", "run", "python", "-c", "print('citrees container ready')"]
