# citrees Docker image with R/partykit support
# Pre-built image for fast worker startup on AWS instances

FROM rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c

ARG SOURCE_GIT_SHA
RUN case "$SOURCE_GIT_SHA" in *[!0-9a-f]*|"") exit 1 ;; esac \
    && test "${#SOURCE_GIT_SHA}" -eq 40
LABEL org.opencontainers.image.revision="$SOURCE_GIT_SHA"

ENV DEBIAN_FRONTEND=noninteractive

# System and Python dependencies
RUN printf '%s\n' \
    'Types: deb' \
    'URIs: https://snapshot.ubuntu.com/ubuntu/20260801T000000Z/' \
    'Suites: noble noble-updates noble-backports noble-security' \
    'Components: main universe restricted multiverse' \
    'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
    'Check-Valid-Until: no' \
    > /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential=12.10ubuntu1 \
    curl=8.5.0-2ubuntu10.11 \
    gfortran=4:13.2.0-7ubuntu1 \
    git=1:2.43.0-1ubuntu7.3 \
    openssh-client=1:9.6p1-3ubuntu13.18 \
    openssl=3.0.13-0ubuntu3.12 \
    python-is-python3=3.11.4-1 \
    python3-dev=3.12.3-0ubuntu2.1 \
    python3-venv=3.12.3-0ubuntu2.1 \
    rsync=3.2.7-1ubuntu1.5 \
    libbz2-dev=1.0.8-5.1build0.1 \
    libcurl4-openssl-dev=8.5.0-2ubuntu10.11 \
    libdeflate-dev=1.19-1build1.1 \
    liblzma-dev=5.6.1+really5.4.5-1ubuntu0.3 \
    libpcre2-dev=10.42-4ubuntu2.1 \
    libssl-dev=3.0.13-0ubuntu3.12 \
    libxml2-dev=2.9.14+dfsg-1.3ubuntu3.8 \
    libzstd-dev=1.5.5+dfsg2-2build1.1 \
    && test "$(dpkg-query --show --showformat='${Version}' openssl)" = "3.0.13-0ubuntu3.12" \
    && rm -rf /var/lib/apt/lists/*

# R packages used by the partykit baselines
RUN set -eux; \
    test "$(R --vanilla --slave -e 'cat(as.character(getRversion()))')" = "4.5.2"; \
    curl -fsSLo /tmp/Formula_1.2-5.tar.gz \
        https://cloud.r-project.org/src/contrib/Archive/Formula/Formula_1.2-5.tar.gz; \
    curl -fsSLo /tmp/inum_1.0-5.tar.gz \
        https://packagemanager.posit.co/cran/2026-08-01/src/contrib/inum_1.0-5.tar.gz; \
    curl -fsSLo /tmp/libcoin_1.0-10.tar.gz \
        https://cloud.r-project.org/src/contrib/Archive/libcoin/libcoin_1.0-10.tar.gz; \
    curl -fsSLo /tmp/mvtnorm_1.3-3.tar.gz \
        https://cloud.r-project.org/src/contrib/Archive/mvtnorm/mvtnorm_1.3-3.tar.gz; \
    curl -fsSLo /tmp/partykit_1.2-24.tar.gz \
        https://cloud.r-project.org/src/contrib/Archive/partykit/partykit_1.2-24.tar.gz; \
    printf '%s  %s\n' \
        86254674600d64e18b65d52f42d7ebfc217c8e1945cb63ac06da22cbf04d355c /tmp/Formula_1.2-5.tar.gz \
        9d1b4365f8e03f4e1e4989b7f91ea9e65c25dc171d807db9b206addbc0eb65fe /tmp/inum_1.0-5.tar.gz \
        3023e0495d0789765bdf04c0ef0990a57b48fefa322c55f20e250d2d70d67eaf /tmp/libcoin_1.0-10.tar.gz \
        e20e7f534bc89b9258ad63d508aabed60f3bd504a7532a33e6ea230ca8fa4171 /tmp/mvtnorm_1.3-3.tar.gz \
        40c06b75458f100f3e203fb47f80d5be6f632628bfafa2704d27250a7914c9ff /tmp/partykit_1.2-24.tar.gz \
        | sha256sum --check; \
    R CMD INSTALL \
        /tmp/Formula_1.2-5.tar.gz \
        /tmp/mvtnorm_1.3-3.tar.gz \
        /tmp/libcoin_1.0-10.tar.gz \
        /tmp/inum_1.0-5.tar.gz \
        /tmp/partykit_1.2-24.tar.gz; \
    R --vanilla --slave -e 'expected <- c(Formula="1.2.5", inum="1.0.5", libcoin="1.0.10", mvtnorm="1.3.3", partykit="1.2.24"); stopifnot(all(vapply(names(expected), function(package) as.character(packageVersion(package)), character(1)) == expected))'; \
    rm -f /tmp/*.tar.gz

# uv package manager
COPY --from=ghcr.io/astral-sh/uv:0.10.2@sha256:94a23af2d50e97b87b522d3cea24aaf8a1faedec1344c952767434f69585cbf9 /uv /usr/local/bin/uv

RUN uv python install 3.12.7

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY citrees ./citrees
COPY paper/benchmark ./paper/benchmark
COPY paper/jss/replication ./paper/jss/replication
COPY tests/__init__.py tests/conftest.py ./tests/
COPY tests/paper ./tests/paper

# Ensure uv installs into a predictable venv and tools are on PATH.
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV UV_PYTHON=3.12.7
ENV UV_PYTHON_DOWNLOADS=never
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV BLIS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMBA_DISABLE_JIT=0
ENV NUMBA_NUM_THREADS=32
ENV NUMEXPR_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV PYTHONHASHSEED=0
ENV R_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1

# Install with paper dependencies
RUN uv sync --frozen --group paper \
    && python -c 'import platform; assert platform.python_version() == "3.12.7"' \
    && python -c 'from importlib.metadata import version; assert version("rpy2") == "3.6.7"; assert version("scikit-learn") == "1.8.0"'

# Pre-compile Numba/dcor caches at build time so workers load cached
# artifacts instead of racing to JIT-compile simultaneously (LLVM SIGABRT).
ENV NUMBA_CACHE_DIR=/app/.numba_cache
RUN python paper/benchmark/infra/docker/warmup_numba.py

ENV R_HOME=/usr/local/lib/R
# S3_BUCKET and CITREES_ARTIFACT_PREFIX are set at runtime via env vars
# CITREES_API_URL is set at runtime for workers

# Default: worker mode (override CMD at runtime for API server)
# Worker: docker run citrees python -m paper.benchmark.api.worker --api-url http://api:8000 --stage rankings
# API:    docker run -p 8000:8000 citrees uvicorn paper.benchmark.api.server:app --host 0.0.0.0
CMD ["python", "-m", "paper.benchmark.api.worker", "--help"]
