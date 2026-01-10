#!/usr/bin/env bash
# Install dependencies for running paper experiments
#
# This script sets up the environment for running citrees experiments
# on AWS EC2 instances or local machines.
#
# Usage:
#   ./install_dependencies.sh
#
# Requirements:
#   - Linux (tested on Amazon Linux 2, Ubuntu 22.04)
#   - sudo access for system packages

set -e

echo "=== Installing system dependencies ==="
if command -v yum &> /dev/null; then
    # Amazon Linux / RHEL
    sudo yum update -y
    sudo yum install -y gcc sqlite-devel openssl-devel bzip2-devel libffi-devel tmux git
elif command -v apt-get &> /dev/null; then
    # Ubuntu / Debian
    sudo apt-get update
    sudo apt-get install -y build-essential libsqlite3-dev libssl-dev libbz2-dev libffi-dev tmux git curl
else
    echo "ERROR: Unsupported package manager. Please install dependencies manually."
    exit 1
fi

echo "=== Installing uv (Python package manager) ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env 2>/dev/null || source $HOME/.local/bin/env 2>/dev/null || true

echo "=== Installing Python 3.12 and project dependencies ==="
cd "$(dirname "$0")/../.."
uv python install 3.12
uv sync --all-extras

echo "=== Installation complete ==="
echo "Run experiments with: uv run python paper/scripts/<script>.py"
