#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv lock
uv sync
.venv/bin/python -c "import torch; print('ok', torch.__version__, 'cuda', torch.cuda.is_available())"
