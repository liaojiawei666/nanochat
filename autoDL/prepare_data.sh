#!/usr/bin/env bash
# nanochat data helpers (AutoDL prepare cascade + standalone OSS sync).
# Defaults: autoDL/config.py
#
# Usage:
#   export OSS_LLM_AK=...
#   export OSS_LLM_SK=...
#
#   # AutoDL cascade: local -> autodl-fs -> OSS -> HF
#   bash autoDL/prepare_data.sh
#   bash autoDL/prepare_data.sh prepare --dry-run
#
#   # Standalone OSS (works outside AutoDL too)
#   bash autoDL/prepare_data.sh upload                          # ~/.cache/nanochat -> OSS
#   bash autoDL/prepare_data.sh upload --local /path/to/nanochat
#   bash autoDL/prepare_data.sh download --local ~/.cache/nanochat
#   bash autoDL/prepare_data.sh has-objects
#   bash autoDL/prepare_data.sh upload --dry-run --skip-existing

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

need_oss2() {
  if ! python -c "import oss2" >/dev/null 2>&1; then
    echo "Missing dependency: oss2" >&2
    echo "Install with:  uv pip install oss2" >&2
    exit 1
  fi
}

cmd="${1:-prepare}"
case "$cmd" in
  -h|--help|help)
    sed -n '2,20p' "$0"
    exit 0
    ;;
  prepare)
    shift || true
    # oss2 only required if cascade reaches OSS; warn but don't hard-fail here
    if ! python -c "import oss2" >/dev/null 2>&1; then
      echo "Note: oss2 not installed (uv pip install oss2). Cascade will fail if OSS is needed." >&2
    fi
    exec python -m autoDL.prepare_data "$@"
    ;;
  upload|download|has-objects)
    shift
    need_oss2
    exec python -m autoDL.upload_to_oss "$cmd" "$@"
    ;;
  # allow flags without subcommand => prepare
  -*)
    if ! python -c "import oss2" >/dev/null 2>&1; then
      echo "Note: oss2 not installed (uv pip install oss2). Cascade will fail if OSS is needed." >&2
    fi
    exec python -m autoDL.prepare_data "$@"
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    echo "Use: prepare | upload | download | has-objects | --help" >&2
    exit 1
    ;;
esac
