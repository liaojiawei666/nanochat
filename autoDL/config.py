"""
Default configuration for AutoDL nanochat data sync.
Override via CLI flags or environment variables where noted.
"""

from __future__ import annotations

import os
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths

def default_local_dir() -> str:
    env = os.environ.get("NANOCHAT_BASE_DIR")
    if env:
        return env
    return str(Path.home() / ".cache" / "nanochat")


LOCAL_DIR = default_local_dir()
AUTODL_FS_DIR = "/root/autodl-fs/nanochat"
DATA_SUBDIR = "base_data_climbmix"
TOKENIZER_SUBDIR = "tokenizer"
TOKENIZER_FILES = ("tokenizer.pkl", "token_bytes.pt")
TASK_SUBDIR = "task_data"
EVAL_BUNDLE_DIRNAME = "eval_bundle"

# -----------------------------------------------------------------------------
# Completeness / HF download (pretrain)

MIN_SHARDS = 170
NUM_SHARDS = 170  # HF download count for python -m nanochat.dataset -n

# -----------------------------------------------------------------------------
# SFT / RL / chat_eval datasets (cached under task_data/<repo>/<subset>/<split>/)
# Prefetch via tasks.common.load_hub_dataset so training later hits local cache.

TASK_DATASETS = (
    # SFT train/val
    {"repo_id": "HuggingFaceTB/smol-smoltalk", "subset": "default", "split": "train"},
    {"repo_id": "HuggingFaceTB/smol-smoltalk", "subset": "default", "split": "test"},
    {"repo_id": "cais/mmlu", "subset": "all", "split": "auxiliary_train"},
    {"repo_id": "cais/mmlu", "subset": "all", "split": "test"},
    {"repo_id": "openai/gsm8k", "subset": "main", "split": "train"},
    {"repo_id": "openai/gsm8k", "subset": "main", "split": "test"},
    # chat_eval (+ RL already covered by gsm8k)
    {"repo_id": "allenai/ai2_arc", "subset": "ARC-Easy", "split": "test"},
    {"repo_id": "allenai/ai2_arc", "subset": "ARC-Challenge", "split": "test"},
    {"repo_id": "openai/openai_humaneval", "subset": "openai_humaneval", "split": "test"},
)

# base_eval CORE metric bundle
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
EVAL_BUNDLE_ZIP = "eval_bundle.zip"

# -----------------------------------------------------------------------------
# OSS

OSS_BUCKET = os.environ.get("OSS_BUCKET", "llm-data-1")
OSS_ENDPOINT = os.environ.get("OSS_ENDPOINT", "https://oss-cn-beijing.aliyuncs.com")
OSS_PREFIX = "nanochat/"
OSS_AK_ENV = "OSS_LLM_AK"
OSS_SK_ENV = "OSS_LLM_SK"

# -----------------------------------------------------------------------------
# Transfer

WORKERS = 8
SKIP_EXISTING = True
MULTIPART_THRESHOLD = 100 * 1024 * 1024
PART_SIZE = 16 * 1024 * 1024
