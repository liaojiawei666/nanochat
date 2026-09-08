"""
Prepare nanochat training data on AutoDL.

Cascade:
  1) local complete  -> done
  2) autodl-fs complete -> copy to local
  3) OSS has objects -> download to autodl-fs
  4) fill any gaps from HF (climbmix / tokenizer / task_data / eval_bundle)
     -> upload OSS if we filled gaps -> copy to local

Covers pretrain + SFT + RL + chat/base eval caches. Defaults: autoDL.config.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from autoDL import config as cfg
from autoDL.upload_to_oss import has_remote_objects, make_bucket


def resolve_dir(path: str) -> Path:
    return Path(path).expanduser().resolve()


def count_parquet(data_dir: Path) -> int:
    if not data_dir.is_dir():
        return 0
    return sum(1 for p in data_dir.glob("*.parquet") if p.is_file() and not p.name.endswith(".tmp"))


def has_tokenizer(root: Path) -> bool:
    tok_dir = root / cfg.TOKENIZER_SUBDIR
    return all((tok_dir / name).is_file() for name in cfg.TOKENIZER_FILES)


def task_manifest_path(root: Path, repo_id: str, subset: str, split: str) -> Path:
    slug = repo_id.replace("/", "--")
    return root / cfg.TASK_SUBDIR / slug / subset / split / "manifest.json"


def missing_task_datasets(root: Path) -> list[dict]:
    missing = []
    for spec in cfg.TASK_DATASETS:
        if not task_manifest_path(root, spec["repo_id"], spec["subset"], spec["split"]).is_file():
            missing.append(spec)
    return missing


def has_eval_bundle(root: Path) -> bool:
    bundle = root / cfg.EVAL_BUNDLE_DIRNAME
    return (bundle / "core.yaml").is_file() and (bundle / "eval_data").is_dir()


def has_complete_data(root: Path, min_shards: int) -> bool:
    data_dir = root / cfg.DATA_SUBDIR
    n = count_parquet(data_dir)
    tok_ok = has_tokenizer(root)
    missing_tasks = missing_task_datasets(root)
    bundle_ok = has_eval_bundle(root)
    ok = n >= min_shards and tok_ok and not missing_tasks and bundle_ok
    print(
        f"  check {root}: "
        f"shards={n}/{min_shards} "
        f"tokenizer={'yes' if tok_ok else 'no'} "
        f"tasks_missing={len(missing_tasks)}/{len(cfg.TASK_DATASETS)} "
        f"eval_bundle={'yes' if bundle_ok else 'no'} "
        f"-> {'OK' if ok else 'MISSING'}"
    )
    return ok


def copy_tree(src: Path, dst: Path, dry_run: bool = False) -> None:
    print(f"Copy: {src} -> {dst}")
    if dry_run:
        print("  (dry-run) skip copy")
        return
    dst.mkdir(parents=True, exist_ok=True)
    rsync = shutil.which("rsync")
    if rsync:
        cmd = [rsync, "-a", "--info=progress2", f"{src}/", f"{dst}/"]
        print(f"  run: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        for path in src.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        print("  copied with shutil (rsync not found)")


def run_cmd(cmd: list[str], env: dict | None = None, dry_run: bool = False) -> None:
    print(f"Run: {' '.join(cmd)}")
    if dry_run:
        print("  (dry-run) skip")
        return
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.run(cmd, check=True, env=merged)


def oss_cli_args(args) -> list[str]:
    out = [
        "--bucket", args.bucket,
        "--endpoint", args.endpoint,
        "--prefix", args.prefix,
        "--workers", str(args.workers),
        "--retries", str(args.retries),
        "--retry-backoff", str(args.retry_backoff),
    ]
    if args.skip_existing:
        out.append("--skip-existing")
    else:
        out.append("--no-skip-existing")
    return out


def oss_has_data(args) -> bool:
    ak = os.environ.get(cfg.OSS_AK_ENV)
    sk = os.environ.get(cfg.OSS_SK_ENV)
    if not ak or not sk:
        print(f"OSS credentials missing ({cfg.OSS_AK_ENV}/{cfg.OSS_SK_ENV}); treating OSS as empty")
        return False
    try:
        bucket = make_bucket(ak, sk, args.endpoint, args.bucket)
        exists = has_remote_objects(bucket, args.prefix)
    except SystemExit:
        raise
    except Exception as e:
        print(f"OSS check failed: {e}; treating OSS as empty")
        return False
    remote = f"oss://{args.bucket}/{args.prefix.lstrip('/')}"
    print(f"  OSS {remote}: {'yes' if exists else 'no'}")
    return exists


def download_from_oss(fs_dir: Path, args) -> None:
    cmd = [
        sys.executable, "-m", "autoDL.upload_to_oss", "download",
        "--local", str(fs_dir),
        *oss_cli_args(args),
    ]
    run_cmd(cmd, dry_run=args.dry_run)


def upload_to_oss(fs_dir: Path, args) -> None:
    cmd = [
        sys.executable, "-m", "autoDL.upload_to_oss", "upload",
        "--local", str(fs_dir),
        *oss_cli_args(args),
    ]
    run_cmd(cmd, dry_run=args.dry_run)


def download_climbmix(fs_dir: Path, num_shards: int, dry_run: bool) -> bool:
    """Return True if an action was taken / needed."""
    n = count_parquet(fs_dir / cfg.DATA_SUBDIR)
    if n >= num_shards:
        print(f"Climbmix already present ({n} shards)")
        return False
    print(f"Downloading climbmix shards (-n {num_shards})")
    fs_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [sys.executable, "-m", "nanochat.dataset", "-n", str(num_shards)],
        env={"NANOCHAT_BASE_DIR": str(fs_dir)},
        dry_run=dry_run,
    )
    return True


def ensure_tokenizer(fs_dir: Path, min_shards: int, dry_run: bool) -> bool:
    if has_tokenizer(fs_dir):
        print("Tokenizer already present on autodl-fs")
        return False
    print("Tokenizer missing; running scripts.tok_train")
    data_dir = fs_dir / cfg.DATA_SUBDIR
    n = count_parquet(data_dir)
    if n < 1 and not dry_run:
        raise RuntimeError(f"Cannot train tokenizer: no parquet under {data_dir}")
    if n < min_shards:
        print(f"Warning: only {n} shards present (wanted {min_shards}); tok_train will use what's available")
    run_cmd(
        [sys.executable, "-m", "scripts.tok_train"],
        env={"NANOCHAT_BASE_DIR": str(fs_dir)},
        dry_run=dry_run,
    )
    return True


def ensure_task_datasets(fs_dir: Path, dry_run: bool) -> bool:
    missing = missing_task_datasets(fs_dir)
    if not missing:
        print(f"All {len(cfg.TASK_DATASETS)} task datasets present")
        return False
    print(f"Prefetching {len(missing)} task dataset(s) into {fs_dir / cfg.TASK_SUBDIR}")
    if dry_run:
        for spec in missing:
            print(f"  (dry-run) {spec['repo_id']} subset={spec['subset']} split={spec['split']}")
        return True
    fs_dir.mkdir(parents=True, exist_ok=True)
    # load_hub_dataset writes under get_base_dir()/task_data/...
    prev = os.environ.get("NANOCHAT_BASE_DIR")
    os.environ["NANOCHAT_BASE_DIR"] = str(fs_dir)
    try:
        from tasks.common import load_hub_dataset
        for spec in missing:
            print(f"  download {spec['repo_id']} subset={spec['subset']} split={spec['split']}")
            load_hub_dataset(spec["repo_id"], spec["subset"], split=spec["split"])
    finally:
        if prev is None:
            os.environ.pop("NANOCHAT_BASE_DIR", None)
        else:
            os.environ["NANOCHAT_BASE_DIR"] = prev
    return True


def ensure_eval_bundle(fs_dir: Path, dry_run: bool) -> bool:
    if has_eval_bundle(fs_dir):
        print("eval_bundle already present")
        return False
    print(f"Downloading eval_bundle from {cfg.EVAL_BUNDLE_URL}")
    if dry_run:
        print("  (dry-run) skip")
        return True
    fs_dir.mkdir(parents=True, exist_ok=True)
    prev = os.environ.get("NANOCHAT_BASE_DIR")
    os.environ["NANOCHAT_BASE_DIR"] = str(fs_dir)
    try:
        from nanochat.common import download_file_with_lock

        def place_eval_bundle(file_path):
            bundle_dir = fs_dir / cfg.EVAL_BUNDLE_DIRNAME
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(tmpdir)
                extracted = Path(tmpdir) / cfg.EVAL_BUNDLE_DIRNAME
                if bundle_dir.exists():
                    shutil.rmtree(bundle_dir)
                shutil.move(str(extracted), str(bundle_dir))
            print(f"Placed eval_bundle at {bundle_dir}")

        download_file_with_lock(cfg.EVAL_BUNDLE_URL, cfg.EVAL_BUNDLE_ZIP, postprocess_fn=place_eval_bundle)
    finally:
        if prev is None:
            os.environ.pop("NANOCHAT_BASE_DIR", None)
        else:
            os.environ["NANOCHAT_BASE_DIR"] = prev
    return True


def fill_gaps_from_source(fs_dir: Path, args) -> bool:
    """Download anything still missing onto autodl-fs. Returns True if anything was fetched/built."""
    filled = False
    filled |= download_climbmix(fs_dir, args.num_shards, args.dry_run)
    filled |= ensure_tokenizer(fs_dir, args.min_shards, args.dry_run)
    filled |= ensure_task_datasets(fs_dir, args.dry_run)
    filled |= ensure_eval_bundle(fs_dir, args.dry_run)
    return filled


def prepare(args) -> int:
    local_dir = resolve_dir(args.local)
    fs_dir = resolve_dir(args.fs_root)
    min_shards = args.min_shards

    print(f"Local   : {local_dir}")
    print(f"AutoDL  : {fs_dir}")
    print(f"OSS     : oss://{args.bucket}/{args.prefix.lstrip('/')}")
    print(f"Shards  : min={min_shards}, hf_download={args.num_shards}")
    print(f"Tasks   : {len(cfg.TASK_DATASETS)} hub datasets + eval_bundle")
    print()

    # 1) local
    print("[1/4] Check local")
    if has_complete_data(local_dir, min_shards):
        print("Local data complete; nothing to do.")
        return 0

    # 2) autodl-fs
    print("[2/4] Check autodl-fs")
    if has_complete_data(fs_dir, min_shards):
        copy_tree(fs_dir, local_dir, dry_run=args.dry_run)
        print("Done (copied from autodl-fs).")
        return 0

    # 3) OSS -> autodl-fs (best effort)
    print("[3/4] Check OSS")
    pulled_oss = False
    if oss_has_data(args):
        download_from_oss(fs_dir, args)
        pulled_oss = True
        if has_complete_data(fs_dir, min_shards):
            copy_tree(fs_dir, local_dir, dry_run=args.dry_run)
            print("Done (OSS -> autodl-fs -> local).")
            return 0
        print("OSS pull incomplete; will fill remaining gaps from HuggingFace / S3.")

    # 4) fill gaps, upload if we added anything, copy to local
    print("[4/4] Fill gaps (climbmix / tokenizer / SFT-RL tasks / eval_bundle)")
    filled = fill_gaps_from_source(fs_dir, args)
    if filled or not pulled_oss:
        # upload so next machine hits OSS (also when OSS was empty)
        upload_to_oss(fs_dir, args)
    elif filled is False and pulled_oss:
        # OSS had partial data and nothing new filled — still try copy
        pass

    if not args.dry_run and not has_complete_data(fs_dir, min_shards):
        print("Warning: autodl-fs still incomplete after fill", file=sys.stderr)
        return 1

    copy_tree(fs_dir, local_dir, dry_run=args.dry_run)
    print("Done (filled -> autodl-fs -> OSS -> local).")
    return 0


def build_parser():
    p = argparse.ArgumentParser(description="Prepare nanochat data on AutoDL (local / fs / OSS / HF)")
    p.add_argument("--local", type=str, default=cfg.LOCAL_DIR, help="local training data dir")
    p.add_argument("--fs-root", type=str, default=cfg.AUTODL_FS_DIR, help="AutoDL persistent fs root")
    p.add_argument("--min-shards", type=int, default=cfg.MIN_SHARDS, help="min parquet files for completeness")
    p.add_argument("--num-shards", type=int, default=cfg.NUM_SHARDS, help="HF climbmix download shard count")
    p.add_argument("--bucket", type=str, default=cfg.OSS_BUCKET)
    p.add_argument("--endpoint", type=str, default=cfg.OSS_ENDPOINT)
    p.add_argument("--prefix", type=str, default=cfg.OSS_PREFIX)
    p.add_argument("--workers", type=int, default=cfg.WORKERS)
    p.add_argument("--retries", type=int, default=cfg.RETRIES, help=f"OSS attempts per file (default: {cfg.RETRIES})")
    p.add_argument("--retry-backoff", type=float, default=cfg.RETRY_BACKOFF_SEC)
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=cfg.SKIP_EXISTING,
        help="skip same-size OSS objects (default: on)",
    )
    p.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    p.add_argument("--dry-run", action="store_true", help="print actions only")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return prepare(args)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode or 1


if __name__ == "__main__":
    sys.exit(main())
