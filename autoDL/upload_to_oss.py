"""
Upload / download a local directory <-> Alibaba Cloud OSS.

Credentials (do not hardcode):
  export OSS_LLM_AK=...
  export OSS_LLM_SK=...

Defaults come from autoDL.config (bucket=llm-data-1 Beijing, prefix=nanochat/).

Examples:
  python -m autoDL.upload_to_oss upload --local ~/.cache/nanochat
  python -m autoDL.upload_to_oss download --local /root/autodl-fs/nanochat
  python -m autoDL.upload_to_oss has-objects
  python -m autoDL.upload_to_oss upload --local ~/.cache/nanochat --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from autoDL import config as cfg


def _require_oss2():
    try:
        import oss2  # noqa: F401
    except ImportError:
        print("Missing dependency: oss2", file=sys.stderr)
        print("Install with:  uv pip install oss2", file=sys.stderr)
        sys.exit(1)
    import oss2
    return oss2


def normalize_prefix(prefix: str) -> str:
    prefix = prefix.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return prefix


def rel_key(local_root: Path, path: Path, prefix: str) -> str:
    return f"{normalize_prefix(prefix)}{path.relative_to(local_root).as_posix()}"


def key_to_local(local_root: Path, key: str, prefix: str) -> Path:
    prefix = normalize_prefix(prefix)
    if not key.startswith(prefix):
        raise ValueError(f"key {key!r} does not start with prefix {prefix!r}")
    rel = key[len(prefix):]
    if not rel or rel.endswith("/"):
        raise ValueError(f"not a file key: {key!r}")
    return local_root / rel


def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1024
    return f"{n}B"


def iter_local_files(local_root: Path):
    for path in sorted(local_root.rglob("*")):
        if path.is_file():
            yield path


def make_bucket(access_key_id: str, access_key_secret: str, endpoint: str, bucket_name: str):
    oss2 = _require_oss2()
    auth = oss2.Auth(access_key_id, access_key_secret)
    return oss2.Bucket(auth, endpoint, bucket_name)


def list_remote_objects(bucket, prefix: str):
    oss2 = _require_oss2()
    prefix = normalize_prefix(prefix)
    for obj in oss2.ObjectIterator(bucket, prefix=prefix):
        if obj.key.endswith("/"):
            continue
        yield obj.key, obj.size


def has_remote_objects(bucket, prefix: str) -> bool:
    for _ in list_remote_objects(bucket, prefix):
        return True
    return False


def upload_one(bucket, local: Path, key: str, multipart_threshold: int, part_size: int, skip_existing: bool):
    if skip_existing and bucket.object_exists(key):
        remote_size = bucket.head_object(key).content_length
        local_size = local.stat().st_size
        if remote_size == local_size:
            return "skip", local, key, local_size, 0.0

    size = local.stat().st_size
    t0 = time.time()
    if size >= multipart_threshold:
        oss2 = _require_oss2()
        oss2.resumable_upload(
            bucket,
            key,
            str(local),
            multipart_threshold=multipart_threshold,
            part_size=part_size,
            num_threads=4,
        )
    else:
        bucket.put_object_from_file(key, str(local))
    dt = time.time() - t0
    return "ok", local, key, size, dt


def download_one(bucket, local: Path, key: str, size: int, multipart_threshold: int, part_size: int, skip_existing: bool):
    if skip_existing and local.is_file() and local.stat().st_size == size:
        return "skip", local, key, size, 0.0

    local.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if size >= multipart_threshold:
        oss2 = _require_oss2()
        oss2.resumable_download(
            bucket,
            key,
            str(local),
            multiget_threshold=multipart_threshold,
            part_size=part_size,
            num_threads=4,
        )
    else:
        bucket.get_object_to_file(key, str(local))
    dt = time.time() - t0
    return "ok", local, key, size, dt


def run_jobs(jobs, workers: int, total_bytes: int, ok_label: str):
    done = skipped = failed = 0
    done_bytes = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, *args): (fn, args) for fn, args in jobs}
        for fut in as_completed(futures):
            try:
                status, local, key, size, dt = fut.result()
            except Exception as e:
                failed += 1
                print(f"[FAIL] {futures[fut][1]}: {e}", file=sys.stderr)
                continue

            done_bytes += size
            if status == "skip":
                skipped += 1
                print(f"[SKIP] {key}")
            else:
                done += 1
                speed = size / dt / (1024 * 1024) if dt > 0 else 0
                print(f"[OK]   {key}  {human_size(size)}  {speed:.1f} MB/s")

            elapsed = time.time() - t_start
            pct = 100.0 * done_bytes / total_bytes if total_bytes else 100.0
            print(f"       progress {pct:.1f}%  {human_size(done_bytes)}/{human_size(total_bytes)}  elapsed {elapsed:.0f}s")

    print("-" * 60)
    print(f"Done. {ok_label}={done} skipped={skipped} failed={failed} in {time.time() - t_start:.1f}s")
    if failed:
        sys.exit(1)


def cmd_upload(args, bucket):
    local_root = Path(args.local).expanduser().resolve()
    if not local_root.is_dir():
        print(f"Local directory not found: {local_root}", file=sys.stderr)
        sys.exit(1)

    files = list(iter_local_files(local_root))
    if not files:
        print(f"No files under {local_root}")
        return

    total_bytes = sum(p.stat().st_size for p in files)
    remote = f"oss://{args.bucket}/{normalize_prefix(args.prefix)}"
    print(f"Mode   : upload")
    print(f"Local  : {local_root}")
    print(f"Remote : {remote}")
    print(f"Files  : {len(files)} ({human_size(total_bytes)})")
    print(f"Workers: {args.workers}")

    if args.dry_run:
        for path in files:
            key = rel_key(local_root, path, args.prefix)
            print(f"  {path.relative_to(local_root)}  ->  {key}  ({human_size(path.stat().st_size)})")
        print("Dry-run only; nothing uploaded.")
        return

    jobs = [
        (
            upload_one,
            (
                bucket,
                path,
                rel_key(local_root, path, args.prefix),
                args.multipart_threshold,
                args.part_size,
                args.skip_existing,
            ),
        )
        for path in files
    ]
    run_jobs(jobs, args.workers, total_bytes, ok_label="uploaded")


def cmd_download(args, bucket):
    local_root = Path(args.local).expanduser().resolve()
    local_root.mkdir(parents=True, exist_ok=True)

    objects = list(list_remote_objects(bucket, args.prefix))
    if not objects:
        print(f"No objects under oss://{args.bucket}/{normalize_prefix(args.prefix)}")
        return

    total_bytes = sum(size for _, size in objects)
    remote = f"oss://{args.bucket}/{normalize_prefix(args.prefix)}"
    print(f"Mode   : download")
    print(f"Remote : {remote}")
    print(f"Local  : {local_root}")
    print(f"Files  : {len(objects)} ({human_size(total_bytes)})")
    print(f"Workers: {args.workers}")

    if args.dry_run:
        for key, size in objects:
            local = key_to_local(local_root, key, args.prefix)
            print(f"  {key}  ->  {local.relative_to(local_root)}  ({human_size(size)})")
        print("Dry-run only; nothing downloaded.")
        return

    jobs = [
        (
            download_one,
            (
                bucket,
                key_to_local(local_root, key, args.prefix),
                key,
                size,
                args.multipart_threshold,
                args.part_size,
                args.skip_existing,
            ),
        )
        for key, size in objects
    ]
    run_jobs(jobs, args.workers, total_bytes, ok_label="downloaded")


def cmd_has_objects(args, bucket):
    remote = f"oss://{args.bucket}/{normalize_prefix(args.prefix)}"
    exists = has_remote_objects(bucket, args.prefix)
    print(f"{remote}: {'yes' if exists else 'no'}")
    sys.exit(0 if exists else 1)


def build_parser():
    parser = argparse.ArgumentParser(description="Upload/download a local directory <-> Aliyun OSS")
    parser.add_argument(
        "mode",
        choices=["upload", "download", "has-objects"],
        help="upload local->OSS, download OSS->local, or check if prefix has objects",
    )
    parser.add_argument(
        "--local",
        type=str,
        default=cfg.LOCAL_DIR,
        help=f"local directory (default: {cfg.LOCAL_DIR})",
    )
    parser.add_argument("--bucket", type=str, default=cfg.OSS_BUCKET)
    parser.add_argument("--endpoint", type=str, default=cfg.OSS_ENDPOINT)
    parser.add_argument("--prefix", type=str, default=cfg.OSS_PREFIX)
    parser.add_argument(
        "--access-key-id",
        type=str,
        default=os.environ.get(cfg.OSS_AK_ENV),
        help=f"from env {cfg.OSS_AK_ENV}",
    )
    parser.add_argument(
        "--access-key-secret",
        type=str,
        default=os.environ.get(cfg.OSS_SK_ENV),
        help=f"from env {cfg.OSS_SK_ENV}",
    )
    parser.add_argument("--workers", type=int, default=cfg.WORKERS)
    parser.add_argument("--multipart-threshold", type=int, default=cfg.MULTIPART_THRESHOLD)
    parser.add_argument("--part-size", type=int, default=cfg.PART_SIZE)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=cfg.SKIP_EXISTING,
        help="skip same-size existing files/objects (default: on)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="always overwrite",
    )
    parser.add_argument("--dry-run", action="store_true", help="list planned transfers only")
    return parser


def main(argv=None):
    oss2 = _require_oss2()
    args = build_parser().parse_args(argv)

    missing = [
        n for n, v in [(cfg.OSS_AK_ENV, args.access_key_id), (cfg.OSS_SK_ENV, args.access_key_secret)] if not v
    ]
    if missing and not (args.dry_run and args.mode == "upload"):
        print("Missing required env:", ", ".join(missing), file=sys.stderr)
        sys.exit(1)

    bucket = None
    if not (args.dry_run and args.mode == "upload"):
        bucket = make_bucket(args.access_key_id, args.access_key_secret, args.endpoint, args.bucket)

    if args.mode == "upload":
        cmd_upload(args, bucket)
    elif args.mode == "download":
        cmd_download(args, bucket)
    else:
        cmd_has_objects(args, bucket)


if __name__ == "__main__":
    main()
