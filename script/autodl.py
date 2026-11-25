"""AutoDL 平台入口。"""
from pathlib import Path

from script.platform_cli import main as run_platform


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> None:
    run_platform("AutoDL", ROOT, argv)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
