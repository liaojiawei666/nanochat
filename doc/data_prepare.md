# 预训练与 SFT

## 本地 data 目录

所有本地数据统一放在仓库的 `data/` 下：

~~~text
data/
├── tokenizer/
│   ├── vocab.tiktoken        # tokenizer 词表
│   └── val_compress.txt      # 验证集压缩率统计
├── pretrain/
│   └── climbmix/
│       ├── parquet/         # 下载的原始文本数据
│       ├── tokenized/       # encode 后的 .bin 分片、配套词表
│       └── data.tar.zst     # 上传到服务器的压缩包
├── sft/
│   ├── raw/                 # 原始对话数据
│   └── processed/           # 清洗、格式化后的训练数据
├── eval/                    # 评测数据
└── runs/                    # 本地模型权重、日志和训练结果
~~~

## 服务器目录与镜像

环境在系统盘安装好后，通过平台控制台保存镜像。日常由本地脚本上传代码和数据压缩包，安装依赖、解压和训练都通过 SSH 执行。以下完整命令以 AutoDL 为例。

| 名称 | 路径 | 说明 |
| --- | --- | --- |
| 代码与环境（两平台） | `/root/nanochat` | 系统盘；`.venv` 在项目中，随系统盘保存到镜像 |
| AutoDL 压缩包 | `/root/autodl-fs/nanochat/data/pretrain/climbmix/data.tar.zst` | 共享盘；换实例后挂载复用 |
| AutoDL 数据根目录 | `/root/autodl-tmp/nanochat` | 对应本地 `data/`，其下目录结构一致 |
| AutoDL 编码数据 | `/root/autodl-tmp/nanochat/pretrain/climbmix/tokenized` | 数据盘；由压缩包解压得到 |
| AutoDL 训练结果 | `/root/autodl-tmp/nanochat/runs/<实验名>` | 数据盘；保存权重、配置和日志 |
| DAMODEL 压缩包 | `/root/shared-storage/nanochat/data/pretrain/climbmix/data.tar.zst` | 共享存储；上传目标取平台 `archive_path` |
| DAMODEL 数据根目录 | `/root/workspace/nanochat` | 对应本地 `data/`，其下目录结构一致 |
| DAMODEL 编码数据 | `/root/workspace/nanochat/pretrain/climbmix/tokenized` | 编码数据目录 |

`config.json.platforms` 保存平台目录：`project_dir` 是代码目录，`shared_dir` 是共享盘挂载点，`archive_path` 是共享盘压缩包，`data_root` 是对应本地 `data/` 的服务器数据根目录。两平台代码目标目录默认均为 `/root/nanochat`。SSH 登录信息放在已忽略提交的 `env.json` 中。

`storage.data_dir` 和 `train.output_dir` 保留本地以 `data/` 开头的路径；服务器命令用所选平台的 `data_root` 替换这个前缀。例如本地 `data/pretrain/climbmix/tokenized` 对应 AutoDL 的 `/root/autodl-tmp/nanochat/pretrain/climbmix/tokenized`，本地 `data/runs/run-d24` 对应 `/root/autodl-tmp/nanochat/runs/run-d24`。配置不会自动根据所在机器切换，以下命令显式选择 AutoDL。

~~~text
/root/autodl-tmp/nanochat/    # 对应本地 data/
├── tokenizer/
├── pretrain/
│   └── climbmix/
│       └── tokenized/       # 解压后的 .bin 分片和词表
├── sft/
├── eval/
└── runs/                   # 权重、配置和日志
~~~

两边采用相同目录结构，按需放置数据；预训练只需解压 `tokenized/`，不必上传原始 Parquet。普通 AutoDL 实例的 `/root/autodl-tmp` 是独立数据盘；Pro 实例的同名目录占用系统盘空间，见[平台说明](https://www.autodl.com/docs/instance_pro_data/)。

AutoDL 镜像保存系统盘，共享盘和数据盘需要分别保留；镜像的范围见[平台说明](https://www.autodl.com/docs/image/)。新建实例时选用已保存的镜像，再挂载共享盘。不要在数据盘中留存唯一一份重要权重，结束训练后及时下载或备份到共享盘。

## 预训练

### 1. 本地准备数据

在本地项目根目录执行。先将原始 Parquet 下载到 `data/pretrain/climbmix/parquet/`，再训练 tokenizer、encode 和打包：

~~~bash
cargo run --release --locked --manifest-path rust-bpe/Cargo.toml -- train
cargo run --release --locked --manifest-path rust-bpe/Cargo.toml -- encode
uv run python -m script.data pack
~~~

已有词表时跳过 `train`，将词表放到 `data/tokenizer/vocab.tiktoken`。Rust 也支持通过 `NANOCHAT_PARQUET_DIR`、`NANOCHAT_TOKEN_DIR` 和 `NANOCHAT_VOCAB_PATH` 覆盖默认路径；自定义编码目录时，打包命令也要传入对应的 `--data-dir`。

压缩包只包含编码后的 `shard_*.bin`（训练集和验证集）及 `vocab.tiktoken`。服务器训练只需要这个压缩包，无需上传原始 Parquet，也无需重复上传未压缩的 `tokenized/`。

### 2. 本地上传代码和压缩包

在本地项目根目录执行；上传数据前先在 AutoDL 控制台挂载共享存储：

~~~bash
uv run --group remote python -m script.autodl code
uv run --group remote python -m script.autodl data
~~~

脚本只支持 `code` 和 `data`，均可加 `--dry-run` 预览。`code` 上传源码、项目配置和依赖文件，保留远端已有的 `.venv`。`data` 上传 `storage.archive_path` 指定的压缩包，远端目标取 `platforms.AutoDL.archive_path`；上传完成后校验 SHA256，相同内容直接跳过。首次连接先通过 `ssh -p PORT USER@HOST` 确认服务器身份，连接信息取自 `env.json`。

只有代码变化时执行 `code`；只有数据变化时重新打包并执行 `data`。上传其他压缩包可用：

~~~bash
uv run --group remote python -m script.autodl data --archive /path/to/data.tar.zst
~~~

DAMODEL 上传时将模块名换成 `script.damodel`，代码目标目录使用 `config.json` 中该平台的 `project_dir`，SSH 登录信息使用 `env.json` 中的 `DAMODEL.ssh`。

### 3. SSH 首次安装环境并保存镜像

以下命令在服务器执行。先确认代码已上传到 `/root/nanochat`：

~~~bash
cd /root/nanochat
unset UV_PYTHON_INSTALL_DIR UV_PROJECT_ENVIRONMENT UV_CACHE_DIR
python3 -m pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple
uv python install 3.12
uv sync --locked --no-dev --managed-python --python 3.12
.venv/bin/python -c "import sys, torch; print('Python:', sys.version); print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.is_available())"
~~~

上述命令清除旧的共享环境路径设置，Python 使用 uv 的系统盘默认目录，依赖安装到 `/root/nanochat/.venv`。

无卡模式可以安装依赖，正式训练前确认 GPU 可用。安装好后可执行 `uv cache clean` 清理下载缓存，再按平台要求关机并在控制台保存镜像；保存环境镜像时无需解压训练数据。

以后只改代码无需重装依赖；依赖文件变化时，先上传代码，再在服务器执行：

~~~bash
cd /root/nanochat
uv sync --locked --no-dev --managed-python --python 3.12
~~~

`pyproject.toml` 与 `uv.lock` 一起在本地更新、一起上传。`--locked` 会在两者不一致时报错；需要复用新版依赖环境时，再保存新的镜像。

### 4. SSH 解压到数据盘

新实例尚无编码数据时，在服务器执行：

~~~bash
cd /root/nanochat
NANOCHAT_ARCHIVE_PATH=$(.venv/bin/python -c 'from nanochat.config import Config; print(Config.load().platforms["AutoDL"]["archive_path"])')
NANOCHAT_DATA_DIR=$(.venv/bin/python -c 'from pathlib import Path; from nanochat.config import Config; c=Config.load(); print(Path(c.platforms["AutoDL"]["data_root"]) / Path(c.storage.data_dir).relative_to("data"))')
.venv/bin/python -m script.data extract \
  --archive-path "$NANOCHAT_ARCHIVE_PATH" \
  --data-dir "$NANOCHAT_DATA_DIR"
~~~

解压会先删除目标编码数据目录及其原有内容，再写入压缩包中的文件。已有相同数据时可跳过；更新数据时直接对同一目录重新解压即可。开始前会检查压缩包可读且位于目标目录之外；开始解压后如果失败，旧数据也已清除，修复问题后重新执行命令。

### 5. SSH 启动训练、查看日志

每次训练选一个实验目录。以下命令将本次配置另存到数据盘，并将权重、日志也写到同一实验目录：

~~~bash
cd /root/nanochat
NANOCHAT_RUN_DIR=$(.venv/bin/python -c 'from pathlib import Path; from nanochat.config import Config; c=Config.load(); print(Path(c.platforms["AutoDL"]["data_root"]) / Path(c.train.output_dir).relative_to("data"))')
mkdir -p "$NANOCHAT_RUN_DIR"

.venv/bin/python - "$NANOCHAT_RUN_DIR" <<'PY'
import json
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
settings = json.loads(Path("config.json").read_text())
data_root = Path(settings["platforms"]["AutoDL"]["data_root"])
settings["storage"]["data_dir"] = str(data_root / Path(settings["storage"]["data_dir"]).relative_to("data"))
settings["train"]["output_dir"] = str(run_dir)
(run_dir / "launch.json").write_text(json.dumps(settings, indent=2) + "\n")
PY

nohup .venv/bin/python -u -m nanochat train \
  --config "$NANOCHAT_RUN_DIR/launch.json" \
  > "$NANOCHAT_RUN_DIR/train.log" 2>&1 < /dev/null &
~~~

查看训练日志：

~~~bash
tail -n 100 -f /root/autodl-tmp/nanochat/runs/run-d24/train.log
~~~

退出日志查看按 `Ctrl+C`，不影响后台训练。训练成功后实验目录中会生成 `model.pt`、`config.json` 和 `vocab.tiktoken`；下次训练使用新的实验名，避免覆盖已有日志和配置。训练没有中断续训功能。

## SFT

本地原始数据放在 `data/sft/raw/`，处理后的数据放在 `data/sft/processed/`。当前上述打包、解压和训练命令针对预训练的 `.bin` 数据；SFT 的处理与训练命令待实现后补充。
