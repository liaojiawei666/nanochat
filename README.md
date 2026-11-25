# Nanochat：单卡训练学习版

主线是：Rust 编码数据 → 准备本地目录 → 训练 → 验证 loss → 保存最终权重 → 生成文本。

流程概览见[预训练与 SFT](doc/data_prepare.md)。

保留全因果注意力、RoPE、value embedding、smear、Muon/AdamW、梯度累积和混合精度。去掉了训练 checkpoint/恢复、哈希指纹、完整 CORE 测评、自动缓存修复、运行状态发布和自动关机。中断后需要重新训练。

## 安装

```bash
uv sync --locked --group dev
cargo build --release --locked --manifest-path rust-bpe/Cargo.toml
```

Python 3.12+。训练和生成要求 CUDA GPU，没有可用 GPU 会直接报错。非敏感配置统一放在根目录 `config.json`：`model` 是模型参数，`train` 是训练参数，`storage` 是默认数据路径，`platforms` 是各平台目录。`env.json` 仅保存 SSH 登录信息，并已忽略提交。命令默认读取项目根目录的 `config.json`，也可通过 `--config` 指定其他 JSON 文件。

## 本地编码与打包

将 parquet 放入 `data/pretrain/climbmix/parquet`，在仓库根目录执行：

```bash
# 已有词表时跳过 train，将词表放到 data/tokenizer/vocab.tiktoken。
cargo run --release --locked --manifest-path rust-bpe/Cargo.toml -- train
cargo run --release --locked --manifest-path rust-bpe/Cargo.toml -- encode

uv run python -m script.data pack
```

Rust 编码器会把词表复制到数据目录，输出仅包含词表和 token 分片：

```text
data/pretrain/climbmix/tokenized/
  vocab.tiktoken
  shard_00000.bin
  shard_00001.bin
  ...
```

BPE 编码器只返回正文 token，同一篇文档中的重复预分词片段会复用编码结果；写入分片时由命令层在每篇文档前添加 BOS。Python 固定按小端 `uint16` 读取分片，BOS 等特殊 token ID 与 Rust 的固定定义一致。按文件名排序后，最后一个分片用于验证，其余用于训练；每个 batch 的标签是输入右移一个 token。

编码目标目录必须不存在或为空，避免旧分片混入。编码先写入相邻的 `.part` 目录，全部完成后才发布正式目录；编码中断后，重试时会自动清理残留的临时目录。

Rust 默认使用上述数据目录，词表和验证集压缩率统计分别写入 `data/tokenizer/vocab.tiktoken`、`data/tokenizer/val_compress.txt`。也可以用 `NANOCHAT_PARQUET_DIR`、`NANOCHAT_TOKEN_DIR`、`NANOCHAT_VOCAB_PATH` 覆盖路径；自定义编码目录时，需同步修改 Python 配置的 `storage.data_dir`。

`script.data pack` 只打包 `vocab.tiktoken` 和 `shard_*.bin`，默认输出 `data/pretrain/climbmix/data.tar.zst`，随后通过 `script.autodl data` 上传到服务器共享存储。

## 服务器操作

采用镜像保存 Python 与依赖环境，本地脚本只需使用 `code`（上传代码）和 `data`（上传编码数据压缩包）。环境安装、解压、启动训练和查看日志通过 SSH 手动执行。

完整命令统一放在[预训练与 SFT 操作流程](doc/data_prepare.md)，按“本地执行”和“服务器执行”区分。代码默认上传到系统盘的 `/root/nanochat`，环境安装到项目的 `.venv`，一起保存到镜像；数据压缩包仍放共享盘。AutoDL 与 DAMODEL 的入口均只支持 `code` 和 `data`，可加 `--dry-run` 预览。

## 训练与生成

以下使用本地 `data/` 目录；服务器数据与训练结果按[操作流程](doc/data_prepare.md)映射到平台的 `data_root`。

```bash
uv run python -m nanochat inspect
uv run python -m nanochat train
uv run python -m nanochat generate \
  --prompt "The capital of France is" --max-tokens 32
```

模型深度决定默认宽度和头数。词表大小在训练时从实际词表读取；`inspect` 使用 JSON 中的词表大小，默认是 32768。训练预算、batch 大小和学习率缩放保留原有推导公式，也可以在 `train` 中显式设置 `max_steps` 和 `total_batch_tokens`。`nanochat train` 和 `script.data pack/extract` 支持 `--data-dir`，覆盖 `storage.data_dir`；数据命令还支持 `--archive-path` 覆盖压缩包位置。这些参数不会修改配置文件。指定目录应直接包含 `vocab.tiktoken` 和 `shard_*.bin`。

每次更新累积 `total_batch_tokens / (micro_batch × sequence_len)` 个 micro-batch 的梯度。参数及优化器状态为 FP32，`dtype = "auto"` 根据 GPU 能力选择 BF16 混合精度或 FP32，也可以显式设置 `fp32`。设备检测统一放在 `resolve_runtime()`，不提供 CPU 训练分支。数据在 CPU 拼接成 batch，再传入 GPU；模型参数在训练期间持续留在 GPU。

日志打印训练 loss、验证 loss、学习率和 tokens/s。训练前、每隔 `eval_every` 步以及训练结束时评估独立验证集。无需下载额外评测包。

训练结束只在 `output_dir` 保存：

```text
run-d24/
  model.pt          # 最终权重，使用 CPU Tensor 保存
  config.json       # 实际模型配置、训练设置和特殊 token ID
  vocab.tiktoken    # 生成文本所需词表
```

生成只需要这三个文件，可以将它们复制到另一台机器，不再依赖原始数据目录。`--temperature 0` 默认贪心生成，正数启用采样。已存在 `model.pt` 的输出目录不会覆盖，下一次训练请修改 `output_dir`。需要记录日志可使用终端重定向或 `tee`；实例由你手动关闭。

调试时可在 `config.json` 中减小 `model.n_layer`、`model.sequence_len`，并将 `train.max_steps` 设为 3。

## 从哪里读代码

| 文件 | 内容 |
| --- | --- |
| `nanochat/train.py` | 初始化 → 训练循环 → 验证 → 保存最终权重 |
| `nanochat/gpt.py` | 模型结构、前向计算和生成 |
| `nanochat/optim.py` | Muon/AdamW 更新公式 |
| `nanochat/dataloader.py` | 读取分片、文档打包、CPU → GPU |
| `nanochat/config.py` | 参数、训练预算和学习率调度 |
| `nanochat/eval.py` | 验证 loss、加载最终模型 |
| `script/data.py` | 独立的数据打包与解压 |

`study/` 保留独立的 AdamW 学习代码。优化器公式参考固定的 [nanochat 92d63d4](https://github.com/karpathy/nanochat/tree/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd)，许可证位于 `third_party/nanochat-LICENSE`。

## 检查

```bash
uv run pytest -q
uv run pyrefly check
```

测试覆盖模型因果性、优化器数值对照、文档打包、梯度累积、训练与保存后生成、基础数据传输。默认离线运行。数值单元测试可使用 CPU；完整训练与生成测试需要 CUDA，BF16 测试还要求相应精度支持。
