"""以公式生成配置、元素数、精度、显存及计算量报告，不实例化模型。"""

from dataclasses import dataclass
from math import log2
from pathlib import Path

ASPECT_RATIO = 64
VOCAB_SIZE = 32768
HEAD_DIM = 128
SEQUENCE_LEN = 2048
NS_STEPS = 5
TOKEN_PARAM_RATIO = 8  # 官方 speedrun 的 --target-param-data-ratio
BATCH_REFERENCE_DEPTH = 12
BATCH_REFERENCE_TOKENS = 2**19
BATCH_SCALING_EXPONENT = 0.383


@dataclass(frozen=True)
class ModelEstimate:
    depth: int
    width: int
    heads: int
    ve_layers: int
    blocks: int
    embedding: int
    value_embeddings: int
    lm_head: int
    params: int
    adamw: int
    muon: int
    fixed: int
    hidden_activations: int
    logits_activations: int
    activations: int
    rope: int
    fixed_bytes: int
    activation_bytes: int
    forward_flops: int
    backward_flops: int
    flash_recompute_flops: int
    adamw_flops: int
    muon_flops: int


@dataclass(frozen=True)
class TrainingEstimate:
    batch: int  # 每次更新的全局样本数
    batch_tokens: int
    target_tokens: int
    training_tokens: int  # 按完整 batch 向下取整
    updates: int
    fixed_flops: int  # 每次优化器更新
    sample_flops: int  # 每个样本的前后向
    total_flops: int  # 全程训练


@dataclass(frozen=True)
class GPUProfile:
    name: str
    memory_gib: int
    peak_tflops: float  # BF16 输入、FP32 累加，稠密
    efficiency_min: float  # 规划假设，不是实测 MFU
    efficiency_max: float


GPU_PROFILES = (
    GPUProfile("RTX 5090", 32, 209.5, 0.25, 0.40),
    GPUProfile("H100 SXM 80GB", 80, 989.0, 0.35, 0.55),
)


def render_hardware_estimate(stats: ModelEstimate) -> list[str]:
    """生成 L=24 的单卡容量与时间规划；效率区间显式作为假设。"""
    training = calculate_training_estimate(stats)
    profiles = GPU_PROFILES
    device_batches = [
        max(
            b for b in range(1, training.batch + 1)
            if training.batch % b == 0
            and stats.fixed_bytes + b * stats.activation_bytes <= p.memory_gib * 2**30
        )
        for p in profiles
    ]
    memory = [
        (stats.fixed_bytes + b * stats.activation_bytes) / 2**30
        for b in device_batches
    ]
    lines = [
        "| 项目 | " + " | ".join(p.name for p in profiles) + " |",
        "| :--- | ---: | ---: |",
    ]
    rows = [
        ("显存容量 $C$（名义容量）", [f"{p.memory_gib} GiB" for p in profiles]),
        ("Device batch $b$（尽量装满）", [str(b) for b in device_batches]),
        ("公式内显存占用率 $u$", [f"{m / p.memory_gib:.1%}" for m, p in zip(memory, profiles)]),
        ("BF16 稠密峰值 $Q_{peak}$", [f"{p.peak_tflops:g} TFLOP/s" for p in profiles]),
        ("有效计算效率 $\u03b7$（保守假设，未实测）", [f"{p.efficiency_min:.0%}–{p.efficiency_max:.0%}" for p in profiles]),
        ("满峰值理想时间 $t_{min}$", [
            f"{training.total_flops / (p.peak_tflops * 1e12 * 3600):.1f} 小时"
            for p in profiles
        ]),
        ("全程训练时间 $t$（估算）", [
            f"{training.total_flops / (p.peak_tflops * p.efficiency_max * 1e12 * 3600):.0f}–"
            f"{training.total_flops / (p.peak_tflops * p.efficiency_min * 1e12 * 3600):.0f} 小时"
            for p in profiles
        ]),
    ]
    lines.extend("| " + name + " | " + " | ".join(values) + " |" for name, values in rows)
    return [
        "## 表格7：L=24 单卡训练预估",
        f"沿用表6的训练预算（约 {training.training_tokens / 1e9:.2f}B tokens、全局 Batch={training.batch}），token/参数比为 {TOKEN_PARAM_RATIO}，与官方 speedrun 对齐。BF16 计算、完整因果注意力、无整层重计算。H100 按 SXM 80GB 版本估算。",
        "\n".join(lines),
        r"$$b=\max\{n\in\mathbb{N}_{+}:n\mid B,\ M_{fix}+nM_{act,1}\le C\},\quad u=\frac{M_{fix}+bM_{act,1}}{C}.$$",
        r"$$t_{min}=\frac{F_{total}}{Q_{peak}},\quad t=\frac{F_{total}}{\eta Q_{peak}}.$$",
        "按表4显存模型取能整除全局 Batch=512 的最大 device batch：5090 取 4；H100 的容量公式允许 17，但 17 不能整除 512，因此取 16。占用率只统计表4的权重、梯度、状态及激活；工作区、临时权重转换和分配器未计入，实际峰值仍需运行测量。",
        "表中的 25–40% / 35–55% 是保守规划假设，缺少实测依据，不代表两张卡实际只能达到这些效率。Tensor Core 峰值对应矩阵乘，训练还包含受显存带宽限制的归一化、激活等运算及数据供给、调度开销。更大的 batch 通常有利于矩阵乘吞吐，但显存占用率不能直接换算成计算效率，SDPA 也不能推出 5090 只能达到 25–40%。参见 [NVIDIA 矩阵乘性能说明](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)、[显存带宽限制层说明](https://docs.nvidia.com/deeplearning/performance/dl-performance-memory-limited/index.html)。",
        "训练时间区间仅表示表中效率假设下的结果，不等同实测预测；不含初始化编译、评测与检查点保存。实测时间可按预热后的吞吐 $r$（tokens/s）计算：$t=D/r$，其中 $D$ 为表6的实际训练 token 数。",
        "官方时间对照：[speedrun 脚本](https://github.com/karpathy/nanochat/blob/master/runs/speedrun.sh)使用 8×H100、L=24、FP8、token/参数比 8，沿用默认 SSSL 注意力；本表采用相同训练预算，计算精度为 BF16，注意力采用完整窗口。[README](https://github.com/karpathy/nanochat/blob/master/README.md)写约 1.5 小时，最新榜单记录为 1.65 小时。",
        fr"量级校验：[旧 BF16 基线](https://github.com/karpathy/nanochat/blob/master/dev/LEADERBOARD.md#run-1)在 token/参数比 12、8×H100 上实测训练迭代耗时 3.04 小时；按训练量与卡数理想线性缩放到本表预算，单卡约 $3.04\times8\times({TOKEN_PARAM_RATIO}/12)={3.04 * 8 * TOKEN_PARAM_RATIO / 12:.1f}$ 小时。跨版本数据仅用于量级核对。",
        "后端前提：H100 使用 FA3；5090 使用 PyTorch SDPA 的融合 Flash 后端，均设 window-pattern=L。5090 为 sm120，核实的 FA3 发布包声明 sm80 / sm90a；PyTorch 2.9.1 的 Flash 后端架构检查允许 sm120，head_dim=128 满足对应训练限制。官方在 FA3 不可用或加载异常时选择 SDPA，但锁定的 kernels 0.11.7 只检查构建文件是否存在，不检查 GPU 架构，不能保证 5090 自动回退。尚未做 GPU 运行验证；若实际选中 math 后端或使用显式滑窗 mask，当前显存和时间预算不适用。",
        "硬件依据：[NVIDIA RTX Blackwell 白皮书（RTX 5090 规格）](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)、[NVIDIA H100 规格](https://www.nvidia.com/en-us/data-center/h100/)。峰值采用 BF16 输入、FP32 累加的非稀疏口径，与 [nanochat 性能常量](https://github.com/karpathy/nanochat/blob/master/nanochat/common.py)一致。",
        "后端依据：[nanochat 注意力选择](https://github.com/karpathy/nanochat/blob/master/nanochat/flash_attention.py)、[GPU 架构](https://developer.nvidia.com/cuda/gpus)、[FA3 发布元数据](https://huggingface.co/kernels-community/flash-attn3/blob/main/build/torch29-cxx11-cu128-x86_64-linux/metadata.json)、[PyTorch 2.9.1 Flash 检查](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp)、[官方依赖锁](https://github.com/karpathy/nanochat/blob/master/uv.lock)、[kernels 0.11.7 检测逻辑](https://github.com/huggingface/kernels/blob/v0.11.7/src/kernels/utils.py#L321-L350)。",
    ]


def calculate_training_estimate(stats: ModelEstimate) -> TrainingEstimate:
    """按 speedrun 训练预算及官方全局 batch 缩放公式汇总整个训练过程。"""
    reference_width = ASPECT_RATIO * BATCH_REFERENCE_DEPTH
    reference_params = (
        12 * BATCH_REFERENCE_DEPTH * reference_width**2
        + VOCAB_SIZE * reference_width
    )
    scaling_params = stats.blocks + stats.lm_head
    predicted_tokens = BATCH_REFERENCE_TOKENS * (
        scaling_params / reference_params
    ) ** BATCH_SCALING_EXPONENT
    batch_tokens = 2 ** round(log2(predicted_tokens))
    if batch_tokens % SEQUENCE_LEN:
        raise ValueError("全局 batch token 数必须能整除序列长度")
    batch = batch_tokens // SEQUENCE_LEN
    target_tokens = TOKEN_PARAM_RATIO * scaling_params
    updates = target_tokens // batch_tokens
    fixed_flops = stats.adamw_flops + stats.muon_flops
    sample_flops = stats.forward_flops + stats.backward_flops
    return TrainingEstimate(
        batch, batch_tokens, target_tokens, updates * batch_tokens, updates,
        fixed_flops, sample_flops,
        updates * (batch * sample_flops + fixed_flops),
    )


def calculate_estimate(depth: int) -> ModelEstimate:
    """统一计算原始元素数、字节数及 FLOPs。"""
    if type(depth) is not int or depth <= 0 or ASPECT_RATIO * depth % HEAD_DIM:
        raise ValueError("depth 必须为正整数且 64L 能整除 head_dim")
    d = ASPECT_RATIO * depth
    n = (depth + 1) // 2
    blocks = 12 * depth * d**2
    embedding = VOCAB_SIZE * d
    ve = n * embedding
    lm_head = embedding
    params = blocks + embedding + ve + lm_head
    adamw = 2 * (embedding + ve + lm_head)
    muon = blocks + 12 * depth * d
    fixed = 2 * params + adamw + muon
    hidden = (20 * depth + n + 6) * SEQUENCE_LEN * d
    logits = 4 * SEQUENCE_LEN * VOCAB_SIZE
    rope = 10 * SEQUENCE_LEN * HEAD_DIM
    # BF16 embedding：权重、梯度、m、v；FP32 lm_head 同理。
    # FP32 block：权重、梯度、动量；归约状态另加。
    fixed_bytes = 8 * (embedding + ve) + 16 * lm_head + 12 * blocks + 4 * (12 * depth * d) + 2 * rope
    activation_bytes = 2 * hidden + 4 * logits
    matmul = blocks + lm_head
    forward = 2 * SEQUENCE_LEN * matmul + 2 * depth * d * SEQUENCE_LEN * (SEQUENCE_LEN + 1)
    # Flash Attention 反向重算 QKᵀ：每个有效因果位置 2d FLOPs。
    flash_recompute = depth * d * SEQUENCE_LEN * (SEQUENCE_LEN + 1)
    return ModelEstimate(
        depth, d, d // HEAD_DIM, n, blocks, embedding, ve, lm_head,
        params, adamw, muon, fixed, hidden, logits, hidden + logits, rope,
        fixed_bytes, activation_bytes, forward, 2 * forward + flash_recompute, flash_recompute,
        15 * (embedding + ve + lm_head), NS_STEPS * 60 * depth * d**3,
    )


def _count(value: int) -> str:
    for scale, suffix in ((10**9, "B"), (10**6, "M"), (10**3, "K")):
        if value >= scale:
            return f"{value / scale:.2f}{suffix}"
    return str(value)


def _table(stats: list[ModelEstimate], rows: list[tuple[str, list[str]]]) -> str:
    lines = [
        "| 项目 | " + " | ".join(f"L={s.depth}" for s in stats) + " |",
        "| :--- | " + " | ".join("---:" for _ in stats) + " |",
    ]
    lines.extend("| " + name + " | " + " | ".join(values) + " |" for name, values in rows)
    return "\n".join(lines)


def render_report(stats: list[ModelEstimate]) -> str:
    if not stats:
        raise ValueError("statistics 不能为空")

    def values(field: str) -> list[str]:
        return [_count(getattr(s, field)) for s in stats]

    def gib(field: str) -> list[str]:
        return [f"{getattr(s, field) / 2**30:.3f} GiB" for s in stats]

    configuration = _table(stats, [
        ("层数 $L$", [str(s.depth) for s in stats]),
        ("隐藏宽度 $d$", [str(s.width) for s in stats]),
        ("Q / KV 头数 $H$", [str(s.heads) for s in stats]),
        ("MLP 中间宽度 $d_{ff}$", [str(4 * s.width) for s in stats]),
        ("VE 层数 $N$", [str(s.ve_layers) for s in stats]),
    ])
    elements = _table(stats, [
        ("Transformer blocks $P_M$", values("blocks")),
        ("输入 embedding $P_E$", values("embedding")),
        ("Value embeddings $P_V$", values("value_embeddings")),
        ("模型参数合计 $P$", values("params")),
        ("AdamW 状态 $S_A$", values("adamw")),
        ("Muon 状态 $S_M$", values("muon")),
        ("训练固定元素合计 $S_{fix}$", values("fixed")),
        ("单样本激活元素合计 $A_1$", values("activations")),
    ])
    parameter_precision = "\n".join([
        "| 类别 | 参数 | 梯度 | 一阶动量 | 二阶动量 / 状态 |",
        "| :--- | :--- | :--- | :--- | :--- |",
        "| 输入 / Value embedding | BF16 | BF16 | BF16 | BF16 |",
        "| lm_head | FP32 | FP32 | FP32 | FP32 |",
        "| Transformer blocks | FP32 | FP32 | FP32 | FP32（归约状态） |",
        "| 门控 / 标量 | FP32 | FP32 | FP32 | FP32 |",
    ])
    activation_precision = "\n".join([
        "| 类别 | 精度 |",
        "| :--- | :--- |",
        "| 主干激活 | BF16 |",
        "| RoPE | BF16 |",
        "| 原始 logits | BF16 |",
        "| softcap、loss 计算 | FP32 |",
    ])
    memory = _table(stats, [
        ("固定占用 $M_{fix}$（含 RoPE）", gib("fixed_bytes")),
        ("单样本激活占用 $M_{act,1}$", gib("activation_bytes")),
        ("总占用 $M(b)$", [f"{s.fixed_bytes / 2**30:.3f} + b × {s.activation_bytes / 2**30:.3f} GiB" for s in stats]),
        ("单样本总占用 $M(1)$", [f"{(s.fixed_bytes + s.activation_bytes) / 2**30:.3f} GiB" for s in stats]),
    ])
    def flops(value: int) -> str:
        if value >= 10**12:
            return f"{value / 1e12:.3f} TFLOPs"
        return f"{value / 1e9:.3f} GFLOPs"

    compute = _table(stats, [
        ("前向 $F_{fwd}$", [flops(s.forward_flops) for s in stats]),
        ("反向（含 Flash 重算）$F_{bwd}$", [flops(s.backward_flops) for s in stats]),
        ("其中：Flash 反向重算 $F_{re}$", [flops(s.flash_recompute_flops) for s in stats]),
        ("AdamW 每次更新 $F_A$", [flops(s.adamw_flops) for s in stats]),
        ("Muon 每次更新 $F_M$", [flops(s.muon_flops) for s in stats]),
    ])
    training = [calculate_training_estimate(s) for s in stats]
    training_summary = _table(stats, [
        ("Batch $B$（全局样本数 / 更新）", [str(s.batch) for s in training]),
        ("训练 token 数 $D$", [_count(s.training_tokens) for s in training]),
        ("固定显存 $M_{fix}$", gib("fixed_bytes")),
        ("单样本显存 $M_{act,1}$（激活）", gib("activation_bytes")),
        ("固定计算 $F_{fix}$（每次更新）", [flops(s.fixed_flops) for s in training]),
        ("单样本计算 $F_1$（前向 + 反向）", [flops(s.sample_flops) for s in training]),
        ("总计算量 $F_{total}$（全程训练）", [flops(s.total_flops) for s in training]),
    ])
    sections = [
        "# GPT 训练规模估算",
        "单样本为长度 T 的序列。K/M/B 表示 10³/10⁶/10⁹ 个元素（数量后缀 B 表示十亿，公式中的 B 表示 batch 大小），GiB = 2³⁰ 字节，GFLOPs / TFLOPs = 10⁹ / 10¹² 次浮点运算（计算量，不是每秒吞吐率）；计算量达到 1 TFLOP 时使用 TFLOPs，否则使用 GFLOPs。忽略小型门控和标量；按完整因果注意力、Flash Attention、无整层激活重计算、单卡未分片估算。",
        "## 表格1：核心配置", configuration,
        r"$$d=64L,\quad H=d/h,\quad d_{ff}=4d,\quad N=\lceil L/2\rceil.$$",
        f"固定配置：$V={VOCAB_SIZE}$、$T={SEQUENCE_LEN}$、$h={HEAD_DIM}$、$k={NS_STEPS}$。Q 与 KV 总宽度均为 d；VE 隔层启用且最后一层启用。",
        "## 表格2：参数与张量元素数", elements,
        r"$$P_M=L(4d^2+2d\cdot4d)=12Ld^2,\quad P_E=Vd,\quad P_V=NVd,\quad P_O=dV.$$",
        r"$$P=P_M+P_E+P_V+P_O,\quad P_A=P_E+P_V+P_O,$$",
        r"$$S_A=2P_A,\quad R=L(4d+2\cdot4d)=12Ld,\quad S_M=P_M+R,\quad S_{fix}=2P+S_A+S_M.$$",
        "P_O 为 lm_head，已计入模型合计；AdamW 行只含 m、v，Muon 行只含动量及二阶归约状态。固定合计包含权重、梯度和优化器状态，不含 RoPE。Muon 对 m×n 矩阵保存 max(m,n) 个二阶归约元素。",
        r"$$A_h=(20L+N+6)Td,\quad A_\ell=4TV,\quad A_1=A_h+A_\ell.$$",
        "每层主干粗估 20Td：Attention 6Td、MLP 8Td、残差/归一化 6Td。MLP 的 8Td 对应 ReLU 输出与平方输出两份 T×4d 张量；VE 另计查表输出 Td，融合后的 V 已纳入 Attention。边界粗估 6Td，logits/loss 粗估 4TV。这些是预算假设，不是逐项实测；Flash Attention 不保存完整 T×T 矩阵，Q/K/V 仍占空间。",
        "## 表格3：精度配置",
        "### 参数、梯度与优化器状态", parameter_precision,
        "### 激活与缓存", activation_precision,
        r"$$b_{BF16}=2\ \mathrm{bytes},\quad b_{FP32}=4\ \mathrm{bytes}.$$",
        "梯度 dtype 跟随参数；AdamW m/v 与参数同 dtype，更新时转 FP32 运算再写回。Muon 状态保存 FP32，正交化矩阵乘法使用 BF16。RoPE 先按 FP32 计算再保存为 BF16；其缓存长度为 10T。",
        "## 表格4：显存占用估计", memory,
        r"$$M_{fix}=4b_{BF16}(P_E+P_V)+4b_{FP32}P_O+3b_{FP32}P_M+b_{FP32}R+b_{BF16}(10Th).$$",
        r"$$M_{act,1}=b_{BF16}A_h+b_{FP32}A_\ell,\quad M(b)=M_{fix}+b M_{act,1}.$$",
        "此处 b 表示同时驻留设备的样本数，与表6的全局 Batch B 区分；梯度累积不要求同时保存整个全局 batch 的激活。",
        "Embedding 和 lm_head 各计权重、梯度、m、v 四份；blocks 计权重、梯度、动量三份，归约状态另加。RoPE cos/sin 共 10Th 个元素，全局共享。输出预算按四份 FP32 等效张量计算；未另计瞬时 BF16 logits、权重转换、优化器工作区、通信与分配器开销，因此不是实际峰值。",
        "## 表格5：单样本计算量", compute,
        r"$$P_{mat}=P_M+P_O,\quad C=T(T+1)/2,$$",
        r"$$F_{fwd}\approx2TP_{mat}+4LdC,\quad F_{re}=2LdC,\quad F_{bwd}\approx2F_{fwd}+F_{re},\quad F_A\approx15P_A.$$",
        "乘加计 2 FLOPs；线性层前向为 2TP_mat，因果注意力 QKᵀ 和 AV 两次乘法合计 4LdC。反向在前向两倍基础上，加上 Flash Attention 重算 QKᵀ 的 2LdC；重算行已包含在反向行中，总量不重复相加。softmax 重算、归一化等逐元素算子仍忽略。AdamW 按每参数约 15 次标量运算预算（包括动量、偏差修正、衰减和更新），平方根/除法各计一次；不等同硬件耗时。",
        r"$$f(m,n)=k(4qr^2+2r^3),\quad q=\max(m,n),\quad r=\min(m,n),$$",
        r"$$F_M\approx L[4f(d,d)+2f(4d,d)]=60kLd^3.$$",
        "Muon 每轮计算 Gram 矩阵、Gram 平方和矩阵更新：分别为 2qr²、2r³、2qr² FLOPs。每层含四张 d×d 和两张 4d×d / d×4d 矩阵；这里只计主导矩阵乘法，未计归一化等逐元素操作。前向、反向按单样本计算，AdamW、Muon 按每次更新计算。",
        "## 表格6：训练汇总", training_summary,
        fr"$$P_{{scale}}=P_M+P_O,\quad D_{{target}}={TOKEN_PARAM_RATIO}P_{{scale}},$$",
        r"$$\widehat B_{tok}=2^{19}\left(\frac{P_{scale}}{P_{scale,L=12}}\right)^{0.383},\quad B_{tok}=2^{\mathrm{round}(\log_2\widehat B_{tok})},\quad B=B_{tok}/T.$$",
        r"$$n_{step}=\left\lfloor\frac{D_{target}}{B_{tok}}\right\rfloor,\quad D=n_{step}B_{tok},$$",
        r"$$F_{fix}=F_A+F_M,\quad F_1=F_{fwd}+F_{bwd},\quad F_{total}=n_{step}(BF_1+F_{fix})=\frac{D}{T}F_1+n_{step}F_{fix}.$$",
        f"Batch 是一次更新的全局样本数，训练 token 数为整个训练过程累计处理的 token，按完整 batch 向下取整。固定显存与单样本激活显存沿用表4；固定计算是每次优化器更新的开销，单样本计算含 Flash 反向重算。总计算量累加全部样本的前后向和全部优化器更新。训练预算采用官方 speedrun 的 token/参数比 {TOKEN_PARAM_RATIO}，Batch 使用官方自动缩放规则；参数数量仍忽略小型门控和标量。",
        "训练预算依据：[speedrun 脚本](https://github.com/karpathy/nanochat/blob/master/runs/speedrun.sh)；Batch 依据：[训练脚本](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py)。",
        "Flash Attention 重算依据：[FlashAttention 论文](https://arxiv.org/abs/2205.14135)。",
        "精度及优化器依据：[模型代码](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py)、[优化器代码](https://github.com/karpathy/nanochat/blob/master/nanochat/optim.py)。",
    ]
    depth24 = next((s for s in stats if s.depth == 24), None)
    if depth24 is not None:
        sections.extend(render_hardware_estimate(depth24))
    return "\n\n".join(sections) + "\n"


def print_param_tables(depths: tuple[int, ...] = (4, 12, 24), *, output_path: str | Path | None = None) -> str:
    report = render_report([calculate_estimate(depth) for depth in depths])
    if output_path is None:
        print(report, end="")
    else:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
    return report


if __name__ == "__main__":
    report_path = Path(__file__).resolve().parent.parent / "local" / "params_count.md"
    print_param_tables(output_path=report_path)
    print(f"已生成报告：{report_path}")
