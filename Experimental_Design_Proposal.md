# 实验方案设计文档

**项目名称**：TurboQuant 相关 VLM 并发测试实验方案  
**实验目标**：在学院 GPU 集群上，以 Qwen 系列小型 VLM 为服务对象，使用 VLM 标准评测数据集构造并发请求，观察不同并发强度和不同量化策略下的吞吐、延迟、显存、稳定性与基础准确率变化。  
**文档版本**：v1.0  
**生成日期**：2026-05-20  

---

## 0. 先给结论

本实验建议分为三层跑通：

1. **最小可跑通层**：使用 `Qwen/Qwen2.5-VL-3B-Instruct` + `LLaVA-Bench-in-the-Wild` 小样本，验证环境、模型加载、图片输入、OpenAI API 请求和日志采集是否正常。
2. **正式并发测试层**：使用 `Qwen/Qwen2.5-VL-3B-Instruct` + **MMStar** 全量或 500/1500 子集，在并发数 `1, 2, 4, 8, 16, 32` 下测吞吐、TTFT、TPOT、端到端延迟、显存、GPU 利用率和错误率。
3. **量化对比层**：比较 `BF16 原始模型`、`Qwen2.5-VL-3B-Instruct-AWQ`、以及可选的 `FP8 KV Cache`。TurboQuant 本身在论文中主要是在线向量量化 / KV cache 量化思路，并不是当前 vLLM 里可直接一键开启的 VLM 服务选项。因此第一阶段先用可部署的 AWQ / FP8 KV cache 量化基线把并发实验跑通；第二阶段再考虑把 TurboQuant 的 KV cache 量化或残差 QJL 思路接入推理框架。

**最终选定主数据集：MMStar。** 原因是：样本量适中，1500 条足够做并发压测；每条样本是视觉依赖型 VLM 问答，能避免很多纯文本泄漏；任务通常较短，适合反复跑并发，不会像长视频或超长 OCR 数据那样一上来就拖垮环境。

---

## 1. 实验背景与问题定义

导师要求“先在 VLM 上，利用 TurboQuant 或其他量化相关研究常用的数据集跑一下并发测试，观察几个主要指标看看并发效果”。这句话可以拆成三个任务：

### 1.1 研究对象

本实验不是先做训练，也不是先证明 TurboQuant 理论，而是先做**服务端推理并发测试**：

- 给定一个 VLM 模型；
- 给定一组图文输入样本；
- 部署成服务；
- 从客户端以不同并发数请求；
- 记录延迟、吞吐、显存、GPU 使用率、错误率、基础准确率；
- 比较不同量化模式是否提高并发能力。

### 1.2 与 TurboQuant 的关系

TurboQuant 论文关注在线向量量化，核心目标是压缩高维向量并尽量保留 MSE 或内积结构。论文明确强调 KV cache 是重要应用场景，因为长上下文推理时 KV cache 随层数、注意力头数和上下文长度增长，会带来显存与延迟瓶颈。TurboQuant 提出两个方向：

- `TurboQuant_mse`：随机旋转后按坐标使用近似最优标量量化，优化 MSE；
- `TurboQuant_prod`：先做 MSE 量化，再对残差做 1-bit QJL，以获得无偏内积估计。

因此，在 VLM 并发服务中，最贴近 TurboQuant 的测试点不是“模型权重量化”本身，而是：

- 并发升高时 KV cache 显存是否成为瓶颈；
- KV cache 量化后是否能容纳更多并发请求；
- 量化后延迟、吞吐、质量是否变化；
- 图像 token + 文本 token 的混合输入是否放大 KV cache 压力。

---

## 2. 已上传资源解读

### 2.1 《学院GPU集群使用方法》中的硬约束

实验必须遵守以下规则：

1. **只能登录主节点做资源申请与管理，不能在主节点跑程序。**
2. 交互式调试使用 `salloc -G 1 -N 1` 申请 GPU，再进入计算节点。
3. 正式实验建议使用 `sbatch` 提交脚本。
4. 进入计算节点后再加载环境，例如 `module add miniconda3`。
5. 用户自建 conda 环境默认在家目录，家目录配额有限，建议把 `.conda` 和 `.cache` 移到 workspace 后软链接。
6. 跑完后及时 `exit` 释放计算资源。

### 2.2 集群资源信息

根据文档中的资源表，集群大致包括：

| 节点 | CPU 核心 | GPU 数量 | GPU 型号 | 建议用途 |
|---|---:|---:|---|---|
| compute1 | 72 | 8 | NVIDIA L40S | VLM 3B/7B 调试与正式跑 |
| compute2 | 72 | 8 | NVIDIA L40S | VLM 3B/7B 调试与正式跑 |
| compute3 | 112 | 8 | NVIDIA H20 | 高并发、较大模型、正式跑 |
| compute5 | 96 | 4 | NVIDIA L40S | 单卡/多卡轻量实验 |
| compute6 | 96 | 8 | NVIDIA H20-3e | 高并发、较大模型、正式跑 |

配额大致为：

| 任务类型 | 最大资源限制 | 最长运行时间 | 建议 |
|---|---:|---:|---|
| `salloc` | 2 GPU | 24 小时 | 只做调试 |
| `sbatch` | 4 GPU | 7 × 24 小时 | 正式实验 |
| `~/home` | 约 50GB | - | 不放模型和数据 |
| `~/workspace` 或 `/data/private/$USER/workspace` | 约 500GB | - | 放模型、数据、日志、输出 |

---

## 3. 联网调研后的数据集选择

### 3.1 候选数据集对比

| 数据集 | 类型 | 规模 | 优点 | 缺点 | 本实验用途 |
|---|---|---:|---|---|---|
| **MMStar** | 视觉依赖型 VLM 多选题 | 1500 | 样本适中，6 大能力、18 个细分维度；适合正式并发和基础准确率统计 | 不如 VQAv2 那样超大规模 | **主数据集** |
| LLaVA-Bench-in-the-Wild | 图文问答 / 视觉指令 | 24 图 / 60 问 | 很小，下载快，适合测试图像输入链路 | 样本太少，不适合正式压测 | smoke test |
| MME | VLM 感知与认知评测 | 约 2374 | 标准 VLM 评测，样本较多 | 数据较大，第一次下载和处理更慢 | 可选压力测试 |
| ScienceQA | 多模态科学问答 | 约 21k | 多选题，方便自动评分，样本多 | 有些样本不一定强视觉依赖 | 可选大样本测试 |
| LongBench | 长文本 LLM 评测 | 21 个任务 | 与 TurboQuant 论文 KV cache 实验更接近 | 不是 VLM 数据集 | 只作为 KV cache 长文本补充实验 |

### 3.2 最终选择

#### 主数据集：MMStar

选择 MMStar 作为主数据集，理由如下：

1. **符合 VLM 要求**：它是图文多模态评测，不是纯文本。
2. **视觉依赖更强**：MMStar 的设计目标之一是减少“只靠文本就能答对”的样本。
3. **规模适合并发测试**：1500 条样本足够覆盖 `1,2,4,8,16,32` 并发测试，不会太小，也不会像 VQAv2 / ScienceQA 全量那样一开始就过重。
4. **易于自动评分**：多选题可以用正则抽取 A/B/C/D，与真实答案比对，得到基础准确率。
5. **适合初学者复现**：比视频、多图、长文档数据更容易跑通。

#### 调试数据集：LLaVA-Bench-in-the-Wild

第一天只建议用它做链路测试，因为样本很少，能快速验证：

- 图片能否正确加载；
- vLLM 是否能接收 multimodal chat request；
- 客户端是否能正确异步并发；
- 日志字段是否完整。

#### 可选补充：LongBench

LongBench 不是 VLM 数据集，但 TurboQuant 论文中 KV cache 实验使用 LongBench-V1 做长上下文下游任务。因此，如果后续导师追问“与 TurboQuant 论文的实验更像的负载在哪里”，可以补跑 LongBench 的少量长上下文样本，作为**非 VLM 的 KV cache 对齐实验**。

---

## 4. 模型与量化方案设计

### 4.1 模型选择

本实验优先使用：

- `Qwen/Qwen2.5-VL-3B-Instruct`
- `Qwen/Qwen2.5-VL-3B-Instruct-AWQ`

理由：

1. 3B 级别适合学院单卡 L40S/H20 上快速跑通；
2. 是 VLM，可以处理图片 + 文本；
3. 有官方 AWQ 版本，便于做量化对比；
4. 与之前“使用 qwen-3.6B 或相近 3B 级模型”的方向一致。

如果学院集群已经有本地模型路径，则优先使用本地路径，避免重复下载：

```bash
/data/private/$USER/workspace/models/Qwen2.5-VL-3B-Instruct
/data/private/$USER/workspace/models/Qwen2.5-VL-3B-Instruct-AWQ
```

### 4.2 量化对照组

| 组别 | 模型 / 配置 | 目的 |
|---|---|---|
| A | Qwen2.5-VL-3B-Instruct, BF16 | 全精度基线 |
| B | Qwen2.5-VL-3B-Instruct-AWQ | 权重量化基线 |
| C | BF16 + `--kv-cache-dtype fp8` | KV cache 量化基线，最接近 TurboQuant 应用方向 |
| D | AWQ + `--kv-cache-dtype fp8` | 权重 + KV cache 组合压缩 |
| E | TurboQuant KV cache 接入版 | 后续扩展，第一阶段不强制 |

说明：TurboQuant 是研究算法，当前不是一个在 vLLM 中可直接 `--quantization turboquant` 开启的成熟选项。为了先把并发实验跑通，第一阶段使用 vLLM 支持较好的 AWQ 和 FP8 KV cache 作为量化基线；当这些指标跑通后，再把 TurboQuant 的思想迁移到 KV cache hook 或自定义 attention kernel 中。

---

## 5. 需要观察的指标

### 5.1 服务性能指标

| 指标 | 英文 | 含义 | 重要性 |
|---|---|---|---|
| 请求吞吐 | requests/s | 每秒完成多少请求 | 并发能力核心指标 |
| 输出吞吐 | output tokens/s | 每秒生成多少 token | 推理服务核心指标 |
| 总吞吐 | total tokens/s | prompt + output 总 token | 对比预填充和解码压力 |
| 端到端延迟 | E2E latency | 从请求发出到完整返回 | 用户体验指标 |
| 首 token 延迟 | TTFT | 从请求发出到第一个 token | prefilling 和排队压力 |
| 每 token 延迟 | TPOT | 输出阶段每个 token 平均时间 | decode 速度 |
| P50/P95/P99 | percentile latency | 不同分位延迟 | 服务稳定性 |
| 错误率 | error rate | 失败 / 超时 / OOM 比例 | 可用性 |

### 5.2 GPU 指标

| 指标 | 含义 |
|---|---|
| 峰值显存 | 并发达到某个值时是否 OOM |
| 平均显存 | 量化后显存是否下降 |
| GPU 利用率 | 是否吃满 GPU |
| 功耗 | 可选 |
| 请求排队现象 | 并发过高时 TTFT 是否爆炸 |

### 5.3 质量指标

| 指标 | 含义 |
|---|---|
| Accuracy | 多选题是否答对 |
| parse_success_rate | 是否能从输出中抽取 A/B/C/D |
| avg_output_len | 输出长度是否异常变化 |
| refusal/error count | 模型是否拒答或输出无关内容 |

并发实验不是完整模型评测，质量指标只用于判断量化是否造成明显退化。

---

## 6. 实验矩阵

### 6.1 第一阶段：最小跑通

| 项目 | 设置 |
|---|---|
| 模型 | Qwen2.5-VL-3B-Instruct BF16 |
| 数据集 | LLaVA-Bench-in-the-Wild |
| 样本数 | 10 / 60 |
| 并发 | 1, 2 |
| max_tokens | 32 |
| 目标 | 验证链路 |

### 6.2 第二阶段：正式并发测试

| 项目 | 设置 |
|---|---|
| 模型 | Qwen2.5-VL-3B-Instruct BF16 |
| 数据集 | MMStar |
| 样本数 | 500 先跑，稳定后 1500 |
| 并发 | 1, 2, 4, 8, 16, 32 |
| max_tokens | 16 或 32 |
| 重复次数 | 每组 3 次 |
| 目标 | 得到并发曲线 |

### 6.3 第三阶段：量化对比

| 组别 | 模型配置 | 并发 |
|---|---|---|
| BF16 | Qwen2.5-VL-3B-Instruct | 1,2,4,8,16,32 |
| AWQ | Qwen2.5-VL-3B-Instruct-AWQ | 1,2,4,8,16,32 |
| FP8 KV | BF16 + FP8 KV cache | 1,2,4,8,16,32 |
| AWQ + FP8 KV | AWQ + FP8 KV cache | 1,2,4,8,16,32 |

### 6.4 第四阶段：极限压力测试

| 项目 | 设置 |
|---|---|
| 数据集 | MMStar 1500 或 MME |
| 并发 | 32, 48, 64 |
| max_tokens | 16 |
| 目标 | 找到 OOM / 延迟崩溃临界点 |
| 注意 | 只在前面阶段稳定后再跑 |

---

## 7. 集群从 0 到 1 操作流程

下面的命令中，`$USER` 表示你的学院集群用户名。

### 7.1 登录主节点

```bash
ssh $USER@202.112.194.86
```

主节点只做下面几件事：

- 查看队列；
- 申请资源；
- 提交 sbatch；
- 查看日志；
- 不运行 Python、下载大模型或启动服务。

### 7.2 查看节点状态

```bash
sinfo -o "%N %T %C %m %G"
squeue --me
```

### 7.3 申请交互式 GPU 做调试

```bash
salloc -G 1 -N 1
```

如果想指定节点：

```bash
salloc -G 1 -N 1 --nodelist=compute2
```

申请成功后进入对应计算节点，例如：

```bash
ssh compute2
```

确认 GPU：

```bash
nvidia-smi
```

### 7.4 加载 conda

```bash
module purge
module add miniconda3
conda env list
```

### 7.5 设置 workspace 与缓存

```bash
mkdir -p /data/private/$USER/workspace/{models,datasets,projects,logs,outputs,cache/hf,cache/torch}

# 推荐写入 ~/.bashrc，之后每次登录自动生效
cat >> ~/.bashrc <<'EOF'
export WORKSPACE=/data/private/$USER/workspace
export HF_HOME=$WORKSPACE/cache/hf
export TRANSFORMERS_CACHE=$WORKSPACE/cache/hf
export HF_DATASETS_CACHE=$WORKSPACE/cache/hf/datasets
export TORCH_HOME=$WORKSPACE/cache/torch
EOF

source ~/.bashrc
```

如果 `.conda` 和 `.cache` 已经占用 home 很多空间，可执行软链接迁移：

```bash
cd ~
mkdir -p /data/private/$USER/workspace/dotfiles

if [ -d ~/.conda ] && [ ! -L ~/.conda ]; then
  mv ~/.conda /data/private/$USER/workspace/dotfiles/.conda
  ln -s /data/private/$USER/workspace/dotfiles/.conda ~/.conda
fi

if [ -d ~/.cache ] && [ ! -L ~/.cache ]; then
  mv ~/.cache /data/private/$USER/workspace/dotfiles/.cache
  ln -s /data/private/$USER/workspace/dotfiles/.cache ~/.cache
fi
```

### 7.6 创建项目目录

```bash
mkdir -p /data/private/$USER/workspace/projects/vlm_concurrency_bench
cd /data/private/$USER/workspace/projects/vlm_concurrency_bench

mkdir -p scripts data/mmstar data/llava_bench results logs sbatch
```

### 7.7 创建 conda 环境

```bash
module purge
module add miniconda3

conda create -n vlm_bench python=3.10 -y
conda activate vlm_bench

python -m pip install --upgrade pip
pip install -U "vllm" "transformers" "accelerate" "datasets" "pillow" \
  "openai" "aiohttp" "pandas" "tqdm" "pyarrow" "qwen-vl-utils[decord]" \
  "huggingface_hub" "modelscope" "pynvml"
```

验证：

```bash
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

---

## 8. 下载模型与数据集

### 8.1 Hugging Face 下载模型

如果集群能访问 Hugging Face：

```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir /data/private/$USER/workspace/models/Qwen2.5-VL-3B-Instruct

huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct-AWQ \
  --local-dir /data/private/$USER/workspace/models/Qwen2.5-VL-3B-Instruct-AWQ
```

### 8.2 ModelScope 下载模型

如果 Hugging Face 访问慢，使用 ModelScope：

```bash
modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct \
  --local_dir /data/private/$USER/workspace/models/Qwen2.5-VL-3B-Instruct

modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct-AWQ \
  --local_dir /data/private/$USER/workspace/models/Qwen2.5-VL-3B-Instruct-AWQ
```

### 8.3 准备 MMStar 数据

创建 `scripts/prepare_mmstar.py`：

```python
import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def save_image(img, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(img, Image.Image):
        img.convert("RGB").save(path)
    else:
        Image.open(img).convert("RGB").save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/mmstar")
    parser.add_argument("--limit", type=int, default=1500)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    out_jsonl = out_dir / "mmstar_requests.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("Lin-Chen/MMStar", split="val")
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    with out_jsonl.open("w", encoding="utf-8") as f:
        for i, item in enumerate(tqdm(ds)):
            image_path = img_dir / f"{i:06d}.jpg"
            image = item.get("image")
            save_image(image, image_path)

            question = item.get("question", "")
            answer = item.get("answer", "")
            category = item.get("category", "")
            l2_category = item.get("l2_category", "")

            prompt = (
                "You are a vision-language model. "
                "Answer the following multiple-choice question. "
                "Only output the option letter, such as A, B, C, or D.\n\n"
                f"{question}"
            )

            row = {
                "id": i,
                "image_path": str(image_path.resolve()),
                "prompt": prompt,
                "answer": answer,
                "category": category,
                "l2_category": l2_category,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"saved: {out_jsonl}")


if __name__ == "__main__":
    main()
```

运行：

```bash
python scripts/prepare_mmstar.py --out_dir data/mmstar --limit 500
```

稳定后准备全量：

```bash
python scripts/prepare_mmstar.py --out_dir data/mmstar_full --limit 1500
```

---

## 9. 启动 vLLM 服务

### 9.1 BF16 基线服务

```bash
MODEL=/data/private/$USER/workspace/models/Qwen2.5-VL-3B-Instruct

vllm serve $MODEL \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen2_5_vl_3b_bf16 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --limit-mm-per-prompt image=1 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 \
  --disable-log-requests \
  --no-enable-prefix-caching
```

说明：

- `--limit-mm-per-prompt image=1` 限制每条请求 1 张图；
- `--max-model-len 8192` 先保守，不要一开始就开 128k；
- `--max-num-seqs 64` 是服务端能同时调度的请求上限之一；
- 正式 benchmark 时建议关闭 prefix caching，否则相同前缀会影响压测结果。

如果你的 vLLM 版本不支持 `--no-enable-prefix-caching`，执行：

```bash
vllm serve --help | grep -i prefix
```

按当前版本的参数名修改。

### 9.2 AWQ 模型服务

```bash
MODEL=/data/private/$USER/workspace/models/Qwen2.5-VL-3B-Instruct-AWQ

vllm serve $MODEL \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen2_5_vl_3b_awq \
  --dtype half \
  --max-model-len 8192 \
  --limit-mm-per-prompt image=1 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 \
  --disable-log-requests \
  --no-enable-prefix-caching
```

如果模型 config 已经声明 AWQ，一般不需要手动加 `--quantization awq`。如果启动日志提示没有识别量化类型，再尝试：

```bash
--quantization awq
```

### 9.3 FP8 KV cache 服务

```bash
MODEL=/data/private/$USER/workspace/models/Qwen2.5-VL-3B-Instruct

vllm serve $MODEL \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen2_5_vl_3b_bf16_fp8kv \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --max-model-len 8192 \
  --limit-mm-per-prompt image=1 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 \
  --disable-log-requests \
  --no-enable-prefix-caching
```

注意：FP8 KV cache 的可用性取决于 vLLM 版本、CUDA、GPU 架构和 attention backend。如果启动失败，先不阻塞主实验，记录失败信息，并继续跑 BF16/AWQ。

---

## 10. 编写并发压测客户端

创建 `scripts/bench_vlm_async.py`：

```python
import argparse
import asyncio
import base64
import json
import re
import statistics
import time
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm import tqdm


def image_to_base64(path: str) -> str:
    suffix = Path(path).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def parse_choice(text: str):
    if not text:
        return None
    text = text.strip().upper()
    m = re.search(r"\b([ABCD])\b", text)
    if m:
        return m.group(1)
    if text and text[0] in "ABCD":
        return text[0]
    return None


async def one_request(session, url, model, item, max_tokens, timeout):
    image_url = image_to_base64(item["image_path"])
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": item["prompt"]},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
    }

    headers = {"Content-Type": "application/json"}
    start = time.perf_counter()
    first_token_time = None
    text_parts = []
    error = None

    try:
        async with session.post(
            url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                error = f"HTTP_{resp.status}: {await resp.text()}"
            else:
                async for raw in resp.content:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj["choices"][0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            text_parts.append(token)
                    except Exception:
                        continue
    except Exception as e:
        error = repr(e)

    end = time.perf_counter()
    output_text = "".join(text_parts).strip()
    pred = parse_choice(output_text)
    gold = str(item.get("answer", "")).strip().upper()[:1]
    ok = (pred == gold) if gold else None

    return {
        "id": item.get("id"),
        "latency_s": end - start,
        "ttft_s": (first_token_time - start) if first_token_time else None,
        "output_text": output_text,
        "pred": pred,
        "gold": gold,
        "correct": ok,
        "error": error,
    }


async def run(args):
    rows = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    rows = rows[: args.num_prompts]
    sem = asyncio.Semaphore(args.concurrency)
    url = args.base_url.rstrip("/") + "/chat/completions"

    async with aiohttp.ClientSession() as session:
        async def bound_request(item):
            async with sem:
                return await one_request(
                    session=session,
                    url=url,
                    model=args.model,
                    item=item,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )

        start = time.perf_counter()
        tasks = [asyncio.create_task(bound_request(item)) for item in rows]
        results = []
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            results.append(await task)
        end = time.perf_counter()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    df = pd.DataFrame(results)
    total_time = end - start
    success_df = df[df["error"].isna()]

    def pct(series, q):
        vals = [x for x in series.tolist() if pd.notna(x)]
        if not vals:
            return None
        vals = sorted(vals)
        k = min(len(vals) - 1, int(round((q / 100) * (len(vals) - 1))))
        return vals[k]

    summary = {
        "model": args.model,
        "data": args.data,
        "num_prompts": len(rows),
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "total_time_s": total_time,
        "requests_per_s": len(rows) / total_time if total_time > 0 else None,
        "success": int(df["error"].isna().sum()),
        "errors": int(df["error"].notna().sum()),
        "error_rate": float(df["error"].notna().mean()),
        "accuracy": float(success_df["correct"].mean()) if len(success_df) else None,
        "latency_mean_s": float(success_df["latency_s"].mean()) if len(success_df) else None,
        "latency_p50_s": pct(success_df["latency_s"], 50),
        "latency_p95_s": pct(success_df["latency_s"], 95),
        "latency_p99_s": pct(success_df["latency_s"], 99),
        "ttft_mean_s": float(success_df["ttft_s"].dropna().mean()) if len(success_df) else None,
        "ttft_p50_s": pct(success_df["ttft_s"].dropna(), 50) if len(success_df) else None,
        "ttft_p95_s": pct(success_df["ttft_s"].dropna(), 95) if len(success_df) else None,
        "parse_success_rate": float(success_df["pred"].notna().mean()) if len(success_df) else None,
    }

    summary_path = Path(args.out).with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
```

---

## 11. GPU 监控脚本

创建 `scripts/gpu_monitor.sh`：

```bash
#!/bin/bash
OUT=$1
INTERVAL=${2:-1}

mkdir -p $(dirname "$OUT")

echo "timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw" > "$OUT"

while true; do
  nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits >> "$OUT"
  sleep "$INTERVAL"
done
```

赋权：

```bash
chmod +x scripts/gpu_monitor.sh
```

---

## 12. 手动调试流程

开一个终端启动服务：

```bash
cd /data/private/$USER/workspace/projects/vlm_concurrency_bench
conda activate vlm_bench

MODEL=/data/private/$USER/workspace/models/Qwen2.5-VL-3B-Instruct

vllm serve $MODEL \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen2_5_vl_3b_bf16 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --limit-mm-per-prompt image=1 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 \
  --disable-log-requests \
  --no-enable-prefix-caching
```

另一个终端进入同一计算节点，准备数据：

```bash
cd /data/private/$USER/workspace/projects/vlm_concurrency_bench
conda activate vlm_bench

python scripts/prepare_mmstar.py --out_dir data/mmstar --limit 100
```

运行并发为 1 的测试：

```bash
python scripts/bench_vlm_async.py \
  --base_url http://127.0.0.1:8000/v1 \
  --model qwen2_5_vl_3b_bf16 \
  --data data/mmstar/mmstar_requests.jsonl \
  --num_prompts 20 \
  --concurrency 1 \
  --max_tokens 32 \
  --out results/debug_bf16_c1.jsonl
```

并发为 4：

```bash
python scripts/bench_vlm_async.py \
  --base_url http://127.0.0.1:8000/v1 \
  --model qwen2_5_vl_3b_bf16 \
  --data data/mmstar/mmstar_requests.jsonl \
  --num_prompts 100 \
  --concurrency 4 \
  --max_tokens 32 \
  --out results/debug_bf16_c4.jsonl
```

成功标准：

- 服务不 OOM；
- 请求能返回；
- `.summary.json` 里有 latency、TTFT、accuracy、error_rate；
- `error_rate` 接近 0；
- `parse_success_rate` 不低于 80%。

---

## 13. 正式 sbatch 脚本

创建 `sbatch/run_vlm_bench.sbatch`：

```bash
#!/bin/bash
#SBATCH -J vlm_bench
#SBATCH -o /data/private/%u/workspace/projects/vlm_concurrency_bench/logs/%j.out
#SBATCH -e /data/private/%u/workspace/projects/vlm_concurrency_bench/logs/%j.err
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00

set -e

module purge
module add miniconda3
source ~/.bashrc
conda activate vlm_bench

cd /data/private/$USER/workspace/projects/vlm_concurrency_bench

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/data/private/$USER/workspace/cache/hf
export TRANSFORMERS_CACHE=/data/private/$USER/workspace/cache/hf
export HF_DATASETS_CACHE=/data/private/$USER/workspace/cache/hf/datasets
export TORCH_HOME=/data/private/$USER/workspace/cache/torch

MODEL_KIND=${MODEL_KIND:-bf16}
DATA=${DATA:-data/mmstar/mmstar_requests.jsonl}
NUM_PROMPTS=${NUM_PROMPTS:-500}
MAX_TOKENS=${MAX_TOKENS:-32}
CONCURRENCY_LIST=${CONCURRENCY_LIST:-"1 2 4 8 16 32"}

if [ "$MODEL_KIND" = "bf16" ]; then
  MODEL_PATH=/data/private/$USER/workspace/models/Qwen2.5-VL-3B-Instruct
  SERVED_NAME=qwen2_5_vl_3b_bf16
  EXTRA_ARGS="--dtype bfloat16"
elif [ "$MODEL_KIND" = "awq" ]; then
  MODEL_PATH=/data/private/$USER/workspace/models/Qwen2.5-VL-3B-Instruct-AWQ
  SERVED_NAME=qwen2_5_vl_3b_awq
  EXTRA_ARGS="--dtype half"
elif [ "$MODEL_KIND" = "bf16_fp8kv" ]; then
  MODEL_PATH=/data/private/$USER/workspace/models/Qwen2.5-VL-3B-Instruct
  SERVED_NAME=qwen2_5_vl_3b_bf16_fp8kv
  EXTRA_ARGS="--dtype bfloat16 --kv-cache-dtype fp8"
else
  echo "Unknown MODEL_KIND=$MODEL_KIND"
  exit 1
fi

mkdir -p results logs

echo "Starting GPU monitor..."
bash scripts/gpu_monitor.sh logs/${SLURM_JOB_ID}_${MODEL_KIND}_gpu.csv 1 &
MONITOR_PID=$!

echo "Starting vLLM server: $MODEL_KIND"
vllm serve $MODEL_PATH \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name $SERVED_NAME \
  --max-model-len 8192 \
  --limit-mm-per-prompt image=1 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 \
  --disable-log-requests \
  --no-enable-prefix-caching \
  $EXTRA_ARGS > logs/${SLURM_JOB_ID}_${MODEL_KIND}_server.log 2>&1 &

SERVER_PID=$!

echo "Waiting for server..."
sleep 120

for C in $CONCURRENCY_LIST; do
  echo "Running concurrency=$C"
  python scripts/bench_vlm_async.py \
    --base_url http://127.0.0.1:8000/v1 \
    --model $SERVED_NAME \
    --data $DATA \
    --num_prompts $NUM_PROMPTS \
    --concurrency $C \
    --max_tokens $MAX_TOKENS \
    --out results/${SLURM_JOB_ID}_${MODEL_KIND}_c${C}.jsonl

  sleep 10
done

echo "Stopping server and monitor..."
kill $SERVER_PID || true
kill $MONITOR_PID || true

echo "Done."
```

提交 BF16：

```bash
sbatch --export=ALL,MODEL_KIND=bf16,NUM_PROMPTS=500,MAX_TOKENS=32,CONCURRENCY_LIST="1 2 4 8 16 32" \
  sbatch/run_vlm_bench.sbatch
```

提交 AWQ：

```bash
sbatch --export=ALL,MODEL_KIND=awq,NUM_PROMPTS=500,MAX_TOKENS=32,CONCURRENCY_LIST="1 2 4 8 16 32" \
  sbatch/run_vlm_bench.sbatch
```

提交 FP8 KV：

```bash
sbatch --export=ALL,MODEL_KIND=bf16_fp8kv,NUM_PROMPTS=500,MAX_TOKENS=32,CONCURRENCY_LIST="1 2 4 8 16 32" \
  sbatch/run_vlm_bench.sbatch
```

查看任务：

```bash
squeue --me
sacct -j JOB_ID
tail -f logs/JOB_ID.out
tail -f logs/JOB_ID_bf16_server.log
```

取消任务：

```bash
scancel JOB_ID
```

---

## 14. 汇总实验结果

创建 `scripts/collect_results.py`：

```python
import glob
import json
from pathlib import Path

import pandas as pd


def main():
    rows = []
    for path in glob.glob("results/*.summary.json"):
        with open(path, "r", encoding="utf-8") as f:
            row = json.load(f)
        row["file"] = path
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No summary found.")
        return

    cols = [
        "model", "concurrency", "num_prompts", "max_tokens",
        "requests_per_s", "latency_mean_s", "latency_p50_s",
        "latency_p95_s", "latency_p99_s", "ttft_mean_s",
        "ttft_p50_s", "ttft_p95_s", "accuracy",
        "parse_success_rate", "error_rate", "file",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values(["model", "concurrency"])
    out = "results/summary_all.csv"
    df.to_csv(out, index=False)
    print(df)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
```

运行：

```bash
python scripts/collect_results.py
```

最终得到：

```bash
results/summary_all.csv
logs/*_gpu.csv
results/*.jsonl
results/*.summary.json
```

---

## 15. 结果分析方法

### 15.1 并发曲线怎么解读

画四类图：

1. `concurrency -> requests/s`
2. `concurrency -> latency_p95_s`
3. `concurrency -> ttft_p95_s`
4. `concurrency -> peak_gpu_memory`

理想情况：

- 从并发 1 到 8，吞吐逐步上升；
- 延迟也上升，但不爆炸；
- 到某个并发后吞吐不再上升，说明达到饱和；
- 再继续增加并发，P95/P99 延迟急剧上升，错误率增加；
- 这个点就是该配置下的合理并发上限。

### 15.2 如何比较量化效果

重点比较同一并发下：

| 对比项 | 观察 |
|---|---|
| BF16 vs AWQ | 权重量化是否降低显存，是否提高吞吐 |
| BF16 vs FP8 KV | KV cache 量化是否让高并发更稳定 |
| AWQ vs AWQ + FP8 KV | 组合量化是否进一步提高并发上限 |
| BF16 vs 量化 | accuracy 是否明显下降 |

### 15.3 预期现象

可能出现以下几种情况：

1. **AWQ 显存下降，但吞吐不一定上升。**  
   权重量化主要减少权重显存，但 VLM 请求包含图像预处理和 KV cache，吞吐还受调度、attention、图像 encoder 等影响。

2. **FP8 KV cache 在高并发或长上下文时更有意义。**  
   如果 prompt 很短、并发很低，KV cache 并不是瓶颈，FP8 KV 的收益可能不明显。

3. **并发过高时 TTFT 比 TPOT 更容易先爆炸。**  
   因为新请求进入时需要排队和 prefill，prefill 阶段会受图像 token 与 prompt token 长度影响。

4. **parse_success_rate 很重要。**  
   如果模型没有按 A/B/C/D 输出，accuracy 会失真，需要改 prompt，让它只输出选项。

---

## 16. 最终报告建议结构

跑完后给导师汇报可以按这个结构：

1. 实验目的  
   在 VLM 并发服务场景下观察量化对吞吐、延迟、显存和基础准确率的影响。

2. 实验环境  
   学院 GPU 集群，单卡 L40S/H20，vLLM，Qwen2.5-VL-3B-Instruct。

3. 数据集选择  
   主数据集 MMStar，调试集 LLaVA-Bench-in-the-Wild。

4. 实验变量  
   并发数、模型量化方式、max_tokens、样本数量。

5. 核心指标  
   requests/s、TTFT、TPOT、P95/P99 latency、显存、错误率、accuracy。

6. 实验结果图  
   至少四张图：吞吐曲线、延迟曲线、显存曲线、准确率柱状图。

7. 初步结论  
   找出每种量化方式的合理并发上限，判断量化是否真正提升服务能力。

8. 下一步  
   将 TurboQuant 的 KV cache 量化机制接入 VLM 推理框架，替换 FP8 KV cache 基线。

---

## 17. 常见问题与处理

### 17.1 主节点能不能跑下载或 Python？

不建议。主节点只做登录、查看队列、申请资源、提交任务。模型下载、数据处理、Python 脚本、vLLM 服务都应在计算节点或 sbatch 任务里完成。

### 17.2 vLLM 服务启动很慢怎么办？

VLM 第一次启动要加载模型、编译 kernel、加载 processor，可能需要几分钟。sbatch 脚本里先 `sleep 120`，如果模型大或节点慢，可以改成 `sleep 240`。

### 17.3 端口 8000 被占用怎么办？

换端口：

```bash
--port 8001
```

客户端同步修改：

```bash
--base_url http://127.0.0.1:8001/v1
```

### 17.4 OOM 怎么办？

按顺序降低：

1. `--max-num-seqs 64 -> 32 -> 16`
2. 并发列表去掉 32 / 64
3. `--max-model-len 8192 -> 4096`
4. `--gpu-memory-utilization 0.90 -> 0.85`
5. 使用 AWQ 或 FP8 KV
6. 换 H20 节点或申请多 GPU

### 17.5 精度很低怎么办？

先确认是不是解析问题：

- 输出是否不是 A/B/C/D；
- prompt 是否太复杂；
- 数据字段是否取错；
- 正确答案是否是 `A`、`B`、`C`、`D` 格式。

可以把 prompt 改成：

```text
Look at the image and answer the question.
You must choose one option from A, B, C, D.
Only output a single letter.
```

### 17.6 AWQ 启动失败怎么办？

尝试：

```bash
vllm serve $MODEL_PATH --quantization awq --dtype half ...
```

如果仍失败，先跳过 AWQ，保留 BF16 和 FP8 KV 两组，把失败日志保存下来。

### 17.7 FP8 KV cache 启动失败怎么办？

这可能是 vLLM 版本、CUDA、GPU 架构或 attention backend 不兼容。不要卡住主实验，先记录失败日志，正式结果里说明“该集群当前软件栈下 FP8 KV 未跑通”。

---



