# TurboQuant 与普通方法并发实验完整中文技术报告

## 1. 摘要

本报告合并并统一整理了本项目中已完成的全部相关报告与实验结果，包括：

- `docs/turboquant_research_notes.md`：TurboQuant 论文与相关研究调研。
- `reports/experiment_report.md`：Qwen2.5-VL-3B-Instruct 在 MMStar 上的 VLM 普通方法并发与量化实验。
- `reports/turboquant_comparison_report.md`：真实 TurboQuant KV cache 与普通 DynamicCache 的初始文本 LLM 对照实验。
- `reports/turboquant_batch_performance_report.md`：更严格的 batch 并发模式下，TurboQuant 与普通方法的性能提升效果验证。

总体结论如下：

1. 当前项目已完成真实 TurboQuant KV cache 实验，但实验对象是文本 LLM，而不是 Qwen2.5-VL 的 vLLM 服务。原因是当前已验证的 VLM 环境为 Python 3.10、vLLM 0.8.5.post1、transformers 4.51.3，不能无风险接入社区新版 TurboQuant vLLM 插件。
2. VLM 普通方法实验中，BF16 是当前最稳定且吞吐最高的 baseline；FP8 KV cache 稳定但在 MMStar 短输出任务中不优于 BF16；AWQ 在低并发稳定，但高并发存在 CUDA/ECC 相关不稳定风险。
3. TurboQuant 文本 LLM 实验中，KV cache 存储压缩效果明确：4096 token 上下文下，每请求 KV 存储从 46.04 MB 降到 14.03 MB，约 3.28 倍压缩，下降约 69.5%。
4. 在当前 HuggingFace 适配实现中，TurboQuant 没有带来端到端吞吐提升；其在线量化、反量化和 materialize 开销抵消了 KV cache 压缩收益。
5. 随上下文变长和 batch 并发提高，TurboQuant 与普通方法的吞吐差距明显缩小：4096 token、并发 8 时，TurboQuant 达到普通 DynamicCache 约 62.6% 的吞吐，但仍未反超。

所有正式推理实验均通过 Slurm 提交到计算节点执行，未在主节点直接运行 GPU 推理。相关作业包括 VLM 实验作业 `9126/9127/9128`，TurboQuant 初始文本作业 `9134`，以及 batch sweep 作业 `9137/9138/9139`。

## 2. 背景与研究基础

### 2.1 TurboQuant 核心思想

TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate 是 Google Research 提出的在线向量量化方法，目标包括 LLM KV-cache 压缩和向量检索场景。其关键特征是：

- 数据无关、在线处理：不依赖特定 calibration 数据，也不需要为具体模型训练码本。
- MSE 路径：对单位向量进行随机正交旋转，使坐标分布趋于可分析的集中分布，再使用近最优标量量化器。
- Inner-product 路径：先用 MSE 量化，再对残差使用 1-bit QJL，从理论上得到无偏内积估计。
- 典型目标区间为 3 到 4 bits/channel，用于降低 KV cache 显存压力。

相关工作包括 QJL、PolarQuant、KIVI、KVQuant，以及当前 vLLM 可直接使用的 FP8 KV cache。需要强调：vLLM FP8 KV cache 不是 TurboQuant，只是同样面向 KV cache 压缩瓶颈的可部署 baseline。

### 2.2 开源实现调研

本项目调研了多个 GitHub 开源实现：

| 实现 | 特点 | 当前环境适配性 |
| --- | --- | --- |
| `0xSero/turboquant` | 包含 TurboQuant 核心量化、Triton kernels 和实验性 vLLM 集成 | README 标注新版本栈，如 vLLM 0.18、Python 3.12，不适合直接污染现有环境 |
| `Alberto-Codes/turboquant-consumer` / `turboquant-vllm` | 提供 vLLM 插件和 `CompressedDynamicCache` | 要求 Python >= 3.12、vLLM >= 0.18、transformers >= 4.57 |
| `back2matching/turboquant` | HuggingFace 版 TurboQuant KV cache 和核心 `TurboQuantMSE` | Python >= 3.10，最接近当前环境，最终用于真实 TurboQuant 实验 |

由于现有 VLM 实验环境已经稳定跑通，不适合直接升级 vLLM/transformers，因此本项目选择 `back2matching/turboquant` 的核心量化实现，并编写 `CompatTurboQuantCache` 适配 transformers 4.51.3。

该实现采用 TurboQuant MSE 路径，即随机旋转 + Beta 分布最优标量量化。仓库文档也说明其默认不使用 QJL 残差路径，因为社区复现实验认为 QJL 噪声在 attention softmax 下可能被放大。

## 3. 实验环境与执行约束

### 3.1 统一约束

- 正式推理和并发实验全部通过 Slurm 提交到计算节点执行。
- 主节点仅用于脚本编辑、结果汇总和报告生成。
- 已下载模型和已配置环境优先复用，避免无必要破坏现有环境。

### 3.2 VLM 普通方法实验环境

- 模型：`Qwen2.5-VL-3B-Instruct`
- 推理框架：vLLM 0.8.5.post1
- 数据集：MMStar full validation set，1500 个图文问答样本，由 ModelScope `evalscope/MMStar` parquet 准备
- 输出上限：16 tokens
- 任务形式：要求模型输出单个 A/B/C/D 选项
- 配置：
  - BF16
  - BF16 + FP8 KV cache
  - AWQ
- 正式作业：
  - BF16：`9126`
  - BF16 + FP8 KV：`9127`
  - AWQ：`9128`

### 3.3 TurboQuant 文本 LLM 实验环境

- 模型：`Qwen2.5-0.5B-Instruct`
- 普通方法：transformers `DynamicCache`
- TurboQuant 方法：`CompatTurboQuantCache` + `back2matching/turboquant` 的 `TurboQuantMSE`
- 输出上限：16 tokens
- residual window：128 tokens
- 初始并发实验：
  - prompt 上下文约 768 tokens
  - 并发 1、2、4
  - 每档 8 请求
  - Slurm 作业 `9134`，运行于 `compute2`
- batch sweep 实验：
  - prompt 上下文约 1024、2048、4096 tokens
  - batch 并发 1、2、4、8
  - 每档 8 请求
  - Slurm 作业 `9137/9138/9139`，运行于 `compute1`

## 4. VLM 普通方法实验结果

### 4.1 最佳零错误点

| 配置 | 最佳零错误并发 | requests/s | P95 延迟(s) | 准确率 |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 32 | 25.5822 | 2.2261 | 0.5453 |
| BF16 + FP8 KV | 32 | 14.5211 | 3.8216 | 0.5473 |
| AWQ | 4 | 12.6400 | 0.6249 | 0.5347 |

### 4.2 全量正式摘要

| 配置 | 并发 | requests/s | P95 延迟(s) | P95 TTFT(s) | 准确率 | 错误率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AWQ | 1 | 5.3212 | 0.2555 | 0.2135 | 0.5347 | 0.0000 |
| AWQ | 2 | 8.3090 | 0.3484 | 0.2849 | 0.5347 | 0.0000 |
| AWQ | 4 | 12.6400 | 0.6249 | 0.4840 | 0.5347 | 0.0000 |
| BF16 | 1 | 6.4846 | 0.2420 | 0.2286 | 0.5453 | 0.0000 |
| BF16 | 2 | 9.2016 | 0.3311 | 0.3010 | 0.5460 | 0.0000 |
| BF16 | 4 | 13.9090 | 0.6100 | 0.4991 | 0.5480 | 0.0000 |
| BF16 | 8 | 18.5592 | 1.0177 | 0.9217 | 0.5467 | 0.0000 |
| BF16 | 16 | 23.5152 | 1.4679 | 1.3003 | 0.5480 | 0.0000 |
| BF16 | 32 | 25.5822 | 2.2261 | 1.9962 | 0.5453 | 0.0000 |
| BF16 + FP8 KV | 1 | 6.4780 | 0.2095 | 0.1947 | 0.5473 | 0.0000 |
| BF16 + FP8 KV | 2 | 10.2878 | 0.3455 | 0.3283 | 0.5487 | 0.0000 |
| BF16 + FP8 KV | 4 | 12.4116 | 0.6473 | 0.6306 | 0.5520 | 0.0000 |
| BF16 + FP8 KV | 8 | 13.8126 | 1.2441 | 1.2181 | 0.5493 | 0.0000 |
| BF16 + FP8 KV | 16 | 14.0384 | 2.2272 | 2.0315 | 0.5487 | 0.0000 |
| BF16 + FP8 KV | 32 | 14.5211 | 3.8216 | 3.0704 | 0.5473 | 0.0000 |

### 4.3 VLM 结果解读

BF16 是当前 Qwen2.5-VL 短输出 MMStar workload 下最强的稳定 serving baseline。它在并发 32 时仍保持零错误，并取得最高吞吐。

BF16 + FP8 KV cache 稳定，但在该任务上没有体现吞吐优势。原因是 MMStar 实验为短输出、多图文输入场景，KV cache 还不是主要瓶颈，FP8 KV 带来的额外开销未被显存节省抵消。

AWQ 在正式全量实验的并发 1、2、4 上稳定，但此前 500 样本高并发探测发现并发 8 及以上不稳定：c=8 error_rate=0.53，c=16 和 c=32 error_rate=1.00。作业 `9120` 的 server log 记录了 vLLM EngineCore death，并伴随 `CUDA error: uncorrectable ECC error encountered`。

## 5. 初始 TurboQuant 文本并发实验

### 5.1 实验设计

初始实验用于验证真实 TurboQuant KV cache 是否能跑通，并与普通 `DynamicCache` 在相同文本 LLM、相同上下文和输出设置下对比。

- 模型：`Qwen2.5-0.5B-Instruct`
- 上下文：约 768 prompt tokens
- 输出上限：16 tokens
- 并发：1、2、4
- 每档请求数：8
- Slurm 作业：`9134`
- 运行节点：`compute2`

### 5.2 结果

| 方法 | 并发 | 请求数 | 错误率 | requests/s | output tokens/s | P95 延迟(s) | P95 TTFT(s) | 平均 KV 存储(MB) | FP16 等价 KV(MB) | KV 压缩率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通 DynamicCache | 1 | 8 | 0.0000 | 2.0795 | 33.2726 | 0.9099 | 0.4597 | 8.6602 | 8.6602 | 1.0000 |
| 普通 DynamicCache | 2 | 8 | 0.0000 | 2.2794 | 36.4705 | 0.8019 | 0.0259 | 8.6602 | 8.6602 | 1.0000 |
| 普通 DynamicCache | 4 | 8 | 0.0000 | 2.2820 | 36.5120 | 1.6674 | 0.0271 | 8.6602 | 8.6602 | 1.0000 |
| TurboQuant KV cache | 1 | 8 | 0.0000 | 0.5641 | 9.0252 | 1.9974 | 1.2543 | 3.5138 | 8.6602 | 2.4646 |
| TurboQuant KV cache | 2 | 8 | 0.0000 | 0.5386 | 8.6180 | 3.6913 | 1.0370 | 3.5138 | 8.6602 | 2.4646 |
| TurboQuant KV cache | 4 | 8 | 0.0000 | 0.5737 | 9.1790 | 7.0916 | 1.0382 | 3.5138 | 8.6602 | 2.4646 |

### 5.3 初始结论

TurboQuant 功能正确，两种方法全部 0 错误。TurboQuant 将平均 KV 存储从 8.66 MB 降至 3.51 MB，约 2.46 倍压缩。但在该实现路径下，吞吐明显低于普通 DynamicCache，P95 延迟也更高，说明在线量化、反量化和 Python/HuggingFace cache 适配开销占主导。

## 6. Batch 并发性能提升验证

### 6.1 为什么需要 batch 模式

初始文本实验虽然能验证功能，但并发请求通过线程模拟，并且模型执行受到锁保护，更接近排队压力，不足以反映真实 serving 中的批处理并发。

因此补充实验将脚本升级为 `batch` 执行模式：并发数 `C` 表示同一轮前向中同时处理 `C` 个请求，更接近推理服务中的 continuous batching 或批量并发行为。

### 6.2 吞吐对比

| 上下文 | 并发 | 普通方法 req/s | TurboQuant req/s | TurboQuant / 普通方法 | 结论 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1024 | 1 | 2.0167 | 0.5355 | 0.266x | 明显低于普通方法 |
| 1024 | 2 | 4.1692 | 1.0163 | 0.244x | 明显低于普通方法 |
| 1024 | 4 | 7.6159 | 2.3214 | 0.305x | 明显低于普通方法 |
| 1024 | 8 | 12.7470 | 3.8561 | 0.303x | 明显低于普通方法 |
| 2048 | 1 | 1.8811 | 0.5811 | 0.309x | 明显低于普通方法 |
| 2048 | 2 | 3.3652 | 1.0390 | 0.309x | 明显低于普通方法 |
| 2048 | 4 | 5.4652 | 1.8331 | 0.335x | 明显低于普通方法 |
| 2048 | 8 | 7.9721 | 3.4446 | 0.432x | 差距开始缩小 |
| 4096 | 1 | 1.3693 | 0.4799 | 0.350x | 仍低于普通方法 |
| 4096 | 2 | 2.1587 | 0.6871 | 0.318x | 仍低于普通方法 |
| 4096 | 4 | 2.8220 | 1.4227 | 0.504x | 差距明显缩小 |
| 4096 | 8 | 3.2825 | 2.0556 | 0.626x | 最接近普通方法 |

### 6.3 延迟对比

| 上下文 | 并发 | 普通方法 P95(s) | TurboQuant P95(s) | TurboQuant / 普通方法 |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 8 | 0.6249 | 2.0717 | 3.32x |
| 2048 | 8 | 1.0004 | 2.3190 | 2.32x |
| 4096 | 8 | 2.4306 | 3.8875 | 1.60x |

TurboQuant 的 P95 延迟仍高于普通方法，但相对劣势随上下文变长而减小。4096 token、并发 8 时，延迟比从 1024 token 的 3.32x 降至 1.60x。

### 6.4 KV cache 存储压缩

| 上下文 | 普通 KV/请求(MB) | TurboQuant KV/请求(MB) | 压缩率 | KV 存储下降 |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 11.4961 | 4.3114 | 2.666x | 62.5% |
| 2048 | 23.0977 | 7.5743 | 3.049x | 67.2% |
| 4096 | 46.0430 | 14.0277 | 3.282x | 69.5% |

TurboQuant 的存储压缩收益非常明确，且随着上下文增长而增强。这符合 residual window 固定时，长上下文中可压缩历史 KV 占比更高的预期。

### 6.5 GPU 峰值显存

| 上下文 | 并发 | 普通方法峰值(MB) | TurboQuant 峰值(MB) | 峰值下降 |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 8 | 3308.55 | 3257.32 | 1.5% |
| 2048 | 8 | 5724.11 | 5610.05 | 2.0% |
| 4096 | 8 | 14965.07 | 14729.70 | 1.6% |

尽管持久 KV 存储下降显著，但 GPU 峰值显存下降有限。原因是当前适配实现每次 attention 仍需要将压缩历史 KV materialize/dequantize 为常规张量参与计算，峰值内存中仍包含临时反量化张量和模型计算中间结果。

## 7. 综合分析

### 7.1 是否观察到 TurboQuant 的性能提升

在当前实验范围内，没有观察到 TurboQuant 相比普通 DynamicCache 的端到端吞吐提升：

- 吞吐：TurboQuant 在所有测试点均低于普通 DynamicCache。
- 延迟：TurboQuant 在所有测试点 P95 延迟更高。
- 稳定性：两种方法在所有测试点均 0 错误。
- KV 存储：TurboQuant 显著降低 KV cache 存储，最高约 3.28 倍压缩。
- 趋势：上下文越长、并发越高，TurboQuant 与普通方法的差距越小。

因此，当前结论不是“TurboQuant 已经在本环境中带来端到端性能提升”，而是“TurboQuant 的 KV cache 压缩有效，但当前 HuggingFace 适配实现的在线量化/反量化开销仍大于压缩收益”。

### 7.2 为什么压缩没有转化为吞吐提升

主要原因有三点：

1. 当前实现不是 fused attention backend。每次 attention 计算前，需要把压缩 KV 反量化并拼接为普通张量。
2. Python/HuggingFace cache 适配路径开销较高，难以达到 vLLM paged attention 或 fused Triton kernel 的效率。
3. 测试模型为 0.5B，模型本身较小，普通 DynamicCache 尚未被 KV cache 显存瓶颈强烈限制；TurboQuant 的压缩收益尚未覆盖额外计算开销。

### 7.3 哪些趋势对 TurboQuant 有利

实验也显示了一些对 TurboQuant 有利的趋势：

- KV 压缩率随上下文长度增长：1024 token 为 2.67x，4096 token 达到 3.28x。
- 吞吐保持率随上下文长度增长：最高并发 8 下，从 1024 token 的 30.3% 提升到 4096 token 的 62.6%。
- P95 延迟相对劣势随上下文长度增长而缩小：最高并发 8 下，从 3.32x 降至 1.60x。

这些趋势说明，如果进入更长上下文、更大模型、更高并发或普通 KV cache 接近 OOM 的场景，TurboQuant 有可能更接近或超过普通方法。但要验证这一点，需要 fused attention / fused dequant kernel，而不是当前的 HuggingFace materialize 路径。

## 8. 最终结论

本项目最终形成两条互补结论：

第一，针对当前 Qwen2.5-VL-3B-Instruct + vLLM 0.8.5.post1 的 VLM serving 任务，最可靠的普通方法仍是 BF16。FP8 KV cache 稳定但没有在短输出 MMStar workload 中超越 BF16；AWQ 低并发可用但高并发存在稳定性风险。

第二，针对真实 TurboQuant KV cache，开源 HuggingFace 适配实现证明了 TurboQuant 的 KV 存储压缩有效，且随上下文变长更明显；但当前实现没有端到端性能提升，主要受在线量化、反量化和缺少 fused attention backend 限制。

因此，本报告建议将当前结果定位为：

- VLM 部分：当前集群和 vLLM 环境下可部署 baseline 的完整并发评估。
- TurboQuant 部分：真实 TurboQuant KV cache 的功能、压缩收益和当前工程开销评估。
- 后续研究方向：单独新建新版 vLLM/TurboQuant 环境，测试 fused backend 在长上下文和高并发下的真实性能上限。

## 9. 后续建议

如果继续推进 TurboQuant 的性能验证，建议避免污染当前已验证的 VLM 环境，单独创建实验环境并测试以下方向：

1. 使用支持 TurboQuant attention backend 的新版 vLLM 插件或原生 PR 分支。
2. 将上下文扩展到 8192、16384、32768 tokens。
3. 使用至少 3B/7B 级别 decoder-only LLM，增大 KV cache 占总显存的比例。
4. 提高 batch 并发，直到普通 DynamicCache 出现 OOM、吞吐饱和或明显抖动。
5. 使用 fused decode attention，避免每步 materialize 完整历史 KV。
6. 若未来要做 VLM TurboQuant，对 Qwen2.5-VL 单独建立新版 vLLM 环境，不建议在当前稳定环境中直接升级依赖。

## 10. 产物与数据路径

### 报告与笔记

- 完整合并报告：`reports/final_complete_technical_report.md`
- VLM 普通方法报告：`reports/experiment_report.md`
- TurboQuant 初始对比报告：`reports/turboquant_comparison_report.md`
- TurboQuant batch sweep 报告：`reports/turboquant_batch_performance_report.md`
- TurboQuant 研究笔记：`docs/turboquant_research_notes.md`

### 脚本

- VLM 异步客户端：`scripts/bench_vlm_async.py`
- VLM Slurm 脚本：`sbatch/run_vlm_bench.sbatch`
- TurboQuant 文本并发脚本：`scripts/bench_text_turboquant_concurrency.py`
- TurboQuant 文本 Slurm 脚本：`sbatch/run_text_turboquant_bench.sbatch`
- GPU 监控脚本：`scripts/gpu_monitor.sh`

### 结果

- VLM 全量 CSV：`reports/formal_summary_1500.csv`
- AWQ 高并发探测：`reports/awq_instability_probe_500.csv`
- TurboQuant 初始正式结果：`results/text_turboquant_formal_c124/text_turboquant_summary.json`
- TurboQuant batch ctx=1024：`results/text_turboquant_batch_ctx1024/text_turboquant_summary.json`
- TurboQuant batch ctx=2048：`results/text_turboquant_batch_ctx2048/text_turboquant_summary.json`
- TurboQuant batch ctx=4096：`results/text_turboquant_batch_ctx4096/text_turboquant_summary.json`

### GPU 日志

- VLM BF16：`logs/9126_bf16_gpu.csv`
- VLM BF16 + FP8 KV：`logs/9127_bf16_fp8kv_gpu.csv`
- VLM AWQ：`logs/9128_awq_gpu.csv`
- TurboQuant 初始文本实验：`logs/9134_text_turboquant_gpu.csv`
- TurboQuant batch ctx=1024：`logs/9137_text_turboquant_gpu.csv`
- TurboQuant batch ctx=2048：`logs/9138_text_turboquant_gpu.csv`
- TurboQuant batch ctx=4096：`logs/9139_text_turboquant_gpu.csv`

### 可视化 Canvas

- VLM 结果 Canvas：`vlm-concurrency-results.canvas.tsx`
- TurboQuant 初始对比 Canvas：`turboquant-comparison-results.canvas.tsx`
- TurboQuant batch sweep Canvas：`turboquant-batch-sweep-results.canvas.tsx`
