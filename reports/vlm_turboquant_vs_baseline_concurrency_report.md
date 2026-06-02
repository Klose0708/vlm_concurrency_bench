# VLM 上 TurboQuant 与普通方法并发对比实验中文技术报告

## 1. 摘要

本报告将原有“VLM 普通方法并发实验”升级为“VLM 上 TurboQuant KV cache 与普通 DynamicCache 的并发对比实验”。实验使用公开专业 VLM 评测数据集 MMStar，在 Slurm 计算节点上运行 Qwen2.5-VL-3B-Instruct，并在同一 HuggingFace Transformers VLM 推理路径中对比：

- 普通方法：Transformers `DynamicCache`。
- TurboQuant 方法：`CompatTurboQuantCache` + `TurboQuantMSE`，对历史 KV cache 进行在线向量量化压缩，保留 128 token residual window。

正式 VLM TurboQuant 对比作业为 `9189`，运行节点为 `compute6`。主节点仅用于脚本编辑、作业提交和结果汇总，没有直接运行 GPU 推理实验。

核心结论如下：

1. TurboQuant 已真实接入 Qwen2.5-VL VLM 推理路径，而不是仅在文本 LLM 上验证。
2. 在 MMStar 前 128 个样本、并发 1/2/4/8、每请求最多 16 个输出 token 的实验中，两种方法均为 0 错误，答案解析成功率均为 100%。
3. TurboQuant 显著降低每请求 KV cache 存储，压缩后 KV 存储相对普通方法下降约 45.9% 到 55.0%，且并发越高，历史 KV 占比越高，压缩效果越明显。
4. 当前 HuggingFace 适配实现下，TurboQuant 端到端吞吐仍低于普通 DynamicCache，但差距随并发增加显著缩小：并发 1 时 TurboQuant 仅为普通方法 20.3% 吞吐，并发 8 时提升到 62.2%。
5. TurboQuant P95 延迟高于普通方法，但相对劣势随并发升高明显收敛：并发 1 为 5.66x，并发 8 降至 1.20x。
6. 本实验验证的是开源 HuggingFace cache 适配路径的工程可行性与代价，不代表 fused vLLM/Triton attention backend 的理论上限。

## 2. 公开数据集与联网调研依据

### 2.1 MMStar 数据集

联网检索确认，MMStar 是面向大视觉语言模型的专业公开评测数据集。其官方 HuggingFace 数据集为 `Lin-Chen/MMStar`，官方仓库为 `MMStar-Benchmark/MMStar`，论文题为 *Are We on the Right Way for Evaluating Large Vision-Language Models?*，NeurIPS 2024。公开资料说明 MMStar 包含 1500 个高质量人工筛选样本，强调视觉依赖、低数据泄漏，并覆盖 6 个核心能力和 18 个细粒度评测轴。

本项目复用现有 `scripts/prepare_mmstar.py` 将 MMStar 转为图文问答 JSONL。正式 VLM TurboQuant 对比实验使用 `data/mmstar/mmstar_requests.jsonl` 的前 128 个样本。

参考来源：

- HuggingFace 数据集：https://huggingface.co/datasets/Lin-Chen/MMStar
- 官方 GitHub：https://github.com/MMStar-Benchmark/MMStar
- 论文页：https://huggingface.co/papers/2403.20330

### 2.2 TurboQuant 公开实现

联网检索确认，TurboQuant 是面向 LLM KV cache 压缩的在线、数据无关向量量化方法。公开实现和插件生态包括 `Alberto-Codes/turboquant-vllm`、`0xSero/turboquant`、`back2matching/turboquant` 等。现有 vLLM 插件通常要求较新的 Python/vLLM/Transformers 版本；当前项目已验证 VLM 环境为 Python 3.10、vLLM 0.8.5.post1、Transformers 4.51.3。为避免破坏现有 VLM 服务环境，本实验选择已在项目中适配成功的 `back2matching/turboquant` 核心 `TurboQuantMSE`，并将其接入 Qwen2.5-VL HuggingFace 推理路径。

参考来源：

- TurboQuant vLLM 插件：https://github.com/Alberto-Codes/turboquant-consumer
- TurboQuant 相关公开实现：https://github.com/vivekvar-dl/turboquant
- SGLang TurboQuant 议题：https://github.com/sgl-project/sglang/issues/21618

## 3. 实验环境与约束

### 3.1 运行约束

- 所有正式 GPU 推理均通过 Slurm 提交到计算节点。
- 本轮正式 VLM TurboQuant 对比作业：`9189`。
- 运行节点：`compute6`。
- 主节点：仅用于编辑、提交、结果读取和报告生成。
- smoke 作业：`9185` 验证最小样本，`9188` 验证 batch 左填充策略。

### 3.2 模型与软件环境

- 模型：`Qwen2.5-VL-3B-Instruct`
- 模型路径：`/data/private/$USER/workspace/models/Qwen2.5-VL-3B-Instruct`
- 推理框架：HuggingFace Transformers 4.51.3
- dtype：BF16
- attention implementation：`eager`
- GPU：NVIDIA H20-3e
- Slurm 脚本：`sbatch/run_vlm_turboquant_bench.sbatch`
- 实验脚本：`scripts/bench_vlm_turboquant_concurrency.py`
- GPU 日志：`logs/9189_vlm_turboquant_gpu.csv`

### 3.3 方法定义

普通方法使用 Transformers `DynamicCache`，KV cache 以未压缩张量形式保存。

TurboQuant 方法使用项目中的 `CompatTurboQuantCache`：

- 量化器：`TurboQuantMSE`
- bits：4
- residual window：128 tokens
- 压缩对象：历史 key/value cache
- 统计指标：压缩 KV 存储、FP16/BF16 等价 KV 存储、压缩率、吞吐、延迟、准确率、解析成功率、错误率

### 3.4 实验设计

正式实验采用 batch 并发模式。并发数 `C` 表示每一轮模型生成中同时处理 `C` 个 MMStar 请求，更接近 VLM serving 中的批处理并发。

- 数据集：MMStar 前 128 个样本
- 并发档位：1、2、4、8
- 每档请求数：128
- 输出上限：16 tokens
- 解码策略：确定性生成，`do_sample=False`
- 任务形式：图文多选题，仅要求输出 A/B/C/D 单个选项

实验过程中发现 decoder-only VLM 批生成需要显式左填充，否则高 batch 档位会造成输出截取和解析偏差。因此最终正式作业在 `AutoProcessor` 加载后设置：

```python
processor.tokenizer.padding_side = "left"
```

## 4. 正式实验结果

### 4.1 汇总表

| 方法 | 并发 | 请求数 | 错误率 | 解析成功率 | 准确率 | requests/s | output tokens/s | P95 延迟(s) | 平均 KV 存储(MB/请求) | 等价未压缩 KV(MB/请求) | KV 压缩率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通 DynamicCache | 1 | 128 | 0.0000 | 1.0000 | 0.5234 | 4.1296 | 8.2592 | 0.2275 | 11.9850 | 11.9850 | 1.0000 |
| 普通 DynamicCache | 2 | 128 | 0.0000 | 1.0000 | 0.5000 | 6.0513 | 12.1026 | 0.3679 | 13.5060 | 13.5060 | 1.0000 |
| 普通 DynamicCache | 4 | 128 | 0.0000 | 1.0000 | 0.5156 | 5.9109 | 11.8218 | 1.3090 | 15.1578 | 15.1578 | 1.0000 |
| 普通 DynamicCache | 8 | 128 | 0.0000 | 1.0000 | 0.5234 | 4.9576 | 9.9151 | 5.2414 | 17.9187 | 17.9187 | 1.0000 |
| TurboQuant KV cache | 1 | 128 | 0.0000 | 1.0000 | 0.5234 | 0.8364 | 1.6729 | 1.2871 | 6.4882 | 11.9850 | 1.8051 |
| TurboQuant KV cache | 2 | 128 | 0.0000 | 1.0000 | 0.5312 | 1.5265 | 3.0530 | 1.4726 | 6.8922 | 13.5060 | 1.9017 |
| TurboQuant KV cache | 4 | 128 | 0.0000 | 1.0000 | 0.5078 | 2.4259 | 4.8517 | 1.9101 | 7.3310 | 15.1578 | 1.9768 |
| TurboQuant KV cache | 8 | 128 | 0.0000 | 1.0000 | 0.5312 | 3.0835 | 6.1669 | 6.2688 | 8.0643 | 17.9187 | 2.0767 |

### 4.2 吞吐对比

| 并发 | 普通方法 req/s | TurboQuant req/s | TurboQuant / 普通方法 | 普通方法相对 TurboQuant |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.1296 | 0.8364 | 20.3% | 4.94x |
| 2 | 6.0513 | 1.5265 | 25.2% | 3.96x |
| 4 | 5.9109 | 2.4259 | 41.0% | 2.44x |
| 8 | 4.9576 | 3.0835 | 62.2% | 1.61x |

当前实现下，TurboQuant 没有超过普通 DynamicCache；但随着并发从 1 增至 8，TurboQuant 的相对吞吐从 20.3% 提升到 62.2%，说明更高并发下 KV 存储压缩开始抵消一部分在线量化/反量化开销。

### 4.3 延迟对比

| 并发 | 普通方法 P95(s) | TurboQuant P95(s) | TurboQuant / 普通方法 |
| ---: | ---: | ---: | ---: |
| 1 | 0.2275 | 1.2871 | 5.66x |
| 2 | 0.3679 | 1.4726 | 4.00x |
| 4 | 1.3090 | 1.9101 | 1.46x |
| 8 | 5.2414 | 6.2688 | 1.20x |

TurboQuant 的 P95 延迟始终更高，但相对差距随并发升高快速缩小。并发 8 时，TurboQuant P95 延迟仅比普通方法高约 19.6%。

### 4.4 KV cache 压缩收益

| 并发 | 普通 KV(MB/请求) | TurboQuant KV(MB/请求) | 压缩率 | KV 存储下降 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 11.9850 | 6.4882 | 1.8051x | 45.9% |
| 2 | 13.5060 | 6.8922 | 1.9017x | 49.0% |
| 4 | 15.1578 | 7.3310 | 1.9768x | 51.6% |
| 8 | 17.9187 | 8.0643 | 2.0767x | 55.0% |

TurboQuant 的 KV 存储下降非常稳定，并且随 batch 并发增加而提高。原因是 residual window 固定为 128 tokens，而 batch 中更长的有效序列和更多历史 KV 使可压缩部分占比增加。

### 4.5 准确率与稳定性

两种方法所有正式档位均 0 错误、解析成功率 100%。TurboQuant 与普通方法准确率在 128 样本子集上差异很小：

- 并发 1：两者均为 0.5234。
- 并发 2：TurboQuant 0.5312，普通方法 0.5000。
- 并发 4：TurboQuant 0.5078，普通方法 0.5156。
- 并发 8：TurboQuant 0.5312，普通方法 0.5234。

这些差异在 128 样本规模下不能解释为显著精度提升或下降，更合理的结论是：在本短输出 MMStar 子集上，TurboQuant 未观察到明显质量劣化。

## 5. 与原 VLM 普通方法实验的关系

原 VLM 普通方法实验使用 vLLM OpenAI 服务接口，在 MMStar 全量 1500 样本上完成 BF16、BF16 + FP8 KV、AWQ 的 serving 并发测试。其最佳零错误结果为：

| 配置 | 最佳零错误并发 | requests/s | P95 延迟(s) | 准确率 |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 32 | 25.5822 | 2.2261 | 0.5453 |
| BF16 + FP8 KV | 32 | 14.5211 | 3.8216 | 0.5473 |
| AWQ | 4 | 12.6400 | 0.6249 | 0.5347 |

本轮新增实验不是替代 vLLM serving baseline，而是在 VLM 模型上补齐真实 TurboQuant KV cache 对照。由于当前 TurboQuant 开源插件要求更高版本 vLLM/Transformers，直接升级会污染已验证环境，因此采用 HuggingFace VLM 路径完成同模型、同数据、同 cache API 下的严格对照。

因此，两个实验的定位不同：

- vLLM 普通方法实验：回答当前环境中可部署 VLM serving baseline 谁更强。
- VLM TurboQuant 对照实验：回答 TurboQuant KV cache 在 Qwen2.5-VL 上是否可用、压缩多少、端到端并发代价如何。

## 6. 技术分析

### 6.1 为什么 TurboQuant 压缩明显但吞吐仍低

本实验使用 HuggingFace cache 适配路径，而不是 fused attention backend。当前 `CompatTurboQuantCache` 在每次 attention 计算前需要将压缩历史 KV 反量化并 materialize 成普通张量，再与 residual window 拼接。这带来三类开销：

1. 在线量化：新进入历史窗口的 KV 需要旋转、量化、打包。
2. 在线反量化：每次 attention 读取历史 KV 时需要解包和反量化。
3. Python/cache 适配：HuggingFace cache API 路径没有 vLLM PagedAttention 的融合调度能力。

因此，在短输出 MMStar workload 下，普通 DynamicCache 仍更快。TurboQuant 的收益主要体现在 KV 存储下降，而不是当前实现的端到端吞吐提升。

### 6.2 为什么高并发时差距缩小

并发从 1 到 8 时，TurboQuant/普通方法吞吐比从 20.3% 提升到 62.2%，P95 延迟比从 5.66x 降到 1.20x。这说明当 batch 更大时，普通方法的 KV 存储和 attention 读写压力上升，而 TurboQuant 的压缩收益开始更明显。

这一趋势与此前文本 batch sweep 的结论一致：上下文越长、batch 越高，TurboQuant 与普通 DynamicCache 的差距越小。但在当前非融合实现下，它仍未反超。

### 6.3 GPU 峰值显存解读

正式结果中普通方法和 TurboQuant 的 `peak_allocated_mb_max` 在相同并发档位上相同或非常接近。这不代表 KV cache 没有压缩，而是因为：

- 模型权重、视觉编码器、图像 token 中间激活占用显存较大。
- 当前实现会在计算前 materialize 反量化 KV，峰值显存包含临时反量化张量。
- PyTorch CUDA allocator 的峰值统计反映进程历史峰值，不等同于持久 KV cache 占用。

因此，本报告以脚本直接统计的 cache 存储量作为 KV 压缩指标，以吞吐和 P95 延迟作为端到端性能指标。

## 7. 工程产物

新增或更新的关键文件如下：

- VLM TurboQuant 并发脚本：`scripts/bench_vlm_turboquant_concurrency.py`
- VLM TurboQuant Slurm 脚本：`sbatch/run_vlm_turboquant_bench.sbatch`
- 正式结果 JSON：`results/vlm_turboquant_formal_mmstar128_leftpad_c1248/vlm_turboquant_summary.json`
- 正式结果 CSV：`results/vlm_turboquant_formal_mmstar128_leftpad_c1248/vlm_turboquant_summary.csv`
- 请求级明细：`results/vlm_turboquant_formal_mmstar128_leftpad_c1248/vlm_turboquant_requests.jsonl`
- 正式作业日志：`logs/9189_vlm_turboquant.out`
- 正式错误日志：`logs/9189_vlm_turboquant.err`
- GPU 监控日志：`logs/9189_vlm_turboquant_gpu.csv`
- 本报告：`reports/vlm_turboquant_vs_baseline_concurrency_report.md`

## 8. 最终结论

本轮任务已将 VLM 普通方法实验升级为 VLM 上 TurboQuant 与普通方法的并发对比实验。实验严格运行在 Slurm 计算节点上，并使用联网检索确认的公开专业 VLM 数据集 MMStar。

结论可以概括为：

- 从功能上看，TurboQuant 已可用于 Qwen2.5-VL 的真实图文推理 KV cache。
- 从存储上看，TurboQuant 在 VLM 并发实验中稳定压缩 KV cache，存储下降约 45.9% 到 55.0%。
- 从质量上看，在 MMStar 128 样本子集上未观察到明显准确率损失。
- 从性能上看，当前 HuggingFace 非融合实现仍慢于普通 DynamicCache，但高并发下差距明显缩小。
- 从后续研究看，若要验证 TurboQuant 的真实性能上限，应在隔离环境中测试新版 vLLM/TurboQuant fused attention backend，并扩大到更长上下文、更高并发和更大样本规模。
