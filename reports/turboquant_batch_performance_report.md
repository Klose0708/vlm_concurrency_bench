# TurboQuant 与普通方法并发性能提升效果补充实验报告

## 1. 实验目的

本轮补充实验专门回答一个问题：TurboQuant 相比普通 KV cache，在并发推理中是否带来性能提升。

上一轮实验采用逐请求串行执行，能够验证 TurboQuant KV cache 的功能和压缩率，但并发含义更接近排队压力。本轮将脚本升级为 `batch` 执行模式：并发数 `C` 表示同一轮前向中同时处理 `C` 个请求，更接近推理服务中的批处理并发。所有推理实验均通过 Slurm 提交到计算节点执行，没有在主节点直接运行 GPU 推理。

## 2. 实验环境与作业

- 模型：`Qwen2.5-0.5B-Instruct`
- 普通方法：transformers `DynamicCache`
- TurboQuant 方法：基于开源仓库 `back2matching/turboquant` 的 `TurboQuantMSE`，适配为 transformers 4.51.3 可用的 `CompatTurboQuantCache`
- 执行模式：`batch`
- 输出上限：16 tokens
- 每档请求数：8
- 并发档位：1、2、4、8
- 上下文长度：约 1024、2048、4096 prompt tokens
- Slurm 作业：
  - `9137`：ctx=1024，运行于计算节点 `compute1`
  - `9138`：ctx=2048，运行于计算节点 `compute1`
  - `9139`：ctx=4096，运行于计算节点 `compute1`

相关结果目录：

- `results/text_turboquant_batch_ctx1024/`
- `results/text_turboquant_batch_ctx2048/`
- `results/text_turboquant_batch_ctx4096/`

## 3. 核心结果

### 3.1 吞吐：TurboQuant 未超过普通方法，但长上下文下差距缩小

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

结论：在本开源 HuggingFace 适配实现中，TurboQuant 没有在测试范围内实现端到端吞吐提升。随着上下文从 1024 增加到 4096，TurboQuant 在最高并发下的吞吐保持率从约 30% 提高到约 63%，说明 KV cache 压缩收益在长上下文下开始抵消部分量化开销，但尚未反超普通方法。

### 3.2 延迟：TurboQuant P95 延迟更高，但长上下文下相对劣势变小

| 上下文 | 并发 | 普通方法 P95(s) | TurboQuant P95(s) | TurboQuant / 普通方法 |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 8 | 0.6249 | 2.0717 | 3.32x |
| 2048 | 8 | 1.0004 | 2.3190 | 2.32x |
| 4096 | 8 | 2.4306 | 3.8875 | 1.60x |

结论：TurboQuant 的 P95 延迟仍高于普通方法，主要来自 prefill 阶段的在线量化和每步 attention 前的反量化/materialize。随着上下文变长，普通方法的 KV 访问成本上升，TurboQuant 的相对延迟劣势有所缩小。

### 3.3 KV cache 存储：TurboQuant 压缩收益明确，并随上下文变长增强

| 上下文 | 普通 KV/请求(MB) | TurboQuant KV/请求(MB) | 压缩率 | KV 存储下降 |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 11.4961 | 4.3114 | 2.666x | 62.5% |
| 2048 | 23.0977 | 7.5743 | 3.049x | 67.2% |
| 4096 | 46.0430 | 14.0277 | 3.282x | 69.5% |

结论：TurboQuant 的主要正收益体现在 KV cache 存储压缩上。上下文越长，固定 residual window 占比越低，压缩率越接近低比特量化的目标区间。

### 3.4 GPU 峰值显存：下降有限

| 上下文 | 并发 | 普通方法峰值(MB) | TurboQuant峰值(MB) | 峰值下降 |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 8 | 3308.55 | 3257.32 | 1.5% |
| 2048 | 8 | 5724.11 | 5610.05 | 2.0% |
| 4096 | 8 | 14965.07 | 14729.70 | 1.6% |

结论：虽然 TurboQuant 的持久 KV 存储显著降低，但当前适配实现每次 attention 仍会将压缩历史 materialize/dequantize 成常规张量参与计算，因此 GPU 峰值显存下降有限。这一点解释了为什么压缩收益没有充分转化为吞吐提升。

## 4. 性能提升效果判断

本轮更严格的 batch 并发实验没有观察到 TurboQuant 相比普通方法的端到端性能提升：

- 吞吐：TurboQuant 在所有测试点均低于普通 DynamicCache。
- 延迟：TurboQuant 在所有测试点 P95 延迟更高。
- 稳定性：两种方法在所有测试点均 0 错误。
- 存储：TurboQuant 明确降低 KV cache 存储，4096 token 时每请求 KV 存储下降约 69.5%。
- 趋势：上下文越长、并发越高，TurboQuant 与普通方法的吞吐差距越小；在 4096 tokens、并发 8 时，TurboQuant 达到普通方法约 62.6% 的吞吐。

因此，当前结论不是“TurboQuant 已经带来端到端性能提升”，而是：

1. TurboQuant 的 KV cache 压缩效果真实存在，且随上下文增长更明显。
2. 当前 HuggingFace 适配实现没有 fused attention / fused dequant kernel，在线量化与反量化开销抵消了压缩收益。
3. 在更长上下文或更高显存压力场景下，TurboQuant 的相对表现改善，但本轮 0.5B 模型、4096 token、并发 8 尚未达到反超点。

## 5. 与 VLM 普通方法结果的关系

此前 Qwen2.5-VL-3B-Instruct 的 VLM 并发实验仍然是普通方法对比，包含 BF16、BF16 + FP8 KV、AWQ，不是 TurboQuant。由于当前 vLLM 0.8.5.post1 环境无法无风险接入新版 TurboQuant vLLM 插件，本轮真实 TurboQuant 实验选择文本 LLM 路径完成。

这两组结果应这样解读：

- VLM 实验回答：当前项目已有 serving 栈中，普通可部署方法的吞吐、延迟、准确率表现如何。
- 文本 TurboQuant 实验回答：真实 TurboQuant KV cache 在并发推理下是否产生性能提升，以及压缩收益是否能抵消实现开销。

## 6. 后续建议

如果目标是验证 TurboQuant 的论文级性能潜力，而不是 HuggingFace 适配实现的工程现状，下一步应单独新建环境，使用支持 TurboQuant attention backend 的新版 vLLM 插件或原生 PR 分支，重点测试：

- 更长上下文：8192、16384、32768 tokens。
- 更大模型：至少 3B/7B 级别 decoder-only LLM。
- 更高并发：直到普通 KV cache OOM 或吞吐明显饱和。
- fused decode attention：避免每步 materialize 完整历史 KV。

在当前环境中，不建议直接升级 vLLM/transformers，因为这会破坏已完成并验证的 Qwen2.5-VL 实验环境。

## 7. 产物路径

- 批处理并发脚本：`scripts/bench_text_turboquant_concurrency.py`
- Slurm 脚本：`sbatch/run_text_turboquant_bench.sbatch`
- ctx=1024 结果：`results/text_turboquant_batch_ctx1024/text_turboquant_summary.json`
- ctx=2048 结果：`results/text_turboquant_batch_ctx2048/text_turboquant_summary.json`
- ctx=4096 结果：`results/text_turboquant_batch_ctx4096/text_turboquant_summary.json`
- GPU 日志：`logs/9137_text_turboquant_gpu.csv`、`logs/9138_text_turboquant_gpu.csv`、`logs/9139_text_turboquant_gpu.csv`
