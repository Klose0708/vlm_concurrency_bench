# TurboQuant 与普通 KV Cache 并发实验技术报告

## 1. 实验目标

本轮实验补充完成用户要求的 TurboQuant 对比：

- 在 GitHub 上寻找可用的 TurboQuant 开源实现。
- 使用真实 TurboQuant KV cache 方法运行并发实验。
- 使用普通方法作为对照运行并发实验。
- 保留此前已完成的 VLM 普通方法结果，并明确其与 TurboQuant 实验的边界。

所有模型推理和并发实验均通过 Slurm 提交到计算节点执行，不在主节点直接进行 GPU 推理实验。本轮文本 TurboQuant 正式作业为 `9134`，日志显示运行节点为 `compute2`。

## 2. 开源实现调研与选择

本轮重点核查了以下 GitHub 实现：

- `0xSero/turboquant`：包含 TurboQuant 核心量化、Triton kernel 与实验性 vLLM 集成，但 README 标注测试环境为 vLLM 0.18.0、Python 3.12、CUDA 12.8 等新版栈。
- `Alberto-Codes/turboquant-consumer` / `turboquant-vllm`：提供 vLLM 插件和 `CompressedDynamicCache`，但 `pyproject.toml` 要求 Python >= 3.12、vLLM >= 0.18、transformers >= 4.57。
- `back2matching/turboquant`：提供 HuggingFace 版 TurboQuant KV cache 和核心 `TurboQuantMSE`，Python >= 3.10，依赖范围更接近当前实验环境。

当前已跑通的 VLM 环境为 Python 3.10、vLLM 0.8.5.post1、transformers 4.51.3。直接安装新版 vLLM TurboQuant 插件会强制升级 vLLM/transformers，并可能破坏已经完成的 Qwen2.5-VL 实验环境。因此本轮没有在现有 VLM vLLM 服务中强行接入新版插件，而是选择 `back2matching/turboquant` 的核心 TurboQuant MSE 量化实现，并编写兼容 transformers 4.51.3 的 cache 适配脚本。

需要注意：该开源实现采用 TurboQuant 的 MSE 路径，即随机旋转 + Beta 分布最优标量量化；仓库文档也说明其默认不使用 QJL 残差路径，因为社区复现实验认为 QJL 噪声会被 attention softmax 放大。

## 3. 实验设计

### 3.1 VLM 普通方法结果

此前已按 `Experimental_Design_Proposal.md` 完成 Qwen2.5-VL-3B-Instruct 在 MMStar 全量 1500 样本上的普通方法实验，配置包括：

- BF16：普通全精度 serving baseline。
- BF16 + FP8 KV cache：vLLM 内置 KV cache 量化。
- AWQ：权重量化模型，因 AWQ Marlin 与 `torch.compile` 触发 CUDA illegal memory access，正式实验使用 `--enforce-eager`。

这部分不是 TurboQuant 实验，而是 VLM 普通方法/可部署 KV cache baseline。其结果用于报告中说明 VLM 任务上的普通 serving 表现。

### 3.2 文本 LLM TurboQuant 对照实验

由于现有 vLLM 版本不能无风险接入新版 TurboQuant 插件，本轮新增文本 LLM 实验以真实使用 TurboQuant KV cache：

- 模型：`Qwen2.5-0.5B-Instruct`。
- 运行节点：Slurm 计算节点 `compute2`。
- 普通方法：transformers `DynamicCache`。
- TurboQuant 方法：基于 `back2matching/turboquant` 的 `TurboQuantMSE`，在自定义 `CompatTurboQuantCache` 中压缩历史 KV，保留 128 token residual window。
- 上下文规模：约 768 prompt tokens。
- 输出上限：16 tokens。
- 并发档位：1、2、4。
- 每档请求数：8。
- 正式作业：`9134`。

实验脚本为 `scripts/bench_text_turboquant_concurrency.py`，Slurm 脚本为 `sbatch/run_text_turboquant_bench.sbatch`。

## 4. 结果

### 4.1 文本 LLM：TurboQuant vs 普通 DynamicCache

| 方法 | 并发 | 请求数 | 错误率 | requests/s | output tokens/s | P95 延迟(s) | P95 TTFT(s) | 平均 KV 存储(MB) | FP16 等价 KV(MB) | KV 压缩率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通 DynamicCache | 1 | 8 | 0.0000 | 2.0795 | 33.2726 | 0.9099 | 0.4597 | 8.6602 | 8.6602 | 1.0000 |
| 普通 DynamicCache | 2 | 8 | 0.0000 | 2.2794 | 36.4705 | 0.8019 | 0.0259 | 8.6602 | 8.6602 | 1.0000 |
| 普通 DynamicCache | 4 | 8 | 0.0000 | 2.2820 | 36.5120 | 1.6674 | 0.0271 | 8.6602 | 8.6602 | 1.0000 |
| TurboQuant KV cache | 1 | 8 | 0.0000 | 0.5641 | 9.0252 | 1.9974 | 1.2543 | 3.5138 | 8.6602 | 2.4646 |
| TurboQuant KV cache | 2 | 8 | 0.0000 | 0.5386 | 8.6180 | 3.6913 | 1.0370 | 3.5138 | 8.6602 | 2.4646 |
| TurboQuant KV cache | 4 | 8 | 0.0000 | 0.5737 | 9.1790 | 7.0916 | 1.0382 | 3.5138 | 8.6602 | 2.4646 |

关键观察：

- 两种方法在全部并发档位均 0 错误。
- TurboQuant 将平均 KV 存储从 8.66 MB 降低到 3.51 MB，约 2.46 倍压缩。
- 在当前 HuggingFace 适配路径下，TurboQuant 吞吐明显低于普通 DynamicCache：最高约 0.57 req/s，而普通方法约 2.28 req/s。
- TurboQuant 的 P95 端到端延迟随并发升高更明显，c=4 达到约 7.09 秒。
- 主要瓶颈来自在线量化、反量化和 Python/HuggingFace cache 适配开销；这不是融合 kernel 版 TurboQuant 在新版 vLLM 中的理论上限。

### 4.2 VLM 普通方法摘要

此前 VLM 普通方法全量实验的最佳 0 错误点如下：

| 配置 | 最佳 0 错误并发 | requests/s | P95 延迟(s) | 准确率 |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 32 | 25.5822 | 2.2261 | 0.5453 |
| BF16 + FP8 KV | 32 | 14.5211 | 3.8216 | 0.5473 |
| AWQ | 4 | 12.6400 | 0.6249 | 0.5347 |

解释：

- BF16 是当前 Qwen2.5-VL 短输出任务中最强的稳定 baseline。
- FP8 KV cache 稳定，但在该 MMStar 短输出 workload 下比 BF16 慢，说明 KV cache 还不是主瓶颈。
- AWQ 在并发 1、2、4 稳定；此前 500 样本高并发探测中并发 8 及以上出现 ECC/CUDA 相关不稳定。

## 5. 结论

本轮确实完成了真实 TurboQuant KV cache 实验，但实验对象从 VLM vLLM 服务调整为文本 LLM HuggingFace 推理，原因是当前 VLM 环境无法无风险接入新版 TurboQuant vLLM 插件。

在当前可落地实现下，TurboQuant 的主要收益是 KV cache 存储压缩，768 token 上下文下约 2.46 倍；但由于使用 HuggingFace 自定义 cache 路径、在线量化和反量化未充分融合，吞吐和延迟明显劣于普通 DynamicCache。因此，当前结果应解读为“开源 HuggingFace TurboQuant 适配实现的工程可行性与代价”，而不是 TurboQuant 论文中 fused attention / fused decode kernel 的性能上限。

对于本项目的 VLM 并发服务，短期最可靠的普通方法结论仍是：BF16 在当前 vLLM 0.8.5.post1 环境下吞吐和稳定性最好；FP8 KV cache 可作为 KV 压缩 baseline，但在短上下文/短输出 MMStar 任务中不占优。若后续要在 Qwen2.5-VL 上做严格 TurboQuant-vLLM 对比，建议单独新建环境并升级到支持 TurboQuant 插件的新版 vLLM/transformers 栈，避免污染当前已验证环境。

## 6. 产物路径

- TurboQuant 文本实验脚本：`scripts/bench_text_turboquant_concurrency.py`
- TurboQuant 文本 Slurm 脚本：`sbatch/run_text_turboquant_bench.sbatch`
- TurboQuant 文本正式结果：`results/text_turboquant_formal_c124/text_turboquant_summary.json`
- TurboQuant 文本正式请求明细：`results/text_turboquant_formal_c124/text_turboquant_requests.jsonl`
- TurboQuant 文本 GPU 日志：`logs/9134_text_turboquant_gpu.csv`
- VLM 普通方法报告：`reports/experiment_report.md`
- VLM 普通方法 CSV：`reports/formal_summary_1500.csv`
- 可视化摘要 Canvas：`turboquant-comparison-results.canvas.tsx`
