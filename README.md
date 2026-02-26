# gpu-benchmark-lab

`gpu-benchmark-lab` 是一个面向 GPU 性能工程的基准测试与分析框架，聚焦于 **PyTorch 工作负载、Transformer/视觉模型核心算子，以及底层 CUDA/Triton 内核** 的可重复评测。
项目目标是提供一套统一、可扩展、可追溯的性能实验基础设施，帮助你系统回答以下问题：
- 哪个模型/算子在当前 GPU 上是主要瓶颈？
- AMP、算子实现差异（PyTorch/CUDA/Triton）对吞吐与延迟的影响？
- 性能退化是否来自内核、调度、数据输入，还是硬件带宽限制？

## Key Capabilities

- 统一基准入口：覆盖 `vision`、`transformer`、`micro` 三类场景
- 多维性能分析：支持 `torch.profiler`、Nsight 运行封装与 trace 解析
- 自定义内核扩展：支持 CUDA C++ 与 Triton 算子开发和对比
- 自动化输出：生成结构化结果（JSON/trace/log）用于报告与回归分析
- 工程化组织：配置、运行、分析、测试模块解耦，便于 CI/CD 集成

## Scope

该仓库适用于以下任务：
- 新硬件或新驱动环境下的性能基线建立
- 版本升级（CUDA/PyTorch/模型代码）前后的回归验证
- 针对关键算子的优化实验与收益量化

## Quick Start

```bash
pip install -r requirements.txt
python -m core.runner --help
```

后续可通过 `configs/` 中的 YAML 配置选择具体 benchmark 任务，并将结果输出到 `results/` 目录统一管理。

## Python venv 环境创建（Linux Bash）

```bash
cd /path/to/gpu-benchmark-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

验证 CUDA 可用性：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

退出虚拟环境：

```bash
deactivate
```
