# PRD 文档目录

这个目录用于记录每个 benchmark demo 的产品说明（PRD），统一描述目标、输入、输出、指标和验证方式。

## 文档规范

- 每个 demo 对应一个 `*.md` 文件。
- 与代码结构一一对应，便于维护和追踪。
- 建议每次改动 demo 参数或行为时同步更新文档。

## 目录映射

- Vision
- [resnet_bench.md](vision/resnet_bench.md)
- [conv2d_bench.md](vision/conv2d_bench.md)
- [amp_vs_fp32.md](vision/amp_vs_fp32.md)

- Transformer
- [attention_bench.md](transformer/attention_bench.md)
- [matmul_bench.md](transformer/matmul_bench.md)
- [layernorm_bench.md](transformer/layernorm_bench.md)

- Micro
- [bandwidth_test.md](micro/bandwidth_test.md)
- [latency_test.md](micro/latency_test.md)
- [kernel_launch_overhead.md](micro/kernel_launch_overhead.md)
- [profiler_demo.md](micro/profiler_demo.md)
