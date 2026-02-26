# matmul_bench PRD

## 1. 目标

- 评估 GEMM/MatMul 在不同矩阵规模与精度下的吞吐和效率。

## 2. 场景定义

- 场景名称：`matmul_bench`
- 测试对象：`torch.matmul` 或自定义 matmul 实现

## 3. 输入参数

- `M/N/K`: `TODO`
- `batch`: `TODO`
- `dtype`: `TODO`（fp32/fp16/bf16）
- `iterations`: `TODO`

## 4. 输出与产物

- 吞吐统计：`TODO`
- 结果存档：`TODO`

## 5. 关键指标

- latency (ms)
- TFLOPS
- occupancy（可选）

## 6. 运行命令

```bash
python benchmarks/transformer/matmul_bench.py
```

