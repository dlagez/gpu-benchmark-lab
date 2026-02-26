# layernorm_bench PRD

## 1. 目标

- 对比 LayerNorm 在不同维度、数据类型和实现方式下的性能。

## 2. 场景定义

- 场景名称：`layernorm_bench`
- 对象：PyTorch 原生或自定义 LayerNorm 算子

## 3. 输入参数

- `batch_size`: `TODO`
- `hidden_dim`: `TODO`
- `dtype`: `TODO`
- `impl`: `TODO`

## 4. 输出与产物

- 统计摘要：`TODO`
- trace：`TODO`

## 5. 关键指标

- kernel latency
- memory bandwidth
- 数值误差（若做实现对比）

## 6. 运行命令

```bash
python benchmarks/transformer/layernorm_bench.py
```

