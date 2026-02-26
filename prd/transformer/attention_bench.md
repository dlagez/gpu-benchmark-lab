# attention_bench PRD

## 1. 目标

- 分析 Attention 模块在不同序列长度和 head 配置下的性能瓶颈。

## 2. 场景定义

- 场景名称：`attention_bench`
- 主要测试项：QKV 投影、attention score、softmax、value 聚合

## 3. 输入参数

- `batch_size`: `TODO`
- `seq_len`: `TODO`
- `num_heads`: `TODO`
- `hidden_size`: `TODO`
- `dtype`: `TODO`

## 4. 输出与产物

- 关键算子耗时分解：`TODO`
- trace：`results/traces/` 或 `TODO`

## 5. 关键指标

- attention block latency
- memory usage
- kernel launch count

## 6. 运行命令

```bash
python benchmarks/transformer/attention_bench.py
```

