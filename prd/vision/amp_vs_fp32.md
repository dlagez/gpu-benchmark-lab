# amp_vs_fp32 PRD

## 1. 目标

- 对比 AMP 与 FP32 在同一模型和数据配置下的性能与数值行为。

## 2. 场景定义

- 场景名称：`amp_vs_fp32`
- 对比维度：吞吐、时延、显存占用、loss 稳定性

## 3. 输入参数

- `batch_size`: `TODO`
- `model`: `TODO`
- `amp_enabled`: `True/False`
- `steps`: `TODO`

## 4. 输出与产物

- 对比表：`TODO`
- 结果文件：`TODO`

## 5. 关键指标

- speedup (AMP vs FP32)
- max memory allocated
- loss divergence check

## 6. 运行命令

```bash
python benchmarks/vision/amp_vs_fp32.py
```

## 7. 判定标准

- AMP 吞吐提升低于 `TODO%` 需要排查。
- loss 异常抖动需检查 scaler 或超参数。

