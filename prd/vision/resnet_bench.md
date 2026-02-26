# resnet_bench PRD

## 1. 目标

- 对 ResNet 推理或训练路径进行基准测试，产出吞吐、时延和显存占用数据。

## 2. 场景定义

- 场景名称：`resnet_bench`
- 适用模型：`ResNet` 系列
- 测试类型：`TODO`（训练/推理）

## 3. 输入参数

- `batch_size`: `TODO`
- `image_size`: `TODO`
- `precision`: `TODO`（fp32/amp）
- `iterations`: `TODO`

## 4. 输出与产物

- 控制台摘要：`TODO`
- 结果文件：`results/metrics.json` 或 `TODO`
- Trace 文件：`results/traces/` 或 `TODO`

## 5. 关键指标

- step time (ms)
- throughput (samples/s)
- GPU memory (MB)
- GPU utilization (%)

## 6. 运行命令

```bash
python benchmarks/vision/resnet_bench.py
```

## 7. 判定标准

- 与基线相比，吞吐下降超过 `TODO%` 视为回归。
- step time 增长超过 `TODO%` 需要排查。

## 8. 后续计划

- `TODO`: 增加更多输入尺寸。
- `TODO`: 支持多卡场景。

