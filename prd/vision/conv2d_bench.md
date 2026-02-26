# conv2d_bench PRD

## 1. 目标

- 评估 Conv2D 算子在不同输入尺寸和参数下的性能表现。

## 2. 场景定义

- 场景名称：`conv2d_bench`
- 核心对象：`torch.nn.Conv2d`
- 测试类型：`TODO`（前向/前后向）

## 3. 输入参数

- `batch_size`: `TODO`
- `in_channels/out_channels`: `TODO`
- `kernel_size/stride/padding`: `TODO`
- `dtype`: `TODO`

## 4. 输出与产物

- 控制台摘要：`TODO`
- 指标文件：`TODO`
- Trace 文件：`TODO`

## 5. 关键指标

- kernel time (ms)
- effective TFLOPS
- memory bandwidth (GB/s)

## 6. 运行命令

```bash
python benchmarks/vision/conv2d_bench.py
```

## 7. 判定标准

- 同参数下，性能波动超过 `TODO%` 触发告警。

