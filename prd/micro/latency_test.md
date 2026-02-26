# latency_test PRD

## 1. 目标

- 量化小规模算子或短序列任务的端到端延迟。

## 2. 场景定义

- 场景名称：`latency_test`
- 关注点：启动开销、同步点、host-device 交互

## 3. 输入参数

- `batch_size`: `TODO`
- `steps`: `TODO`
- `sync_mode`: `TODO`

## 4. 输出与产物

- P50/P90/P99 延迟：`TODO`
- trace：`TODO`

## 5. 关键指标

- average latency
- tail latency
- jitter

## 6. 运行命令

```bash
python benchmarks/micro/latency_test.py
```

