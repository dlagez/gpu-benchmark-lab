# profiler_demo PRD

## 1. 目标

- 构造可控瓶颈场景，快速学习如何用 `torch.profiler` 定位 GPU 性能问题。
- 为后续真实模型压测提供一套可复现的分析基线。

## 2. 场景定义

- 代码入口：[benchmarks/micro/profiler_demo.py](../../benchmarks/micro/profiler_demo.py)
- 支持场景：
- `dataloader`: 增大 CPU 增强与模拟 I/O 延迟，观察数据供给瓶颈。
- `sync`: 强制 `.item()` 触发同步，观察隐式同步影响。
- `tiny_ops`: 注入大量小算子链，观察 kernel launch 过多问题。
- `memory_churn`: 每步创建大量临时 tensor，观察显存分配抖动。

## 3. 输入参数

- 关键参数：
- `--scenario`: 场景选择，默认 `dataloader`
- `--batch_size`: 默认 `64`
- `--num_workers`: 默认 `2`
- `--pin_memory`: 是否启用 pinned memory
- `--amp`: 是否启用 AMP
- `--warmup_steps`: 默认 `10`
- `--profile_steps`: 默认 `8`
- `--trace_dir`: 默认 `results/traces/profiler_demo`
- `--output_json`: 默认 `results/metrics_demo.json`

## 4. 输出与产物

- 控制台输出：
- `avg_step_time_s`
- `avg_dataloader_wait_s`
- Top CUDA ops 表格（按 `self_cuda_time_total` 排序）

- 文件输出：
- Trace: `results/traces/profiler_demo*`
- 指标摘要: `results/metrics_demo*.json`

## 5. 关键指标和判定

- `avg_step_time_s` 持续升高：优先看 GPU 计算和同步问题。
- `avg_dataloader_wait_s` 偏高：优先看数据加载线程、I/O 和 CPU augment。
- Top ops 中小 kernel 占比高：优先排查算子碎片化与 launch overhead。

## 6. 运行命令

- 直接运行：

```bash
python benchmarks/micro/profiler_demo.py --scenario dataloader --pin_memory --amp
```

- 脚本运行：

```bash
bash scripts/run_profiler.sh
```

- 切换场景：

```bash
python benchmarks/micro/profiler_demo.py --scenario tiny_ops --pin_memory --amp
```

- 可视化：

```bash
tensorboard --logdir results/traces
```

## 7. 验收标准

- 程序可在 CUDA 环境正常结束。
- 生成至少 1 份 trace 与 1 份 JSON 统计。
- 不同 `scenario` 下关键指标有可观察差异。

## 8. 风险与注意事项

- 非 CUDA 环境会直接报错退出。
- `num_workers=0` 时 `prefetch_factor` 和 `persistent_workers` 不生效。
- `profile_steps` 过小会导致样本不足，分析不稳定。
