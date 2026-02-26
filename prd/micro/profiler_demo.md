# profiler_demo PRD

## 1. 目标

- 构造可控瓶颈场景，快速学习如何用 `torch.profiler` 定位 GPU 性能问题。
- 为后续真实模型压测提供一套可复现的分析基线。

## 2. 场景定义

- 代码入口：[benchmarks/micro/profiler_demo.py](../../benchmarks/micro/profiler_demo.py)
- 支持场景：
- `dataloader`：增大 CPU 增强与模拟 I/O 延迟，观察数据供给瓶颈。
- `sync`：强制 `.item()` 触发同步，观察隐式同步影响。
- `tiny_ops`：注入大量小算子链，观察 kernel launch 过多问题。
- `memory_churn`：每步创建大量临时 tensor，观察显存分配抖动。

## 3. 输入参数

- 关键参数：
- `--scenario`：场景选择，默认 `dataloader`
- `--batch_size`：默认 `64`
- `--num_workers`：默认 `2`
- `--pin_memory`：是否启用 pinned memory
- `--amp`：是否启用 AMP
- `--warmup_steps`：默认 `10`
- `--profile_steps`：默认 `8`
- `--trace_dir`：默认 `results/traces/profiler_demo`
- `--output_json`：默认 `results/metrics_demo.json`

## 4. train 过程详解（对应代码 123-232 行）

### 4.1 初始化与设备检查

- 固定随机种子：`torch.manual_seed(args.seed)`，提高复现实验的一致性。
- 设备选择：优先 `cuda`，否则 `cpu`。
- 若不是 CUDA，直接 `RuntimeError` 退出。这个 demo 只用于 CUDA profiling。

### 4.2 按场景设定数据压力

- `scenario == dataloader` 时：
- `cpu_work = 4000`，`io_sleep_ms = 2`
- 作用：在 `Dataset.__getitem__` 中增加 CPU 计算和 I/O 等待，放大“数据加载慢”问题。
- 其他场景：`cpu_work = 200`，`io_sleep_ms = 0`，减少数据侧干扰。

### 4.3 构建 Dataset/DataLoader

- `SlowCPUDataset` 用于生成合成图像和标签，内部可注入 CPU augment 与 sleep 延迟。
- `DataLoader` 核心配置：
- `pin_memory=args.pin_memory`：配合 `non_blocking=True` 可优化 H2D 拷贝。
- `num_workers=args.num_workers`：控制并行加载进程。
- `persistent_workers`、`prefetch_factor` 仅在 `num_workers > 0` 时生效。
- `drop_last=True`：保证 batch 形状稳定，减少尾批差异影响统计。

### 4.4 模型、优化器与 AMP 组件

- 初始化 `SimpleCNN`（主干）和 `TinyOpsBlock`（仅 `tiny_ops` 场景用）。
- 优化器：`AdamW`，同时更新两部分参数。
- AMP：`GradScaler(enabled=args.amp)`，仅在开启 AMP 时进行动态缩放。

### 4.5 Warmup 预热阶段

- 手动取 `iterator = iter(loader)`，运行 `warmup_steps` 步。
- 每步执行：`zero_grad -> run_step -> backward -> step -> scaler.update`。
- 目的：让 cudnn 算法选择、缓存分配等进入稳定状态，避免冷启动污染 profile。

### 4.6 输出目录准备

- `trace_dir` 必建。
- 若启用 `output_json`，其父目录也会创建。

### 4.7 Profiler 配置

- 活动类型：CPU + CUDA。
- `record_shapes=True`：记录算子输入 shape。
- `profile_memory=True`：记录内存相关信息。
- `with_stack=args.with_stack`：可选记录调用栈。
- `on_trace_ready=tensorboard_trace_handler(trace_dir)`：自动落盘 trace，供 TensorBoard 查看。
- `schedule(wait=1, warmup=1, active=profile_steps, repeat=1)`：
- 前 1 步等待，不采样。
- 再 1 步 warmup。
- 后 `profile_steps` 步正式采样。

### 4.8 主循环计时与采样

- 循环上限：`1 + 1 + profile_steps` 步，和 schedule 对齐。
- `data_wait_times`：每步开始时，用 `now - end_prev` 估算“上一步结束后到拿到本步 batch”的等待时间。
- 步耗时测量：
- `torch.cuda.synchronize(); t0 = time.time()`
- 执行一个 `train_step`（含前向/反向/优化）
- `torch.cuda.synchronize(); t1 = time.time()`
- 记录 `step_times.append(t1 - t0)`
- 调用 `prof.step()` 推进 profiler 状态机。

### 4.9 统计汇总与输出

- 生成 `stats`：
- 实验配置（`scenario/batch_size/num_workers/pin_memory/amp`）
- 指标（`avg_step_time_s`、`avg_dataloader_wait_s`）
- 路径（`trace_dir`）
- 控制台输出：
- quick stats
- 按 `self_cuda_time_total` 排序的 Top CUDA ops 表
- 若 `output_json` 开启，写入 JSON 统计文件。

### 4.10 如何理解两个核心指标

- `avg_step_time_s`：单步训练时间（含前向、反向、优化器）。
- `avg_dataloader_wait_s`：等待数据时间。
- 若后者明显偏大，通常先查数据管线（`num_workers`、I/O、CPU augment）。
- 若前者偏大且后者不高，通常先查算子与内核执行效率。

## 5. 输出与产物

- 控制台输出：
- `avg_step_time_s`
- `avg_dataloader_wait_s`
- Top CUDA ops 表格（按 `self_cuda_time_total` 排序）
- 文件输出：
- Trace: `results/traces/profiler_demo*`
- 指标摘要: `results/metrics_demo*.json`

## 6. 关键指标和判定

- `avg_step_time_s` 持续升高：优先看 GPU 计算和同步问题。
- `avg_dataloader_wait_s` 偏高：优先看数据加载线程、I/O 和 CPU augment。
- Top ops 中小 kernel 占比高：优先排查算子碎片化与 launch overhead。

## 7. 运行命令

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

## 8. 验收标准

- 程序可在 CUDA 环境正常结束。
- 生成至少 1 份 trace 与 1 份 JSON 统计。
- 不同 `scenario` 下关键指标有可观察差异。

## 9. 风险与注意事项

- 非 CUDA 环境会直接报错退出。
- `num_workers=0` 时 `prefetch_factor` 和 `persistent_workers` 不生效。
- `profile_steps` 过小会导致样本不足，分析不稳定。
