# kernel_launch_overhead PRD

## 1. 目标

- 评估大量小 kernel 场景下的 launch overhead 占比。

## 2. 场景定义

- 场景名称：`kernel_launch_overhead`
- 对象：高频小算子调用链

## 3. 输入参数

- `num_kernels`: `TODO`
- `tensor_size`: `TODO`
- `dtype`: `TODO`

## 4. 输出与产物

- launch 开销统计：`TODO`
- trace：`TODO`

## 5. 关键指标

- total launch time
- launch/compute ratio
- kernels per second

## 6. 运行命令

```bash
python benchmarks/micro/kernel_launch_overhead.py
```

