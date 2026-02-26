# bandwidth_test PRD

## 1. 目标

- 测试设备在特定访存模式下的有效带宽上限。

## 2. 场景定义

- 场景名称：`bandwidth_test`
- 测试对象：全局内存读写路径

## 3. 输入参数

- `tensor_size`: `TODO`
- `dtype`: `TODO`
- `iterations`: `TODO`

## 4. 输出与产物

- 带宽统计：`TODO`
- 结果文件：`TODO`

## 5. 关键指标

- achieved bandwidth (GB/s)
- variance across runs

## 6. 运行命令

```bash
python benchmarks/micro/bandwidth_test.py
```

