# Transformer 每日手写练习

这个仓库按天整理 Transformer 与大模型基础组件的手写练习。每一天都从
数学原理出发，再完成代码实现和自动化测试。

## 每日目录结构

```text
dayXX_topic/
├── README.md       # 公式推导、实现说明、复杂度与面试追问
├── exercise.py     # 保留 TODO 的练习骨架
├── solution.py     # 完整参考实现
├── test_dayXX.py   # 自动化测试
└── run_tests.sh    # 一键运行测试
```

建议先独立完成 `exercise.py`，通过测试后再对照 `solution.py`。

## 学习进度

| 天数 | 主题 | 状态 |
| --- | --- | --- |
| Day 01 | 数值稳定的 Softmax、LogSoftmax 与交叉熵 | 已完成 |

## 环境

```bash
python3 -m pip install -r requirements.txt
```

进入当天目录后运行测试：

```bash
./run_tests.sh
```
