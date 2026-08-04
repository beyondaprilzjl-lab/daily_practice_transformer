# Transformer 面试手写练习

这是一个按天整理的 Transformer 与大模型基础组件手写练习仓库。

仓库目标不是实现生产级深度学习框架，而是帮助你在面试现场能够：

- 从公式推导出代码；
- 说清楚输入、输出和中间张量的形状；
- 在有限时间内写出简洁的核心实现；
- 使用一个最小示例当场验证结果；
- 回答数值稳定性、复杂度和常见追问。

## 主要使用场景

这个仓库主要面向以下场景：

1. **算法与大模型岗位面试**

   练习现场手写 Softmax、交叉熵、Attention、Transformer Block 等常见题目。

2. **面试前限时热身**

   给自己 10 至 20 分钟，不看答案完成一个组件，再运行文件内示例验证。

3. **从数学公式到代码**

   不只背 API，而是理解公式为什么能转换成对应的 NumPy 或 PyTorch 操作。

4. **口头表达训练**

   写代码的同时说明张量形状、数值稳定性、时间复杂度和空间复杂度。

这个仓库不追求完整的参数校验、工程封装或大型测试框架。面试时可以先向
面试官说明输入假设，再集中完成核心算法。

## 每日目录

```text
dayXX_topic/
├── README.md       # 公式推导、实现说明、复杂度与面试追问
├── exercise.py     # 留有 TODO 的练习骨架和验证示例
└── solution.py     # 完整参考实现和相同验证示例
```

`exercise.py` 和 `solution.py` 使用相同的函数签名与示例，方便练习后直接
对照。

## 使用方式

### 1. 安装依赖

```bash
python3 -m pip install -r requirements.txt
```

### 2. 进入当天目录

以 Day 01 为例：

```bash
cd day01_softmax_ce
```

### 3. 阅读公式

先阅读当天的 `README.md`，确保能够说明：

- 公式表达什么；
- 每个变量和维度代表什么；
- 为什么需要数值稳定处理；
- 时间和空间复杂度是多少。

### 4. 限时完成练习

打开 `exercise.py`，不查看 `solution.py`，将 TODO 替换为完整实现。

建议时间：

```text
基础算子：10 分钟
Attention 或 Transformer 组件：20 至 30 分钟
```

### 5. 运行文件内示例

```bash
python3 exercise.py
```

示例放在文件底部：

```python
if __name__ == "__main__":
    ...
```

它模拟面试官给出一组输入，由你现场判断输出是否合理，不依赖单独的测试
脚本。

### 6. 对照参考实现

完成并解释清楚后，再查看：

```bash
python3 solution.py
```

重点比较：

- 是否使用了正确公式；
- 是否保持向量化计算；
- 变量命名是否简洁；
- 是否存在不必要的工程代码；
- 是否能进一步缩短现场书写时间。

## 推荐练习流程

```text
公式推导
   ↓
说明输入输出形状
   ↓
限时完成 exercise.py
   ↓
运行文件内示例
   ↓
口头解释结果与复杂度
   ↓
对照 solution.py
```

## 当前进度

| 天数 | 主题 | 状态 |
| --- | --- | --- |
| Day 01 | 数值稳定的 Softmax、LogSoftmax 与交叉熵 | 已完成 |
| Day 02 | 多头自注意力（MHA） | 已完成 |
| Day 03 | 旋转位置编码（RoPE） | 已完成 |
| Day 04 | RMSNorm | 进行中 |

## Day 01 快速开始

```bash
cd day01_softmax_ce
python3 exercise.py
```

Day 01 需要能够独立解释和实现：

- 为什么 Softmax 要减去最大值；
- 为什么直接计算 LogSoftmax 更稳定；
- 交叉熵如何从 `-sum(y * log(p))` 化简为 `-log(p_target)`；
- 为什么 Softmax 与交叉熵的梯度是 `p - y`；
- `none`、`mean` 和 `sum` 三种 reduction 的区别。

## Day 02 快速开始

```bash
cd day02_mha
python3 exercise.py
```

Day 02 需要能够独立解释和实现：

- `[B, S, D]` 如何拆成 `[B, H, S, Dh]`；
- `QK^T` 为什么得到 `[B, H, S, S]`；
- 为什么注意力分数除以 `sqrt(Dh)`；
- causal mask 为什么要在 Softmax 之前应用；
- 多个注意力头如何合并回 `[B, S, D]`；
- 标准注意力为什么有序列长度平方级的开销。

## Day 03 快速开始

```bash
cd day03_rope
python3 exercise.py
```

Day 03 需要能够独立解释和实现：

- 二维向量的旋转公式如何转成代码；
- `rotate_half` 为什么得到 `[-x2, x1]`；
- 不同特征维度的旋转频率如何构造；
- `[S, Dh]` 的 `cos`、`sin` 如何广播到 `[B, H, S, Dh]`；
- 为什么 RoPE 应用到 `Q` 和 `K`，通常不应用到 `V`；
- 为什么旋转后的点积只依赖相对位置 `n - m`。

## Day 04 快速开始

```bash
cd day04_rmsnorm
python3 exercise.py
```

Day 04 需要能够独立解释和实现：

- RMS 如何由特征平方的平均值得到；
- 为什么沿最后一个特征维度归一化；
- `keepdim=True` 如何帮助广播；
- 可学习参数 `weight` 为什么是 `[D]`；
- RMSNorm 和 LayerNorm 的公式有什么区别；
- RMSNorm 在 Pre-Norm Transformer Block 中的位置。
