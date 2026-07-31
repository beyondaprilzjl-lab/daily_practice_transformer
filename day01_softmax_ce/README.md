# Day 01：Softmax、LogSoftmax 与交叉熵

## 今日目标

使用 NumPy 从零实现：

1. 数值稳定的 `stable_softmax`
2. 数值稳定的 `stable_log_softmax`
3. 直接接收 logits 的多分类 `cross_entropy`

今天重点不是记住三行代码，而是理解为什么实现必须使用
log-sum-exp 技巧，以及 Softmax 和交叉熵为什么通常合并计算。

## 文件说明

```text
day01_softmax_ce/
├── README.md       # 公式推导与实现说明
├── exercise.py     # 练习骨架，所有核心计算留有 TODO
├── solution.py     # 完整参考实现
├── test_day01.py   # 数学正确性与数值稳定性测试
└── run_tests.sh    # 一键测试脚本
```

## 1. Softmax

### 1.1 定义

给定一个类别维度上的 logits：

```text
z = [z_1, z_2, ..., z_C]
```

Softmax 将任意实数映射为概率分布：

```text
             exp(z_i)
p_i = -----------------------
       sum_j exp(z_j)
```

因此：

```text
p_i > 0
sum_i p_i = 1
```

Softmax 不改变张量形状。若输入为 `[N, C]`，沿类别维 `axis=1`
计算后，输出仍为 `[N, C]`。

### 1.2 朴素实现为什么会溢出

浮点数能表示的范围有限。当 `z_i` 很大时，`exp(z_i)` 会溢出为
正无穷。例如直接计算 `exp(10000)` 无法得到有限结果，后续可能出现：

```text
inf / inf = NaN
```

给分母加 epsilon 不能解决指数函数已经溢出的问题。

### 1.3 平移不变性

对所有 logits 同时减去任意常数 `m`：

```text
softmax(z_i - m)

       exp(z_i - m)
= -----------------------
  sum_j exp(z_j - m)

       exp(z_i) exp(-m)
= -----------------------
  sum_j exp(z_j) exp(-m)

       exp(z_i)
= -----------------------
  sum_j exp(z_j)

= softmax(z_i)
```

因此，给同一组 logits 加上或减去同一个常数，不会改变 Softmax。

选择：

```text
m = max(z)
```

则所有平移后的值都满足 `z_i - m <= 0`，最大的指数项是
`exp(0) = 1`，从而避免正向溢出：

```text
shifted = z - max(z)
p = exp(shifted) / sum(exp(shifted))
```

很小的指数项下溢为 0 通常是可接受的，因为它本来就代表接近 0 的概率。

## 2. LogSoftmax

### 2.1 直接推导

从 Softmax 取对数：

```text
log p_i

             exp(z_i)
= log -----------------------
       sum_j exp(z_j)

= z_i - log(sum_j exp(z_j))
```

结合最大值平移：

```text
log p_i
= (z_i - m) - log(sum_j exp(z_j - m))

其中 m = max(z)
```

这就是数值稳定的 LogSoftmax。

### 2.2 为什么不写成 log(softmax(z))

某个类别概率极小时，Softmax 结果可能下溢并被舍入为 0：

```text
log(0) = -inf
```

直接使用上面的 LogSoftmax 公式，不需要先把极小值压缩到概率空间，
能够保留有限且更准确的对数概率。

## 3. 多分类交叉熵

### 3.1 从定义化简

对 one-hot 标签 `y`，交叉熵定义为：

```text
L = -sum_i y_i log p_i
```

真实类别记为 `t`。因为只有 `y_t = 1`，其余位置都是 0：

```text
L = -log p_t
```

代入 LogSoftmax：

```text
L = -z_t + log(sum_j exp(z_j))
```

数值稳定形式为：

```text
L = -(z_t - m) + log(sum_j exp(z_j - m))
```

代码中先计算稳定的 LogSoftmax，再用高级索引取出每个样本真实类别的
对数概率：

```python
log_probs = stable_log_softmax(logits, axis=1)
losses = -log_probs[np.arange(batch_size), targets]
```

### 3.2 Batch 与 reduction

输入和输出形状：

```text
logits:  [N, C]
targets: [N]
losses:  [N]      reduction="none"
loss:    scalar   reduction="mean" 或 "sum"
```

- `none`：返回每个样本的损失。
- `mean`：返回 batch 平均损失。
- `sum`：返回 batch 损失之和。

## 4. Softmax + 交叉熵的梯度

单个样本的损失：

```text
L = -z_t + log(sum_j exp(z_j))
```

对第 `k` 个 logit 求导：

```text
dL/dz_k

= -1[k = t] + exp(z_k) / sum_j exp(z_j)

= p_k - 1[k = t]
```

写成向量形式：

```text
dL/dz = p - y
```

这说明：

- 对真实类别，梯度是 `p_t - 1`，训练会提高它的 logit。
- 对其他类别，梯度是 `p_k`，训练会降低它们的 logit。
- 当预测概率已经接近真实标签时，梯度自然变小。

## 5. 面试时到底写多少

面试手写默认题目已经约定好输入：

```text
logits 是浮点数组
logits 形状为 [N, C]
targets 形状为 [N]，并且类别下标合法
```

因此不需要现场编写 `_validate_logits`，也不需要检查空 batch、数据类型、
NaN 或越界标签。先向面试官说明这些输入假设，然后专注写出核心算法。

参考实现只有三个关键步骤：

```text
Softmax:       减最大值 -> exp -> 除以 exp 之和
LogSoftmax:    减最大值 -> 减去 log(exp 之和)
CrossEntropy:  LogSoftmax -> 取真实类别 -> reduction
```

只有当面试官明确追问“生产环境如何处理非法输入”时，再口头补充参数校验。

## 6. 实现约束

- 不调用现成的 softmax、log-softmax 或 cross-entropy。
- 不使用 epsilon 掩盖数值问题。
- 不遍历 batch 或类别，全部使用向量化运算。
- Softmax 和 LogSoftmax 支持正数与负数 `axis`。
- `cross_entropy` 只接收 `[N, C]` logits 和 `[N]` 整数标签。

## 7. 复杂度

对形状为 `[N, C]` 的输入：

```text
时间复杂度：O(N * C)
空间复杂度：O(N * C)
```

最大值、指数、求和和归一化都只需要线性扫描。参考实现为了代码清晰会保存
中间数组；框架内部可以通过算子融合减少中间结果和显存读写。

## 8. 运行方式

安装依赖：

```bash
python3 -m pip install -r ../requirements.txt
```

测试完整实现：

```bash
./run_tests.sh
```

完成 `exercise.py` 后，用同一套测试检查练习代码：

```bash
DAY01_MODULE=exercise ./run_tests.sh
```

也可以直接运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_day01.py
```

## 9. 完成标准

- 所有测试通过。
- `10000`、`-10000` 等极端 logits 仍产生有限结果。
- 没有逐样本或逐类别循环。
- 能独立推导稳定 Softmax、LogSoftmax 和交叉熵。
- 能解释交叉熵关于 logits 的梯度为什么是 `p - y`。
- 能在 10 分钟内写完三个函数的核心实现。

## 10. 面试追问

1. 所有 logits 都相等时，输出概率是多少？
2. 给所有 logits 加同一个常数，结果为什么不变？
3. 为什么交叉熵通常接收 logits，而不是 Softmax 后的概率？
4. 如何将实现扩展到 `[B, S, V]` 的语言模型 logits？
5. 如何加入 `ignore_index`、类别权重和 label smoothing？
6. 为什么一组 logits 中出现正无穷时需要单独定义处理策略？
