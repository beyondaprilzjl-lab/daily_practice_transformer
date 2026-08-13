# Day 08：KL 散度

## 今日目标

使用 NumPy 从零实现：

1. 单个或 Batch 概率分布的 `KL(p || q)`；
2. `none`、`mean`、`sum` 三种 reduction；
3. KL 散度和交叉熵之间的关系。

今天的重点不是背 API，而是理解：

```text
KL(p || q) 表示用 q 近似 p 时损失了多少信息。
```

## 文件说明

```text
day08_kl_divergence/
├── README.md       # 公式推导、性质和面试追问
├── exercise.py     # 练习骨架 + 文件底部示例
└── solution.py     # 完整参考实现 + 相同示例
```

## 1. KL 散度公式

给定两个离散概率分布：

```text
p = [p1, p2, ..., pC]
q = [q1, q2, ..., qC]
```

KL 散度定义为：

```text
                 p_i
KL(p || q) = sum p_i log ---
                  i      q_i
```

拆开对数：

```text
KL(p || q) = sum_i p_i * (log p_i - log q_i)
```

对应代码：

```python
losses = np.sum(p * (np.log(p_safe) - np.log(q_safe)), axis=-1)
```

其中：

```text
p：真实分布、目标分布或 teacher 分布
q：用于近似 p 的分布或 student 分布
```

## 2. 为什么写成 KL(p || q)

KL 散度有方向：

```text
KL(p || q) != KL(q || p)
```

`KL(p || q)` 的权重是 `p_i`：

```text
sum_i p_i * log(p_i / q_i)
```

而 `KL(q || p)` 的权重是 `q_i`：

```text
sum_i q_i * log(q_i / p_i)
```

权重不同，所以结果通常不同。

因此调用函数时，参数顺序不能随意交换：

```python
kl_divergence(p, q)
```

表示计算：

```text
KL(p || q)
```

## 3. KL 散度不是距离

KL 散度满足：

```text
KL(p || q) >= 0
```

并且：

```text
KL(p || q) = 0
```

当且仅当两个分布相同。

但它不是严格的距离，因为：

```text
KL(p || q) != KL(q || p)
```

也不满足普通距离要求的对称性。

面试时更准确的说法是：

```text
KL 散度衡量两个概率分布之间的差异，不是欧氏距离。
```

## 4. 为什么 KL 散度非负

使用不等式：

```text
log x <= x - 1
```

令：

```text
x = q_i / p_i
```

则：

```text
-log(q_i / p_i) >= 1 - q_i / p_i
```

两边乘以 `p_i` 并求和：

```text
KL(p || q)

= sum_i p_i log(p_i / q_i)

>= sum_i (p_i - q_i)

= 1 - 1

= 0
```

这称为 Gibbs 不等式。

## 5. 和交叉熵的关系

分布 `p` 的熵：

```text
H(p) = -sum_i p_i log p_i
```

`p` 和 `q` 的交叉熵：

```text
H(p, q) = -sum_i p_i log q_i
```

KL 散度：

```text
KL(p || q)

= sum_i p_i log p_i - sum_i p_i log q_i

= H(p, q) - H(p)
```

所以：

```text
H(p, q) = H(p) + KL(p || q)
```

如果目标分布 `p` 固定，那么 `H(p)` 是常数：

```text
最小化交叉熵 H(p, q)

等价于

最小化 KL(p || q)
```

## 6. One-hot 标签时

如果真实标签为类别 `t`，one-hot 分布满足：

```text
p_t = 1
其他 p_i = 0
```

那么：

```text
KL(p || q)

= 1 * log(1 / q_t)

= -log q_t
```

同时 one-hot 分布的熵为：

```text
H(p) = 0
```

因此此时 KL 散度就是普通分类交叉熵。

## 7. Batch 和 Shape

单个分布：

```text
p: [C]
q: [C]
```

沿最后一个类别维求和后：

```text
loss: scalar
```

Batch：

```text
p: [N, C]
q: [N, C]
```

先对每个样本沿类别维求和：

```text
losses: [N]
```

再执行 reduction：

```text
none: [N]
mean: scalar
sum:  scalar
```

## 8. reduction

`reduction="none"`：

```python
return losses
```

返回每个样本各自的 KL 散度。

`reduction="mean"`：

```python
return float(np.mean(losses))
```

返回 Batch 平均值。

`reduction="sum"`：

```python
return float(np.sum(losses))
```

返回 Batch 总和。

## 9. 为什么需要 clip

公式中包含：

```text
log p_i
log q_i
```

如果概率等于 0：

```text
log(0) = -inf
```

代码中使用：

```python
p_safe = np.clip(p, eps, 1.0)
q_safe = np.clip(q, eps, 1.0)
```

但最终的权重仍然使用原始 `p`：

```python
p * (log(p_safe) - log(q_safe))
```

这是因为数学上约定：

```text
0 * log(0 / q) = 0
```

当 `p_i = 0` 时，该类别不应贡献 KL。

如果把外面的权重也换成 `p_safe`，原本概率为 0 的类别就会产生一个很小但
不必要的贡献。

## 10. q 为 0 时会怎样

如果：

```text
p_i > 0
q_i = 0
```

理论上：

```text
p_i log(p_i / 0) = +inf
```

因为 `q` 认为这个事件绝不可能发生，但 `p` 认为它可能发生。

本日面试代码用 `eps` 将它近似为一个很大的有限值，避免运行时出现无穷大：

```python
q_safe = np.clip(q, eps, 1.0)
```

实际深度学习训练中，通常从 `log_softmax` 得到对数概率，会比先计算概率再
取对数更稳定。

## 11. 知识蒸馏中的 KL

知识蒸馏中：

```text
p = teacher 的概率分布
q = student 的概率分布
```

常见目标是：

```text
KL(teacher || student)
```

Teacher 的软概率不仅告诉 Student 正确类别是什么，也表达不同错误类别之间
的相似程度。

加入温度 `T`：

```text
p = softmax(teacher_logits / T)
q = softmax(student_logits / T)
```

温度越高，分布通常越平滑。

蒸馏损失常乘以：

```text
T^2
```

用于补偿除以温度后梯度尺度的变化。

本日核心代码只实现概率分布上的 KL，不额外加入温度逻辑。

## 12. 面试核心实现

```python
def kl_divergence(p, q, reduction="mean", eps=1e-12):
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)

    losses = np.sum(
        p * (np.log(p_safe) - np.log(q_safe)),
        axis=-1,
    )

    if reduction == "none":
        return losses
    if reduction == "sum":
        return float(np.sum(losses))
    if reduction == "mean":
        return float(np.mean(losses))
    raise ValueError(...)
```

现场可以先向面试官说明：

```text
p 和 q 是同形状的合法概率分布
最后一维是类别维
使用自然对数
用 eps 处理 log(0)
```

这样就不需要写大量输入校验。

## 13. 为什么使用自然对数

使用自然对数时，KL 的单位称为：

```text
nat
```

如果使用以 2 为底的对数，单位称为：

```text
bit
```

深度学习库通常使用自然对数。

## 14. 复杂度

对形状 `[N, C]` 的输入：

```text
时间复杂度：O(N * C)
空间复杂度：O(N * C)
```

如果不保存额外中间结果，理论上的辅助空间可以更低；但 NumPy 向量化运算会
创建中间数组。

## 15. 常见错误

### 错误一：忘记 p 的权重

错误：

```python
np.sum(np.log(p) - np.log(q))
```

正确：

```python
np.sum(p * (np.log(p) - np.log(q)))
```

### 错误二：把参数方向写反

```python
kl_divergence(q, p)
```

计算的是 `KL(q || p)`，不是 `KL(p || q)`。

### 错误三：沿 Batch 维求和

类别维是最后一维，所以应写：

```python
axis=-1
```

### 错误四：认为 KL 一定小于 1

KL 散度没有固定的 1 上界，可以大于 1，甚至在理论上为无穷大。

### 错误五：把 KL 当成对称距离

交换 `p`、`q` 后结果通常不同。

## 16. 面试追问

### Q1：KL 散度越小表示什么？

表示 `q` 对 `p` 的近似越好；等于 0 时两个分布相同。

### Q2：KL 散度可能为负数吗？

理论上不会。浮点误差可能产生非常接近 0 的微小负数。

### Q3：为什么交叉熵训练等价于最小化 KL？

因为 `H(p, q) = H(p) + KL(p || q)`，标签分布固定时 `H(p)` 是常数。

### Q4：KL(q || p) 和 KL(p || q) 有什么区别？

两者对不同分布加权，优化行为也不同，因此不能交换。

### Q5：PyTorch 的 KLDivLoss 为什么容易写反？

它通常接收模型的 `log q` 作为 input、目标分布 `p` 作为 target，计算的仍是
`KL(p || q)`。面试时最好先写清楚数学方向，再对应 API。

## 17. 运行方式

完成练习：

```bash
python3 day08_kl_divergence/exercise.py
```

运行参考实现：

```bash
python3 day08_kl_divergence/solution.py
```

示例会验证：

```text
相同分布的 KL 为 0
KL(p || q) 和 KL(q || p) 不相等
Batch reduction 正确
```
