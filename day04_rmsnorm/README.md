# Day 04：RMSNorm

## 今日目标

使用 PyTorch 从零实现：

1. 函数版 `rms_norm`；
2. 带可学习缩放参数的 `RMSNorm` 模块。

今天的重点是理解 RMSNorm 在最后一个特征维度上做了什么，以及它和
LayerNorm 的区别。

## 文件说明

```text
day04_rmsnorm/
├── README.md       # 公式推导、形状说明与面试追问
├── exercise.py     # 练习骨架 + 文件底部验证示例
└── solution.py     # 完整参考实现 + 相同验证示例
```

## 1. RMS 是什么

给定一个 `D` 维向量：

```text
x = [x1, x2, ..., xD]
```

它的均方根 RMS 为：

```text
                 1
RMS(x) = sqrt( ----- sum_i xi^2 )
                 D
```

加入防止除零的 `eps`：

```text
                       1
RMS(x) = sqrt(eps +  ----- sum_i xi^2)
                       D
```

RMS 也可以理解为向量的 L2 范数除以 `sqrt(D)`：

```text
RMS(x) = ||x||_2 / sqrt(D)
```

## 2. RMSNorm 公式

先用 RMS 缩放输入：

```text
x_hat = x / RMS(x)
```

再乘以可学习参数 `gamma`：

```text
y = gamma * x_hat
```

完整公式：

```text
RMSNorm(x) = gamma * x / sqrt(mean(x^2) + eps)
```

对应代码：

```python
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
return x / rms * weight
```

## 3. 为什么沿最后一维计算

Transformer 中的输入通常为：

```text
x: [B, S, D]
```

RMSNorm 对每个样本、每个 token 的 `D` 个特征分别归一化：

```text
输入： [B, S, D]
RMS：  [B, S, 1]
输出： [B, S, D]
```

因此：

```python
mean(dim=-1, keepdim=True)
```

- `dim=-1`：只在特征维 `D` 上求平均；
- `keepdim=True`：保留大小为 1 的维度，方便与 `x` 广播相除。

如果沿序列维 `S` 计算，就会混合不同 token，不是 RMSNorm 的定义。

## 4. weight 为什么只有一个维度

可学习参数：

```text
weight: [D]
```

输入：

```text
x_hat: [B, S, D]
```

PyTorch 会把 `[D]` 广播到所有 batch 和 token：

```text
[B, S, D] * [D] -> [B, S, D]
```

每个特征都有自己的缩放参数，但所有 token 共享同一组参数。

初始化时：

```python
weight = torch.ones(dim)
```

这样模块刚开始训练时不会额外改变归一化结果。

## 5. 和 LayerNorm 的区别

LayerNorm：

```text
mean = mean(x)
var = mean((x - mean)^2)

LayerNorm(x)

             x - mean
= gamma * --------------- + beta
           sqrt(var + eps)
```

RMSNorm：

```text
RMSNorm(x)

                 x
= gamma * -------------------
           sqrt(mean(x^2) + eps)
```

主要区别：

| 对比项 | LayerNorm | RMSNorm |
| --- | --- | --- |
| 减去均值 | 是 | 否 |
| 归一化依据 | 方差 | 均方根 |
| 可学习缩放 `gamma` | 有 | 有 |
| 可学习偏置 `beta` | 通常有 | 通常没有 |
| 保证输出均值为 0 | 是 | 否 |

RMSNorm 只控制向量的整体尺度，不会把特征移动到零均值。

## 6. 为什么需要 eps

当输入全部为 0：

```text
mean(x^2) = 0
```

如果没有 `eps`：

```text
x / 0
```

会产生 `NaN` 或无穷大。

将 `eps` 放在开根号内部：

```text
sqrt(mean(x^2) + eps)
```

可以保证分母大于 0。

## 7. 归一化后的 RMS

暂时忽略 `eps` 和 `weight`：

```text
x_hat = x / RMS(x)
```

那么：

```text
RMS(x_hat)

= RMS(x / RMS(x))

= RMS(x) / RMS(x)

= 1
```

因此文件底部示例会检查输出最后一维的 RMS 是否接近 1。

乘上可学习的 `weight` 后，每个特征可以再调整到模型需要的尺度。

## 8. 在 Transformer 中的位置

现代大模型经常使用 Pre-Norm 结构：

```text
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

也就是先归一化，再进入 Attention 或 FFN，最后与残差连接相加。

RMSNorm 不负责残差连接，它只处理传入的张量。

## 9. 面试时的核心实现

现场只需要完成：

```text
1. square
2. mean over the last dimension
3. add eps and sqrt
4. divide x by rms
5. multiply weight
```

默认可以先向面试官说明：

```text
x 的最后一维等于 dim
输入是浮点张量
沿最后一个特征维归一化
本题不处理混合精度等生产环境细节
```

## 10. 复杂度

对形状为 `[B, S, D]` 的输入：

```text
时间复杂度：O(B * S * D)
参数量：    O(D)
```

核心中间结果 `rms` 的形状为 `[B, S, 1]`。

## 11. 面试常见追问

- RMSNorm 为什么不减去均值？
- `dim=-1` 和 `keepdim=True` 分别有什么作用？
- 为什么 `weight` 的形状是 `[D]`？
- 为什么通常没有可学习偏置？
- RMSNorm 和 LayerNorm 的公式有什么区别？
- 输入全为 0 时为什么不会产生 `NaN`？
- RMSNorm 在 Pre-Norm Transformer Block 的什么位置？
