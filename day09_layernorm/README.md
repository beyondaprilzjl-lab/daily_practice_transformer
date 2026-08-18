# Day 09：LayerNorm

## 今日目标

使用 PyTorch 从零实现：

1. 函数版 `layer_norm`；
2. 带可学习缩放和偏置的 `LayerNorm` 模块；
3. 使用文件底部示例与 `torch.nn.LayerNorm` 对照。

面试时需要说清楚 LayerNorm 的均值、方差、张量形状，以及它和
RMSNorm 的区别。

## 文件说明

```text
day09_layernorm/
├── README.md       # 公式推导、形状说明与面试追问
├── exercise.py     # 练习骨架 + 文件底部验证示例
└── solution.py     # 完整参考实现 + 相同验证示例
```

## 1. LayerNorm 公式

给定一个 `D` 维向量：

```text
x = [x1, x2, ..., xD]
```

先计算均值：

```text
       1
mu = ----- sum_i xi
       D
```

再计算方差：

```text
        1
var = ----- sum_i (xi - mu)^2
        D
```

归一化：

```text
             x - mu
x_hat = ----------------
          sqrt(var + eps)
```

最后使用可学习参数进行缩放和平移：

```text
y = gamma * x_hat + beta
```

完整公式：

```text
                     x - mean(x)
LayerNorm(x) = gamma * -------------------- + beta
                     sqrt(var(x) + eps)
```

## 2. 公式如何变成代码

```python
mean = x.mean(dim=-1, keepdim=True)
centered = x - mean
var = centered.pow(2).mean(dim=-1, keepdim=True)
x_hat = centered / torch.sqrt(var + eps)
return x_hat * weight + bias
```

对应关系：

```text
mean      -> mu
centered  -> x - mu
var       -> mean((x - mu)^2)
weight    -> gamma
bias      -> beta
```

## 3. 为什么沿最后一维计算

Transformer 的输入通常为：

```text
x: [B, S, D]
```

LayerNorm 对每个样本、每个 token 的 `D` 个特征单独归一化：

```text
输入：   [B, S, D]
mean：   [B, S, 1]
var：    [B, S, 1]
weight： [D]
bias：   [D]
输出：   [B, S, D]
```

因此使用：

```python
x.mean(dim=-1, keepdim=True)
```

- `dim=-1`：沿最后一个特征维 `D` 计算；
- `keepdim=True`：保留大小为 1 的维度，方便广播；
- 不同 batch、不同 token 之间不会互相影响。

如果沿 `dim=1` 计算，就会把不同 token 混在一起，不再是这里的
LayerNorm。

## 4. 为什么方差除以 D

LayerNorm 使用总体方差：

```text
var = sum((x - mean)^2) / D
```

而不是样本方差：

```text
sum((x - mean)^2) / (D - 1)
```

如果用 `torch.var`，需要写：

```python
x.var(dim=-1, keepdim=True, unbiased=False)
```

本练习直接写成：

```python
centered.pow(2).mean(dim=-1, keepdim=True)
```

这样更容易从公式推导，也不会混淆 `unbiased` 参数。

## 5. weight 和 bias 的作用

归一化后的结果大致满足零均值、单位方差，但模型不一定希望所有层都固定
保持这种分布，因此加入：

```text
weight: [D]
bias:   [D]
```

它们会广播到 `[B, S, D]`：

```text
[B, S, D] * [D] + [D] -> [B, S, D]
```

初始化为：

```python
weight = torch.ones(dim)
bias = torch.zeros(dim)
```

此时模块一开始只做标准化，不会额外缩放或平移结果。

## 6. 为什么需要 eps

如果一个 token 的所有特征都相同：

```text
x = [2, 2, 2, 2]
```

那么：

```text
x - mean = 0
var = 0
```

没有 `eps` 时会出现除以 0。正确写法是：

```python
torch.sqrt(var + eps)
```

`eps` 放在开根号内部，保证分母大于 0。

## 7. 和 RMSNorm 的区别

| 对比项 | LayerNorm | RMSNorm |
| --- | --- | --- |
| 减去均值 | 是 | 否 |
| 归一化依据 | 方差 | 均方根 |
| 可学习缩放 `gamma` | 有 | 有 |
| 可学习偏置 `beta` | 通常有 | 通常没有 |
| 输出零均值 | 归一化后近似满足 | 不保证 |

LayerNorm：

```text
             x - mean(x)
gamma * -------------------- + beta
          sqrt(var(x) + eps)
```

RMSNorm：

```text
                 x
gamma * -------------------
          sqrt(mean(x^2) + eps)
```

一句话回答：

```text
LayerNorm 会先中心化再缩放，RMSNorm 只根据均方根缩放。
```

## 8. 在 Transformer 中的位置

Pre-Norm：

```text
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

Post-Norm：

```text
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

二者的区别是归一化发生在子层之前还是残差相加之后。本题只实现
LayerNorm 本身。

## 9. 面试时的核心实现

现场可以按下面六步写：

```text
1. mean
2. centered = x - mean
3. var = mean(centered^2)
4. divide by sqrt(var + eps)
5. multiply weight
6. add bias
```

核心代码：

```python
mean = x.mean(dim=-1, keepdim=True)
centered = x - mean
var = centered.pow(2).mean(dim=-1, keepdim=True)
return centered / torch.sqrt(var + eps) * weight + bias
```

## 10. Shape 和复杂度

输入为 `[B, S, D]` 时：

```text
时间复杂度：O(B * S * D)
额外空间复杂度：O(B * S * D)
参数量：2D
```

额外空间主要来自中心化结果和归一化结果。可学习参数 `weight` 和 `bias`
各有 `D` 个元素。

## 11. 常见错误

### 错误一：忘记减均值

```python
x / torch.sqrt(x.pow(2).mean(...))
```

这是 RMSNorm，不是 LayerNorm。

### 错误二：沿错误维度计算

```python
x.mean(dim=1)
```

这会混合不同 token。Transformer 中通常沿最后一个特征维计算。

### 错误三：使用样本方差

```python
x.var(dim=-1)
```

`torch.var` 的参数行为需要明确。面试现场直接用平方差的均值更清楚。

### 错误四：忘记 `keepdim=True`

均值和方差少一个维度后，不一定能和原张量正确广播。

### 错误五：只有 weight，没有 bias

标准 LayerNorm 通常同时包含可学习的 `weight` 和 `bias`。

## 12. 运行练习

```bash
cd day09_layernorm
python3 exercise.py
```

完成 TODO 后，示例会输出：

- LayerNorm 结果；
- 每个 token 的输出均值；
- 每个 token 的输出方差；
- 是否与 `torch.nn.LayerNorm` 一致。

再运行参考实现：

```bash
python3 solution.py
```

## 面试口述模板

```text
LayerNorm 对每个 token 的最后一个特征维独立计算均值和总体方差。
先减均值得到中心化结果，再除以 sqrt(var + eps)，最后乘可学习的
gamma 并加 beta。输入输出形状都为 [B, S, D]，时间复杂度是
O(BSD)。它和 RMSNorm 的主要区别是 LayerNorm 会减均值，并且通常
还有可学习偏置。
```
