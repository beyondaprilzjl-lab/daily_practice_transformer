# Day 03：旋转位置编码（RoPE）

## 今日目标

使用 PyTorch 从零实现：

1. `rotate_half`；
2. RoPE 的 `cos`、`sin` 缓存；
3. 将 RoPE 应用到多头注意力的 `Q` 和 `K`。

今天的重点是从二维旋转公式推导代码，并理解 RoPE 为什么能让注意力分数
包含相对位置信息。

## 文件说明

```text
day03_rope/
├── README.md       # 公式推导、形状说明与面试追问
├── exercise.py     # 练习骨架 + 文件底部验证示例
└── solution.py     # 完整参考实现 + 相同验证示例
```

## 1. 为什么需要位置编码

自注意力根据 token 之间的内容计算相关性，但它本身不知道 token 的先后
顺序。

如果不加入位置信息，交换两个 token 的位置，只会交换对应输出，模型无法
区分：

```text
我喜欢你
你喜欢我
```

RoPE 不把位置向量直接加到输入上，而是根据 token 的位置旋转 `Q` 和
`K`。

## 2. 二维旋转公式

给定二维向量：

```text
x = [x1, x2]
```

旋转角度 `theta` 后：

```text
        [cos(theta)  -sin(theta)] [x1]
R(x) = [sin(theta)   cos(theta)] [x2]
```

展开：

```text
y1 = x1 * cos(theta) - x2 * sin(theta)
y2 = x1 * sin(theta) + x2 * cos(theta)
```

RoPE 将特征维度两两配对：

```text
(x0, x1), (x2, x3), ..., (x[D-2], x[D-1])
```

因此每个注意力头的维度 `Dh` 必须是偶数。

## 3. rotate_half

对于每一对特征：

```text
[x1, x2] -> [-x2, x1]
```

于是二维旋转可以改写为：

```text
R(x) = x * cos(theta) + rotate_half(x) * sin(theta)
```

验证第一项：

```text
x1 * cos(theta) + (-x2) * sin(theta)
= x1 * cos(theta) - x2 * sin(theta)
```

验证第二项：

```text
x2 * cos(theta) + x1 * sin(theta)
= x1 * sin(theta) + x2 * cos(theta)
```

代码中先取出偶数位和奇数位：

```python
x1 = x[..., 0::2]
x2 = x[..., 1::2]
```

再把 `[-x2, x1]` 交错排列回原来的特征顺序。

## 4. 不同维度使用不同频率

第 `i` 对特征的基础频率为：

```text
freq_i = base^(-2i / Dh)
```

通常：

```text
base = 10000
```

位置 `m` 对应的旋转角度：

```text
theta_(m,i) = m * freq_i
```

代码中：

```python
idx = [0, 2, 4, ..., Dh - 2]
freq = base ** (-idx / Dh)
theta = outer(position, freq)
```

不同维度使用不同旋转速度：

- 前面的维度频率较高，旋转较快；
- 后面的维度频率较低，旋转较慢；
- 多种频率组合后，可以表示不同距离的位置关系。

## 5. cos 和 sin 的形状

假设：

```text
Q, K: [B, H, S, Dh]
```

`theta` 最初的形状为：

```text
[S, Dh / 2]
```

因为一对特征共享同一个角度，所以将每个角度重复两次：

```text
[theta_0, theta_0, theta_1, theta_1, ...]
```

最终：

```text
cos, sin: [S, Dh]
```

它们与 `[B, H, S, Dh]` 相乘时，PyTorch 会自动在 `B` 和 `H`
维度广播。

注意：这里的 `dim` 是每个头的维度 `Dh`，不是整个模型维度 `D`。

## 6. 为什么 RoPE 表示相对位置

位置 `m` 的 `Q` 和位置 `n` 的 `K` 分别旋转：

```text
q_m' = R_m q
k_n' = R_n k
```

它们的点积：

```text
(R_m q)^T (R_n k)

= q^T R_m^T R_n k

= q^T R_(n-m) k
```

最终只与位置差：

```text
n - m
```

有关。这就是 RoPE 的核心：输入使用绝对位置进行旋转，`QK^T` 的点积
自然得到相对位置信息。

## 7. 为什么只旋转 Q 和 K

注意力分数来自：

```text
QK^T
```

RoPE 的目标是让这个分数包含 token 之间的相对位置，因此旋转 `Q` 和
`K` 即可。

`V` 表示被加权汇总的内容，不参与注意力分数计算，通常不应用 RoPE。

## 8. 放在 Attention 的哪个位置

结合 Day 02 的多头注意力：

```text
x
-> q_proj / k_proj / v_proj
-> split_heads
-> 对 Q、K 应用 RoPE
-> QK^T / sqrt(Dh)
-> mask
-> softmax
-> attn @ V
```

核心代码：

```python
cos, sin = rope_cache(s, dh, device=q.device)
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)
```

## 9. 面试时的核心实现

现场只需要完成三个步骤：

```text
1. 每两个特征组成一个二维向量
2. 根据 position 和 freq 构造 cos、sin
3. x * cos + rotate_half(x) * sin
```

默认可以先向面试官说明：

```text
Q、K 的形状为 [B, H, S, Dh]
Dh 是偶数
Q 和 K 使用相同的位置与频率
本题不考虑 KV Cache 和 position offset
```

## 10. 一个重要性质

旋转不会改变向量长度：

```text
||R(x)|| = ||x||
```

因为旋转矩阵满足：

```text
R^T R = I
```

这也是文件底部示例验证的性质之一。

## 11. 复杂度

对形状 `[B, H, S, Dh]` 的 `Q` 或 `K`：

```text
时间复杂度：O(B * H * S * Dh)
额外缓存：  O(S * Dh)
```

RoPE 不会产生 `[S, S]` 矩阵，因此它不是注意力平方复杂度的来源。

## 12. 面试常见追问

- 为什么 `Dh` 必须是偶数？
- `rotate_half` 对每一对特征做了什么？
- 为什么同一对特征共享一个旋转角度？
- 为什么位置 0 的向量保持不变？
- 为什么 RoPE 只应用到 `Q` 和 `K`？
- RoPE 如何让点积依赖相对位置 `n - m`？
- 增量推理使用 KV Cache 时，新的 token 应该使用哪个 position？
- 调整 `base` 会怎样影响不同频率的旋转速度？
