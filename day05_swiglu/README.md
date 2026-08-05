# Day 05：SwiGLU 前馈网络

## 今日目标

使用 PyTorch 从零实现：

1. `SiLU` 激活函数；
2. Transformer 中常用的 `SwiGLU` 前馈网络。

今天的重点是理解为什么 SwiGLU 需要两条升维分支，以及逐元素乘法如何
形成门控。

## 文件说明

```text
day05_swiglu/
├── README.md       # 公式推导、形状说明与面试追问
├── exercise.py     # 练习骨架 + 文件底部验证示例
└── solution.py     # 完整参考实现 + 相同验证示例
```

## 1. SiLU 公式

Sigmoid：

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

SiLU，也叫 Swish：

```text
SiLU(x) = x * sigmoid(x)
```

对应代码：

```python
return x * torch.sigmoid(x)
```

SiLU 不像 ReLU 那样把所有负数直接变成 0，它是一条平滑曲线。

## 2. 普通 FFN

Transformer 中普通的两层前馈网络可以写成：

```text
h = activation(x W1)
y = h W2
```

形状变化：

```text
x: [B, S, D]
h: [B, S, F]
y: [B, S, D]
```

其中：

- `D` 是模型维度 `d_model`；
- `F` 是中间维度 `d_ff`；
- 每个 token 独立经过同一个 FFN。

## 3. SwiGLU 公式

SwiGLU 使用两条从 `D` 升到 `F` 的分支：

```text
g = x W_gate
u = x W_up
```

将门控分支经过 SiLU，再和另一条分支逐元素相乘：

```text
h = SiLU(g) * u
```

最后降回模型维度：

```text
y = h W_down
```

完整公式：

```text
SwiGLU(x) = (SiLU(x W_gate) * (x W_up)) W_down
```

面试代码：

```python
g = silu(self.gate(x))
u = self.up(x)
return self.down(g * u)
```

## 4. 为什么需要两条分支

`u` 分支保存要传递的特征：

```text
u = x W_up
```

`g` 分支生成一个与 `u` 同形状的门：

```text
g = SiLU(x W_gate)
```

然后逐元素控制 `u`：

```text
h = g * u
```

因此可以把它理解为：

```text
一条分支提供内容，另一条分支决定每个特征通过多少。
```

两条分支必须都是 `[B, S, F]`，才能逐元素相乘。

## 5. Shape 变化

输入：

```text
x: [B, S, D]
```

两条升维分支：

```text
gate(x): [B, S, F]
up(x):   [B, S, F]
```

门控相乘：

```text
g * u: [B, S, F]
```

降维：

```text
down(g * u): [B, S, D]
```

因此 SwiGLU 不改变输入和输出的整体形状：

```text
[B, S, D] -> [B, S, D]
```

## 6. 为什么乘法是逐元素乘法

代码使用：

```python
g * u
```

这里的 `*` 是逐元素乘法：

```text
[B, S, F] * [B, S, F] -> [B, S, F]
```

它不是矩阵乘法。矩阵乘法使用 `@`，会在某个维度上做乘加。

门控需要让 `g` 中的每个值控制 `u` 中对应位置的值，所以使用 `*`。

## 7. 参数量

忽略 bias：

```text
gate: D * F
up:   D * F
down: F * D
```

总参数量：

```text
3 * D * F
```

普通两层 FFN 的参数量约为：

```text
2 * D * F
```

因此在参数量接近的情况下，SwiGLU 的 `F` 通常会比普通 FFN 的中间维度
更小。

例如普通 FFN 使用 `F = 4D` 时：

```text
普通 FFN 参数量 = 2 * D * 4D = 8D^2
```

若让 SwiGLU 参数量接近：

```text
3 * D * F = 8D^2

F ≈ 8D / 3
```

实际模型还会将 `F` 调整为适合硬件计算的整数倍。

## 8. 复杂度

对输入 `[B, S, D]` 和中间维度 `F`：

```text
时间复杂度：O(B * S * D * F)
中间激活：  O(B * S * F)
```

这里省略了三次线性层带来的常数。

FFN 对每个 token 独立计算，所以不会像标准 Attention 那样产生
`S * S` 的注意力矩阵。

## 9. 面试时的核心实现

现场只需要完成：

```text
1. 手写 SiLU
2. 创建 gate、up、down 三个线性层
3. 两条分支升维
4. SiLU(gate) 和 up 逐元素相乘
5. down 投影回 D
```

默认可以先向面试官说明：

```text
x 的形状是 [B, S, D]
最后一维等于 d_model
本题不处理 dropout 和混合精度等工程细节
```

## 10. 常见错误

### 错误一：两条分支共用一个线性层

```python
g = self.up(x)
u = self.up(x)
```

这样两条分支没有独立参数，不是标准 SwiGLU。

### 错误二：使用矩阵乘法

```python
g @ u
```

门控要求对应位置相乘，应使用：

```python
g * u
```

### 错误三：忘记降维

门控后的形状是 `[B, S, F]`，还需要经过 `down` 回到
`[B, S, D]`，才能用于残差连接。

## 11. 面试追问

### Q1：SwiGLU 中谁是门，谁是内容？

`SiLU(gate(x))` 是门，`up(x)` 是被控制的内容。

### Q2：为什么输出还要经过 down？

门控后的最后一维是 `F`，需要投影回 `D`，才能与 Transformer 主干中的
残差 `x` 相加。

### Q3：为什么 FFN 不混合不同 token？

线性层只作用于最后一个特征维度，同一组参数独立应用到每个 token。
序列中的 token 交互主要由 Attention 完成。

### Q4：SwiGLU 和普通 ReLU FFN 的主要区别是什么？

普通 FFN 只有一条升维分支；SwiGLU 使用两条升维分支，并通过逐元素乘法
进行门控。

## 12. 运行方式

完成练习后运行：

```bash
python3 day05_swiglu/exercise.py
```

对照参考实现：

```bash
python3 day05_swiglu/solution.py
```

示例将三个线性层设置为单位矩阵，因此可以直接验证：

```text
output = SiLU(x) * x
```
