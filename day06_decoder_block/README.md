# Day 06：Pre-Norm Decoder Block

## 今日目标

把前面练习过的组件组合成一个 Decoder Block：

```text
RMSNorm -> Causal MHA -> Residual
RMSNorm -> SwiGLU    -> Residual
```

今天不要求重新手写所有基础组件，重点是能够在面试现场正确写出 Block 的
数据流，并说明每一步的 Shape。

## 文件说明

```text
day06_decoder_block/
├── README.md       # 公式、数据流、Shape 与面试追问
├── exercise.py     # 已提供基础组件，只练习 DecoderBlock
└── solution.py     # 完整实现 + 文件底部验证示例
```

## 1. Block 公式

输入：

```text
x: [B, S, D]
```

第一部分是因果自注意力：

```text
h = x + MHA(RMSNorm(x), mask)
```

第二部分是前馈网络：

```text
y = h + SwiGLU(RMSNorm(h))
```

完整公式：

```text
h = x + Attention(Norm(x))
y = h + FFN(Norm(h))
```

对应代码：

```python
a, attn = self.attn(self.norm1(x), mask)
x = x + a
x = x + self.ffn(self.norm2(x))
return x, attn
```

## 2. 为什么叫 Pre-Norm

Pre-Norm 表示先归一化，再进入子层：

```text
x + Sublayer(Norm(x))
```

当前 Block 中有两个子层：

```text
Attention
SwiGLU FFN
```

所以需要两个独立的 RMSNorm：

```python
self.norm1 = RMSNorm(d_model)
self.norm2 = RMSNorm(d_model)
```

Post-Norm 则是先做残差相加，再归一化：

```text
Norm(x + Sublayer(x))
```

这两种结构的数据流不同，面试时不要混写。

## 3. 第一条残差路径

先归一化：

```text
n1 = RMSNorm(x)
```

再计算因果自注意力：

```text
a = MHA(n1, mask)
```

最后与原始输入相加：

```text
h = x + a
```

Shape 始终为：

```text
x:  [B, S, D]
n1: [B, S, D]
a:  [B, S, D]
h:  [B, S, D]
```

Attention 必须输出 `D` 维，才能与残差 `x` 相加。

## 4. 第二条残差路径

第二个 RMSNorm 的输入必须是更新后的 `h`：

```text
n2 = RMSNorm(h)
f = SwiGLU(n2)
y = h + f
```

Shape：

```text
h:  [B, S, D]
n2: [B, S, D]
f:  [B, S, D]
y:  [B, S, D]
```

常见错误是再次使用原始 `x`：

```python
x = x + self.ffn(self.norm2(original_x))
```

正确做法是让 FFN 接收 Attention 残差更新后的结果。

## 5. Causal Mask

对长度为 `S` 的序列：

```python
mask = torch.tril(torch.ones(S, S, dtype=torch.bool))
```

例如 `S = 4`：

```text
True  False False False
True  True  False False
True  True  True  False
True  True  True  True
```

第 `i` 个 token 只能看到自己和之前的 token，不能看到未来 token。

在 Attention 中：

```python
scores = scores.masked_fill(~mask, float("-inf"))
```

`[S, S]` 的 mask 会广播到注意力分数：

```text
scores: [B, H, S, S]
mask:         [S, S]
```

## 6. 为什么需要残差连接

如果只写：

```text
x = Attention(Norm(x))
```

原始表示会被子层完全替换。

残差连接：

```text
x = x + Attention(Norm(x))
```

为信息和梯度提供了一条直接路径，也要求子层的输入、输出 Shape 相同。

## 7. Block 中各组件负责什么

```text
RMSNorm：控制每个 token 特征向量的尺度
MHA：    让不同 token 交换信息
SwiGLU：独立变换每个 token 的特征
Residual：保留原始信息并帮助梯度传播
Mask：   阻止当前位置看到未来 token
```

Attention 负责 token 之间的信息混合，FFN 负责每个 token 内部的特征变换。

## 8. RoPE 放在哪里

本日代码复用 Day 02 的最小 MHA，因此没有重复加入 RoPE。

真实模型中，RoPE 一般放在 Attention 内部：

```text
q = split_heads(q_proj(x))
k = split_heads(k_proj(x))

q = apply_rope(q)
k = apply_rope(k)

scores = q @ k^T
```

也就是说，RoPE 作用于 `Q` 和 `K`，不是写在两条残差连接之间。

## 9. Shape 总结

```text
x                         [B, S, D]
norm1(x)                  [B, S, D]
split q, k, v             [B, H, S, Dh]
attention scores          [B, H, S, S]
attention output          [B, S, D]
first residual            [B, S, D]
norm2(x)                  [B, S, D]
SwiGLU hidden             [B, S, F]
SwiGLU output             [B, S, D]
second residual / output  [B, S, D]
```

其中：

```text
D = H * Dh
```

## 10. 复杂度

Attention：

```text
线性投影：O(B * S * D^2)
注意力：  O(B * S^2 * D)
```

SwiGLU：

```text
O(B * S * D * F)
```

整个 Block：

```text
O(B * S * D^2 + B * S^2 * D + B * S * D * F)
```

注意力矩阵的空间复杂度：

```text
O(B * H * S^2)
```

## 11. 面试时需要写的核心

如果面试官已经提供 `MHA`、`RMSNorm` 和 `SwiGLU`，现场核心代码只有：

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MHA(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, mask=None):
        a, attn = self.attn(self.norm1(x), mask)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x, attn
```

可以先向面试官说明：

```text
采用 Pre-Norm 结构
mask 中 True 表示允许关注
Attention 和 FFN 的输出都保持 [B, S, D]
暂不加入 dropout、KV Cache 和混合精度细节
```

## 12. 常见错误

### 错误一：忘记残差连接

```python
x, attn = self.attn(self.norm1(x), mask)
x = self.ffn(self.norm2(x))
```

这样会丢失两条残差路径。

### 错误二：第二个 Norm 用错输入

第二个 Norm 应处理 Attention 更新后的 `x`。

### 错误三：在残差相加前没有降回 D

MHA 和 SwiGLU 最终都必须输出 `[B, S, D]`。

### 错误四：两个子层共用同一个 Norm

标准 Block 中两个 RMSNorm 有各自独立的可学习参数。

## 13. 面试追问

### Q1：为什么一个 Block 需要两个 Norm？

Attention 和 FFN 是两个不同子层，每个子层前分别归一化，并拥有独立的
缩放参数。

### Q2：为什么 FFN 不直接处理原始 x？

FFN 应继续处理 Attention 已经融合上下文后的表示。

### Q3：为什么 Decoder 使用 causal mask？

自回归生成时，当前位置只能依赖已经出现的 token，不能提前看到未来答案。

### Q4：为什么 Block 的输入和输出 Shape 相同？

多个 Block 需要连续堆叠，同时两个子层都要与主干做残差相加。

## 14. 运行方式

完成练习：

```bash
python3 day06_decoder_block/exercise.py
```

运行参考实现：

```bash
python3 day06_decoder_block/solution.py
```

文件底部示例会检查：

```text
输入输出 Shape 是否一致
未来位置的注意力概率是否为 0
梯度是否能够回传到输入
```
