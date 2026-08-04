# Day 02 升级版：完整 Attention

## 升级目标

原版保留最适合首次手写的多头自注意力核心。升级版进一步覆盖：

- Batch；
- 拆分与合并 Head；
- Padding Mask；
- Causal Mask；
- Cross Attention；
- 手写数值稳定 Softmax；
- Self/Cross Attention 的 Shape 和复杂度。

对应文件：

```text
upgrade_exercise.py   # 升级版练习骨架 + 主函数示例
upgrade_solution.py   # 升级版完整实现 + 相同示例
```

## 1. 统一 Self 和 Cross Attention

升级版接口：

```python
forward(q_x, kv_x=None, pad=None, causal=False)
```

### Self Attention

只传入 `q_x`：

```python
out, attn = mha(x, pad=pad, causal=True)
```

此时：

```text
Q、K、V 都来自 x
kv_x = q_x
Tq = Tk = S
```

形状：

```text
x:      [B, S, D]
scores: [B, H, S, S]
out:    [B, S, D]
```

### Cross Attention

分别传入 Query 和 K/V 的来源：

```python
out, attn = mha(q_x, kv_x=memory, pad=memory_pad)
```

通常：

```text
Q 来自 decoder
K、V 来自 encoder 或其他 memory
```

形状：

```text
q_x:    [B, Tq, D]
kv_x:   [B, Tk, D]
Q:      [B, H, Tq, Dh]
K、V:   [B, H, Tk, Dh]
scores: [B, H, Tq, Tk]
out:    [B, Tq, D]
```

Self Attention 是 Cross Attention 公式在 `q_x == kv_x` 时的特殊情况。

## 2. Batch 与 Head

投影后的张量：

```text
[B, T, D]
```

拆分 Head：

```text
[B, T, D]
-> [B, T, H, Dh]
-> [B, H, T, Dh]
```

其中：

```text
Dh = D / H
```

合并 Head：

```text
[B, H, T, Dh]
-> [B, T, H, Dh]
-> [B, T, D]
```

所有计算都保留 Batch 维度，不需要遍历 batch 或 head。

## 3. 数值稳定 Softmax

朴素 Softmax：

```text
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

当分数很大时，`exp(x)` 可能溢出。利用 Softmax 的平移不变性：

```text
softmax(x) = softmax(x - max(x))
```

稳定实现：

```python
x = x - x.max(dim=-1, keepdim=True).values
exp_x = torch.exp(x)
attn = exp_x / exp_x.sum(dim=-1, keepdim=True)
```

Softmax 沿最后一个 Key 维度 `Tk` 计算，因此每个 Query 对所有 Key 的
注意力概率之和为 1。

## 4. Padding Mask

Padding Mask 表示哪些 Key 是真实 token：

```text
pad: [B, Tk]
True  -> 可以关注
False -> padding，禁止关注
```

例如：

```text
[
  [True, True, False, False],
  [True, True, True,  False],
]
```

为了广播到注意力分数：

```python
pad = pad[:, None, None, :]
```

形状变化：

```text
[B, Tk] -> [B, 1, 1, Tk]
```

它会自动广播到：

```text
[B, H, Tq, Tk]
```

Padding Mask 通常屏蔽 `K/V` 中的 padding。对于 padding Query 产生的
输出，一般由后续计算或 loss 忽略。

## 5. Causal Mask

Causal Mask 防止 Query 看到未来 Key：

```python
causal_mask = torch.tril(
    torch.ones(Tq, Tk, dtype=torch.bool)
)
```

增加 Batch 和 Head 广播维：

```text
[Tq, Tk] -> [1, 1, Tq, Tk]
```

在标准 Self Attention 中：

```text
Tq = Tk = S
```

第 `i` 个 token 只能关注位置 `0` 到 `i`。

Cross Attention 通常不需要 causal mask，因为 encoder memory 已经完整
可见，但统一接口仍允许按题目需要传入。

## 6. 组合两个 Mask

Padding Mask：

```text
[B, 1, 1, Tk]
```

Causal Mask：

```text
[1, 1, Tq, Tk]
```

逻辑与后：

```text
[B, 1, Tq, Tk]
```

再广播到所有 Head：

```text
[B, H, Tq, Tk]
```

代码：

```python
mask = pad_mask & causal_mask
scores = scores.masked_fill(~mask, float("-inf"))
```

mask 必须在 Softmax 之前使用。升级版假设每个 Query 至少存在一个允许
关注的 Key，否则整行都是负无穷，Softmax 会产生 `NaN`。

## 7. 完整数据流

```text
q_x [B, Tq, D] ---- q_proj ----> Q [B, H, Tq, Dh]

kv_x [B, Tk, D] --- k_proj ----> K [B, H, Tk, Dh]
                  \- v_proj ----> V [B, H, Tk, Dh]

Q @ K^T / sqrt(Dh)
        |
        v
scores [B, H, Tq, Tk]
        |
Padding Mask + Causal Mask
        |
stable softmax
        |
attn @ V [B, H, Tq, Dh]
        |
merge_heads
        |
out_proj
        |
out [B, Tq, D]
```

## 8. 复杂度口述

### Self Attention

`Tq = Tk = S`。

投影层：

```text
O(B * S * D^2)
```

注意力矩阵计算与加权求和：

```text
O(B * S^2 * D)
```

总时间复杂度：

```text
O(B * S * D^2 + B * S^2 * D)
```

注意力矩阵空间复杂度：

```text
O(B * H * S^2)
```

### Cross Attention

Query 长度为 `Tq`，Key/Value 长度为 `Tk`。

投影层：

```text
O(B * (Tq + 2Tk) * D^2)
```

注意力部分：

```text
O(B * Tq * Tk * D)
```

总时间复杂度：

```text
O(B * (Tq + 2Tk) * D^2 + B * Tq * Tk * D)
```

注意力矩阵：

```text
O(B * H * Tq * Tk)
```

## 9. 面试时的实现顺序

```text
1. split_heads / merge_heads
2. stable_softmax
3. Padding Mask 与 Causal Mask
4. scaled dot-product attention
5. Self/Cross Attention 的 Q、K、V 来源
6. Shape 与复杂度口述
```

建议先完成 Self Attention，再把 `K/V` 的来源改为可选 `kv_x`，即可自然
扩展到 Cross Attention。

## 10. 检查清单

- `Q/K/V` 是否包含 Batch 和 Head 维度？
- 分数是否除以 `sqrt(Dh)`？
- Padding Mask 是否从 `[B, Tk]` 增加两个广播维？
- Causal Mask 是否在 Softmax 前应用？
- 两个 mask 是否使用逻辑与组合？
- Softmax 是否先减去每一行的最大值？
- Cross Attention 的输出长度是否等于 `Tq`？
- Self Attention 是否为 `Tq = Tk = S` 的特殊情况？
