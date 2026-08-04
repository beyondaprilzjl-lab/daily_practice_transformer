# Day 02：多头自注意力（MHA）

## 今日目标

使用 PyTorch 从零实现：

1. 拆分和合并多个注意力头；
2. 缩放点积注意力；
3. 多头自注意力 `MHA`。

今天的重点是能够在面试现场写清楚张量形状，而不是实现生产级
`MultiheadAttention`。

## 文件说明

```text
day02_mha/
├── README.md              # 基础版公式推导与形状说明
├── exercise.py            # 基础版练习骨架
├── solution.py            # 基础版完整实现
├── UPGRADE.md             # 完整 Attention 升级说明
├── upgrade_exercise.py    # 升级版练习骨架
└── upgrade_solution.py    # 升级版完整实现
```

## 1. 自注意力公式

输入：

```text
X: [B, S, D]
```

其中：

- `B`：batch size；
- `S`：序列长度；
- `D`：模型维度；
- `H`：注意力头数；
- `Dh = D / H`：每个头的维度。

先对同一个输入做三次线性投影：

```text
Q = XWq
K = XWk
V = XWv
```

然后计算每个头：

```text
Attention(Q, K, V)

                   QK^T
= softmax(--------------------) V
                 sqrt(Dh)
```

多个头的结果拼接后再经过输出投影：

```text
MHA(X) = Concat(head_1, ..., head_H) Wo
```

## 2. 为什么要除以 sqrt(Dh)

`Q` 和 `K` 的点积包含 `Dh` 项。假设每一项均值为 0、方差为 1，
点积的方差会随 `Dh` 增大。

当分数过大时，Softmax 会变得非常尖锐，梯度也容易变小。除以
`sqrt(Dh)` 后，可以把分数的尺度控制在更稳定的范围。

注意缩放的是每个头的维度 `Dh`，不是整个模型维度 `D`。

## 3. 拆分注意力头

线性投影后的 `Q`、`K`、`V` 形状都是：

```text
[B, S, D]
```

将最后一维拆成 `H` 个头：

```text
[B, S, D]
    reshape
[B, S, H, Dh]
    transpose
[B, H, S, Dh]
```

代码核心：

```python
x.reshape(b, s, h, dh).transpose(1, 2)
```

把 `H` 放到 `S` 前面后，每个头都可以独立计算一个 `[S, S]`
注意力矩阵，同时仍然使用一次批量矩阵乘法，不需要写循环。

## 4. 注意力分数的形状

每个头中：

```text
Q:                    [B, H, S, Dh]
K.transpose(-2, -1):  [B, H, Dh, S]
Q @ K^T:              [B, H, S, S]
```

最后两个 `S` 的含义不同：

- 倒数第二维：当前正在查询的 token；
- 最后一维：它可以关注的 token。

因此 Softmax 必须沿最后一维计算：

```python
attn = torch.softmax(scores, dim=-1)
```

这样每个 token 对所有可关注位置的概率之和为 1。

## 5. Causal mask

生成式模型中，当前位置不能看到未来 token。长度为 3 时，mask 为：

```text
[
  [True,  False, False],
  [True,  True,  False],
  [True,  True,  True ],
]
```

在 Softmax 之前，把不允许关注的位置改成负无穷：

```python
scores = scores.masked_fill(~mask, float("-inf"))
```

因为：

```text
exp(-inf) = 0
```

所以这些位置经过 Softmax 后概率为 0。

mask 必须在 Softmax 之前应用。如果先计算 Softmax，再把某些概率改成
0，剩余概率之和将不再等于 1。

## 6. 合并注意力头

每个头计算完成后的形状：

```text
[B, H, S, Dh]
```

合并过程与拆分过程相反：

```text
[B, H, S, Dh]
    transpose
[B, S, H, Dh]
    contiguous + view
[B, S, D]
```

代码核心：

```python
x.transpose(1, 2).contiguous().view(b, s, h * dh)
```

`transpose` 只改变张量的观察顺序，内存不一定连续。调用 `view` 前使用
`contiguous()`，让元素按照新的维度顺序连续存放。

## 7. 完整形状变化

```text
x                         [B, S, D]
q_proj / k_proj / v_proj  [B, S, D]
split_heads               [B, H, S, Dh]
scores                    [B, H, S, S]
attn                      [B, H, S, S]
attn @ v                  [B, H, S, Dh]
merge_heads               [B, S, D]
out_proj                  [B, S, D]
```

## 8. 面试时的核心实现

现场只需要完成四个步骤：

```text
1. Q、K、V 线性投影并拆头
2. scores = QK^T / sqrt(Dh)
3. mask -> softmax -> attn @ V
4. 合并多头并执行输出投影
```

默认可以先向面试官说明：

```text
x 的形状是 [B, S, D]
D 可以被 H 整除
mask 是布尔矩阵，True 表示允许关注
mask 的每一行至少存在一个 True
```

因此练习代码只保留 `D % H == 0` 这一项必要检查，不编写大量输入校验。

## 9. 多头为什么不是重复计算

每个头接收的是投影后不同位置的特征：

```text
Q_i, K_i, V_i: [B, S, Dh]
```

`Wq`、`Wk` 和 `Wv` 会学习不同的表示子空间，因此不同头可以学习不同
关系，例如局部关系、长距离关系或不同语义特征。

## 10. 复杂度

注意力分数和加权求和的时间复杂度：

```text
O(B * S^2 * D)
```

注意力矩阵的空间复杂度：

```text
O(B * H * S^2)
```

序列长度 `S` 的平方项，是标准注意力处理长文本时的主要瓶颈。

## 11. 面试常见追问

- 为什么 Softmax 使用 `dim=-1`？
- 为什么缩放因子是 `sqrt(Dh)`？
- mask 为什么要在 Softmax 之前应用？
- `transpose` 后为什么要调用 `contiguous()`？
- 多头注意力为什么可以学习不同关系？
- 哪个张量带来了 `S^2` 的显存开销？
- self-attention 和 cross-attention 的 `Q`、`K`、`V` 来源有什么不同？

## 12. 独立升级版

完成基础版后，再练习：

```bash
python3 upgrade_exercise.py
```

升级版单独保存在 `upgrade_exercise.py` 和 `upgrade_solution.py`，不会增加
基础版的现场书写负担。

升级版覆盖：

- Batch；
- Padding Mask 与 Causal Mask 的组合；
- Self Attention 与 Cross Attention；
- 手写数值稳定 Softmax；
- `Tq`、`Tk` 不同时的 Shape 与复杂度。

详细推导见 `UPGRADE.md`。
