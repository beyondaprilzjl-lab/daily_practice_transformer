# Day 07：KV Cache

## 今日目标

实现自回归推理中的 KV Cache，并理解：

1. 为什么缓存 `K` 和 `V`；
2. 为什么通常不缓存 `Q`；
3. `prefill` 和 `decode` 有什么区别；
4. Cache 如何随生成长度增长；
5. KV Cache 节省了什么计算，又增加了什么内存。

今天采用面试版假设：

```text
cache=None：输入完整 prompt，执行 prefill
cache!=None：每次只输入一个新 token，执行 decode
```

## 文件说明

```text
day07_kv_cache/
├── README.md       # 公式、Shape、复杂度与面试追问
├── exercise.py     # KV Cache 练习骨架 + 文件底部示例
└── solution.py     # 完整实现 + 相同验证示例
```

## 1. 没有 KV Cache 时的问题

假设已经生成：

```text
x1, x2, ..., xt
```

为了生成下一个 token，如果每次都把整个序列重新送入模型，就会重复计算
历史 token 的：

```text
K = x Wk
V = x Wv
```

历史 token 没有变化，因此它们对应的 `K`、`V` 也不会变化。

KV Cache 的核心思想就是：

```text
历史 K、V 只计算一次，之后直接复用。
```

## 2. 单步 Decode 公式

当前输入只有新 token：

```text
x_t: [B, 1, D]
```

只计算当前 token 的投影：

```text
q_t = x_t Wq
k_t = x_t Wk
v_t = x_t Wv
```

将新的 `k_t`、`v_t` 追加到历史 Cache：

```text
K_1:t = concat(K_1:t-1, k_t)
V_1:t = concat(V_1:t-1, v_t)
```

当前 Query 与所有历史 Key 计算注意力：

```text
scores_t = q_t K_1:t^T / sqrt(Dh)
p_t = softmax(scores_t)
o_t = p_t V_1:t
```

只需要计算新 token 的输出，不再重新计算旧 token 的输出。

## 3. 为什么缓存 K 和 V

Attention：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(Dh))V
```

生成当前 token 时：

- 当前 `Q` 要查询所有历史位置；
- 历史 `K` 用来和当前 `Q` 计算相关性；
- 历史 `V` 用来根据注意力概率聚合信息。

所以当前步骤仍然需要全部历史 `K` 和 `V`。

如果不缓存，就必须再次从历史 hidden states 计算它们。

## 4. 为什么通常不缓存 Q

每个位置的 `Q` 只在该位置产生输出时使用一次。

例如生成第 `t` 个 token 时，只需要：

```text
q_t
```

未来生成第 `t+1` 个 token 时，会使用新的：

```text
q_t+1
```

旧的 `q_t` 不会再次参与计算，所以通常不需要缓存。

## 5. Cache 的 Shape

当前输入：

```text
x: [B, T, D]
```

拆分多头后：

```text
k: [B, H, T, Dh]
v: [B, H, T, Dh]
```

其中：

```text
D = H * Dh
```

假设 Cache 已经保存 `P` 个历史 token：

```text
old_k: [B, H, P, Dh]
old_v: [B, H, P, Dh]
```

当前 decode 输入一个 token：

```text
k: [B, H, 1, Dh]
v: [B, H, 1, Dh]
```

沿序列维 `-2` 拼接：

```python
k = torch.cat((old_k, k), dim=-2)
v = torch.cat((old_v, v), dim=-2)
```

新 Cache：

```text
k: [B, H, P + 1, Dh]
v: [B, H, P + 1, Dh]
```

## 6. 为什么沿 dim=-2 拼接

多头张量形状为：

```text
[B, H, T, Dh]
```

维度编号：

```text
 0  1  2   3
 B  H  T  Dh
```

`-1` 是 `Dh`，`-2` 是序列长度 `T`。

KV Cache 要增加历史 token 数量，所以沿 `T` 拼接：

```python
dim=-2
```

不能沿 Head 或特征维拼接。

## 7. Prefill

Prefill 一次处理完整 prompt：

```text
x: [B, S, D]
```

此时要使用 causal mask：

```python
mask = torch.tril(torch.ones(S, S, dtype=torch.bool))
```

每个 prompt token 只能看到自己和之前的位置。

Prefill 同时产生：

```text
输出：[B, S, D]
K Cache：[B, H, S, Dh]
V Cache：[B, H, S, Dh]
```

## 8. Decode

Decode 每次输入一个新 token：

```text
x: [B, 1, D]
```

把新 `K`、`V` 追加到 Cache 后：

```text
q: [B, H, 1, Dh]
k: [B, H, S + 1, Dh]
v: [B, H, S + 1, Dh]
```

注意力分数：

```text
scores: [B, H, 1, S + 1]
```

当前 Cache 中只有过去和当前位置，没有未来位置，因此单 token decode 不需要
额外的 causal mask。

如果一次 decode 多个新 token，则仍然需要构造带位置偏移的 causal mask。
本日面试版不处理这种情况。

## 9. 面试核心代码

```python
q = split_heads(self.q(x), self.h)
k = split_heads(self.k(x), self.h)
v = split_heads(self.v(x), self.h)

if cache is None:
    s = x.size(1)
    mask = torch.tril(
        torch.ones(s, s, dtype=torch.bool, device=x.device)
    )
else:
    old_k, old_v = cache
    k = torch.cat((old_k, k), dim=-2)
    v = torch.cat((old_v, v), dim=-2)
    mask = None

scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))

if mask is not None:
    scores = scores.masked_fill(~mask, float("-inf"))

p = torch.softmax(scores, dim=-1)
out = self.out(merge_heads(p @ v))
return out, (k, v)
```

真正新增的逻辑只有：

```text
1. 接收旧 cache
2. 沿序列维拼接新 K、V
3. 返回更新后的 cache
```

## 10. 完整计算和缓存计算为什么应一致

完整 causal attention 中，最后一个 token 只能看到：

```text
token 1, token 2, ..., token t
```

Cached decode 中，最后一个 token 的 Cache 也正好包含：

```text
token 1, token 2, ..., token t
```

两种方式使用相同的 `Q`、`K`、`V` 和权重，因此最后一个 token 的输出应该
一致。

文件底部示例会比较：

```text
完整四 token 计算的最后一个输出

前三个 token prefill
+ 第四个 token cached decode 的输出
```

## 11. 复杂度

生成到第 `t` 个 token 时，不使用 Cache 并重新计算整个前缀，Attention
大约需要：

```text
O(t^2 * D)
```

使用 KV Cache 后，只计算当前 Query 和 `t` 个 Key：

```text
O(t * D)
```

因此 KV Cache 将单步 Attention 从前缀长度的平方级计算，降低到线性级。

但它没有让完整生成过程变成常数复杂度。随着序列增长，每个新 Query 仍然
需要读取所有历史 K、V。

## 12. Cache 的内存开销

每层需要缓存 K 和 V：

```text
K：[B, H, S, Dh]
V：[B, H, S, Dh]
```

忽略元素字节数，单层总元素量：

```text
2 * B * H * S * Dh
= 2 * B * S * D
```

如果模型有 `L` 层：

```text
总 Cache 元素量 = 2 * L * B * S * D
```

因此 KV Cache 用更多显存换取更少的重复计算。

## 13. KV Cache 和 RoPE

真实模型通常先对当前 `Q`、`K` 应用对应位置的 RoPE：

```text
q_t = RoPE(q_t, position=t)
k_t = RoPE(k_t, position=t)
```

再将旋转后的 `k_t` 写入 Cache。

通常缓存的是已经应用位置信息的 Key，而不是每一步重新旋转所有历史 Key。

## 14. 训练时为什么通常不用 KV Cache

训练时整个序列通常可以并行计算：

```text
[B, S, D]
```

并且反向传播需要保存计算图。

KV Cache 主要用于自回归推理，因为推理是逐 token 生成，历史 K、V 会被
反复使用。

## 15. 常见错误

### 错误一：沿错误维度拼接

```python
torch.cat((old_k, k), dim=-1)
```

`-1` 是 `Dh`，正确的序列维是 `-2`。

### 错误二：只缓存 K

计算 `p @ v` 时仍然需要所有历史 V，所以 K 和 V 都要缓存。

### 错误三：Prefill 忘记 causal mask

整段 prompt 并行计算时，前面的 token 不能看到后面的 token。

### 错误四：每一步仍传入完整前缀

有 Cache 后，decode 应只传入新 token，否则仍然会重复计算。

### 错误五：不同层共用同一份 Cache

每个 Transformer 层产生的 K、V 不同，因此每层都有自己的 KV Cache。

## 16. 面试追问

### Q1：KV Cache 缓存的是什么？

每一层 Attention 中，已经计算过的 Key 和 Value。

### Q2：KV Cache 的代价是什么？

显存占用随 batch、层数和序列长度线性增长。

### Q3：KV Cache 会减少模型参数量吗？

不会。它保存的是推理过程中的中间张量，不是模型参数。

### Q4：为什么 Cache 长度会不断增加？

每生成一个新 token，就会追加一组新的 K 和 V。

### Q5：Beam Search 时 Cache 怎么处理？

不同 beam 选择不同历史路径，因此需要根据 beam 索引复制或重新排列
对应的 Cache。

## 17. 运行方式

完成练习：

```bash
python3 day07_kv_cache/exercise.py
```

运行参考实现：

```bash
python3 day07_kv_cache/solution.py
```

预期看到：

```text
prefill cache shape: [1, 2, 3, 4]
decode cache shape:  [1, 2, 4, 4]
matches full output: True
```
