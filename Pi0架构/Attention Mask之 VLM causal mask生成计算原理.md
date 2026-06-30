
# VLM Causal Mask 计算过程完整解析

---

## 1. 这是做什么？

在 `Emu3Pi0` 的 `forward()` 中，VLM 和 Action 的 token 被拼接成一个长序列做联合注意力。为了让 VLM 部分保持**标准的自回归行为**（每个 token 只能看自己及前面的 token），需要构造一个 **Causal Mask**。

**目标**：生成一个 `[B, 1, VLM_S, VLM_S]` 的 mask 矩阵，告诉模型"谁可以看谁"。

---

## 2. 输入

| 变量 | 形状 | 含义 |
|------|------|------|
| `vlm_attention_mask_original` | `[B, VLM_S]` | 原始 padding mask，`1`=有效 token，`0`=padding |
| `vlm_seq_len` | 标量 | VLM 序列长度（如 2400） |
| `combined_mask`（已初始化） | `[B, 1, total_seq_len, total_seq_len]` | 全部填了 `-1.7e38`（屏蔽值） |

---

## 3. 四步构造 Causal Mask

### 第 1 步：找出有效 token（排除 padding）

```python
vlm_valid = vlm_attention_mask_original.bool()   # [B, VLM_S]
```

**示例**（假设 `vlm_seq_len=4`，第 2 位是 padding）：

```
vlm_valid = [True, True, False, True]
```

---

### 第 2 步：构造 Causal 下三角（核心）

```python
vlm_seq_idx = torch.arange(vlm_seq_len).unsqueeze(0).unsqueeze(0)
# → shape: [1, 1, 4]
# 内容: [[[0, 1, 2, 3]]]

vlm_causal = vlm_seq_idx.transpose(1, 2) >= vlm_seq_idx
# transpose(1,2) → [1, 4, 1]
# 广播比较 → [1, 4, 4]
```

**比较过程**（`i >= j`，行是 Query 位置，列是 Key 位置）：

```
         Key0   Key1   Key2   Key3
Query 0  [0>=0   0>=1   0>=2   0>=3]   = [T, F, F, F]
Query 1  [1>=0   1>=1   1>=2   1>=3]   = [T, T, F, F]
Query 2  [2>=0   2>=1   2>=2   2>=3]   = [T, T, T, F]
Query 3  [3>=0   3>=1   3>=2   3>=3]   = [T, T, T, T]
```

这就是**标准 GPT 下三角 mask**：每个位置只能看自己及左边。

---

### 第 3 步：合并三个条件（padding + causal）

```python
vlm_attend = vlm_valid.unsqueeze(2) & vlm_valid.unsqueeze(1) & vlm_causal
# → shape: [B, 4, 4]
```

三个条件分别控制什么：

| 条件 | shape | 控制什么 |
|------|-------|---------|
| `vlm_valid.unsqueeze(2)` | `[B, 4, 1]` | **Query（行）**是否有效 |
| `vlm_valid.unsqueeze(1)` | `[B, 1, 4]` | **Key（列）**是否有效 |
| `vlm_causal` | `[1, 4, 4]` | **因果顺序**（只能看前面） |

**`&`（逻辑与）的效果**（用示例 `vlm_valid=[T,T,F,T]`）：

```
Step A: vlm_valid.unsqueeze(2)     Step B: vlm_valid.unsqueeze(1)
       [T]                                  [T, T, F, T]
       [T]                                  （Key 有效性）
       [F]
       [T]                                  （Query 有效性）

Step C: A & B & causal
         K0    K1   K2(pad) K3
Q0      [T&T   T&T   T&F   T&T]   & [T,F,F,F]  = [T, F, F, F]
Q1      [T&T   T&T   T&F   T&T]   & [T,T,F,F]  = [T, T, F, F]
Q2      [F&T   F&T   F&F   F&T]   & [T,T,T,F]  = [F, F, F, F]  ← 自己是padding
Q3      [T&T   T&T   T&F   T&T]   & [T,T,T,T]  = [T, T, F, T]
```

**三个规则同时生效**：
1. **Causal**：下三角，不能看未来
2. **Query 有效**：自己是 padding 就不能发起查询（第 2 行全 False）
3. **Key 有效**：padding token 不能被任何人看到（第 2 列全 False）

---

### 第 4 步：写入 Mask 矩阵

```python
combined_mask[:, 0, :vlm_seq_len, :vlm_seq_len] = torch.where(
    vlm_attend,           # 如果允许 attend
    torch.tensor(0.0),    #   改成 0.0（不修改 attention score）
    combined_mask[...]    #   否则保持 -1.7e38（屏蔽）
)
```

**效果**：

```
         K0     K1     K2     K3
Q0      [0.0   -inf   -inf   -inf]
Q1      [0.0    0.0   -inf   -inf]
Q2      [-inf  -inf   -inf   -inf]   ← padding Query 全屏蔽
Q3      [0.0    0.0   -inf    0.0]   ← K2 是 padding 也被屏蔽
```

> `-inf` 是 `torch.finfo(dtype).min / 2` 的近似，实际值约 `-1.7e38`。

#### 为什么用这个值？

在 `scaled_dot_product_attention` 中，mask 会被**加到 attention scores** 上：

```
attn_score = Q·K^T / sqrt(d) + mask
```

| mask 值 | 效果 |
|---------|------|
| `0.0` | 不改变 score，该位置正常参与注意力 |
| `-inf` 或极小值 | `exp(极小值) ≈ 0`，softmax 后该位置权重为 0，即**被屏蔽** |

#### 为什么不用 `-inf` 而用 `min / 2`？

1. **数值安全**：`exp(-inf)` 在某些硬件/实现上可能是 `NaN`
2. **避免溢出**：`min / 2` 已经小到足够让 softmax 输出为 0，但不会在中间计算中出界
3. **保险余量**：除以 2 留一点空间，防止某些精度损失导致的不稳定


**最终效果**：mask 矩阵中大部分是被屏蔽的 `-1.7e38`，只有符合规则的位置是 `0.0`。



## 4. 完整流程图

```
输入
├── vlm_attention_mask_original: [B, VLM_S] (1=有效, 0=padding)
│        │
│        ▼
│   vlm_valid = bool() ───────────────────┐
│        │                                │
│        ├──→ unsqueeze(2) ──→ [B, S, 1] │  (Query 有效性)
│        │                                │
│        └──→ unsqueeze(1) ──→ [B, 1, S] │  (Key 有效性)
│                                         │
├── vlm_seq_len ──→ arange() ──→ [1,1,S] │
│        │                                │
│        └──→ transpose(1,2) ──→ [1,S,1] │
│             │                           │
│             └──→ >= arange ──→ [1,S,S] │  (Causal 下三角)
│                      │                  │
│                      ▼                  ▼
│                   vlm_attend = Query_valid & Key_valid & Causal
│                      │                  │
│                      ▼                  ▼
│              [B, S, S] Bool Mask        │
│                      │                  │
└──────────────────────┘                  ▼
                                   combined_mask 的 VLM 区域
                                   True → 0.0      (允许)
                                   False → -1.7e38  (屏蔽)
```

---

## 5. 一句话总结

> **Causal Mask 的构造分三步：先用 `arange + transpose` 生成下三角因果矩阵，再用 `unsqueeze` 分别控制 Query 和 Key 的 padding 有效性，三者 `&` 合并后，把 `True` 位置填 `0.0`、`False` 位置保留 `-1.7e38`，最终形成标准的 GPT 因果注意力掩码。**