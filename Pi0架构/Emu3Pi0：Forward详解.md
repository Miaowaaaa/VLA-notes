# Emu3Pi0 Forward详解

`Emu3Pi0.forward()` 是该模型的**训练入口**。它的核心任务是：接收一批驾驶数据（画面+文字+历史动作+未来动作），让模型学习"看到什么 → 该怎么做"。

下面按代码执行顺序，逐段拆解。

---

## 1. 输入参数：模型"看到"什么？

| 参数 | 形状 | 通俗含义 |
|------|------|---------|
| `action` | `(B, action_frames, action_dim)` | **未来动作的真值**（ground truth）。比如接下来 8 帧每帧的转向角、油门、刹车等 |
| `pre_action` | `(B, pre_action_frames, action_dim)` | **历史动作**。比如过去 3 帧实际执行了什么动作 |
| `cmd` | `(B, 4)` | **驾驶指令**的 one-hot 编码。比如 `[左转, 右转, 直行, 停车]` 哪个为 1 |
| `input_ids` | `(B, vlm_seq_len)` | **VLM 输入**。包含图像 token + 文本 token，描述当前场景和任务 |
| `attention_mask` | `(B, vlm_seq_len)` | VLM 的 padding mask，标记哪些是有效 token |
| `position_ids` | `(B, vlm_seq_len)` | 位置编码，告诉每个 token 在序列中的位置 |
| `labels` | `(B, vlm_seq_len)` | **文本标签**。用于语言建模监督（当前被 `vlm_loss_weight=0` 屏蔽） |

> `B` = batch_size（批次大小），`vlm_seq_len` 通常是 2400（图像+文本的 token 总数）。

---

## 2. 数据准备阶段：把各种输入变成向量

### 2.1 VLM Embedding：把文字/图像变成向量

```python
vlm_initial_hidden_states = self.vlm.model.embed_tokens(input_ids)
```

| 输入 | 输出 |
|------|------|
| `input_ids`: `(B, 2400)` 整数 token ID | `(B, 2400, hidden_size)` 稠密向量 |

**意义**：和 GPT 一样，先把离散的 token ID 映射成连续的向量。图像 patch 和文字 token 在这里一视同仁，都是一串向量。

---

### 2.2 Flow Matching 加噪：给未来动作"掺沙子"

```python
noise = torch.randn_like(action)                    # 纯随机噪声
tau_values = self.rf.sample_t(batch_size)           # 采样时间步 τ ~ Beta(1.5, 1.0)
noisy_action = self.rf.add_noise(action, noise, tau_values)  # 加噪
```

**公式**：

```
noisy_action = (1 - τ) × action + τ × noise
```

| τ 值 | 含义 |
|------|------|
| 0 | 完全真实动作 |
| 1 | 完全噪声 |
| 0.3 | 30% 噪声，70% 真实动作 |

**意义**：红脑的任务是学会"去噪"——从 `noisy_action` 预测出"往哪个方向走能回到真实动作"。这就是 **Flow Matching（流匹配）** 的训练方式。

---

### 2.3 Action Embedding：把带噪动作翻译成红脑语言

```python
tau_emb = self.tau_emb(tau_values)                          # (B, hidden_size)
tau_emb_expanded = tau_emb.unsqueeze(1).expand(-1, action_frames, -1)  # (B, 8, hidden_size)

action_hidden_states_no_state = self.action_projector(noisy_action, tau_emb_expanded)
# → (B, 8, hidden_size)
```

| 操作 | 说明 |
|------|------|
| `tau_emb` | 把时间步 τ 编码成向量，告诉红脑"当前加噪到什么程度了" |
| `tau_emb_expanded` | 把 τ 向量复制 8 份（对应 8 帧动作）|
| `action_projector` | 三层 MLP，把原始动作维度 → hidden_size，并融合时间信息 |

**意义**：原始动作可能只有 7 维（转向、油门等），而 Transformer 需要 hidden_size 维（比如 2048）。`action_projector` 就是做这个升维和特征融合的。

---

### 2.4 State Token：给红脑"历史记忆"

```python
state_input = torch.cat([pre_action.view(batch_size, -1), cmd], dim=1)
# → (B, pre_action_frames × action_dim + 4)

state_token_embedding = self.state_projector(state_input).unsqueeze(1)
# → (B, 1, hidden_size)
```

| 组件 | 来源 | 含义 |
|------|------|------|
| `pre_action` | 数据集 | 过去几帧实际执行的动作 |
| `cmd` | 数据集 | 高层指令（左转/右转/直行/停车）|
| `state_projector` | 两层 MLP + SiLU | 把历史状态压缩成一个 token |

**意义**：红脑不能只看"未来要做什么"，还得知道"现在正在做什么"和"上级命令是什么"。`state_token` 就是把这些上下文信息打包成一个向量，插在动作序列最前面。

---

### 2.5 拼接 Action 序列

```python
action_initial_hidden_states = torch.cat([state_token_embedding, action_hidden_states_no_state], dim=1)
# → (B, 1 + action_frames, hidden_size)
```

最终 Action 输入的结构：

```
[ state_token | action_frame_1 | action_frame_2 | ... | action_frame_8 ]
     ↑历史+指令        ↑第1帧带噪动作      ↑第2帧带噪动作           ↑第8帧带噪动作
```

**意义**：把"我现在在哪"（state）和"我要去哪"（noisy action）拼在一起，作为红脑的初始输入。

---

## 3. Attention Mask：规定"谁能看到谁"

这是 `Emu3Pi0` 最关键的设计之一。它构造了一个**非标准的 attention mask**，让 VLM 和 Action 以特定方式互相看。

```python
combined_attention_mask_4d = self.create_causal_style_attention_mask(
    vlm_seq_len, action_seq_len, attention_mask,
    input_ids, batch_size, device, dtype
)
```

### Mask 的四大规则

假设总序列 = `[VLM 部分 (2400 tokens)] [Action 部分 (9 tokens: 1 state + 8 frames)]`

| 区域 | 规则 | 通俗解释 | 为什么 |
|------|------|---------|-----|
| **VLM → VLM** | Causal | VLM 内部只能看前面的 token（标准 GPT 行为）| - |
| **Action → Action** | 双向 | Action 内部可以互相看（默认）| [原因](./Attention%20Mask之%20Action%20%26%20Action.md) |
| **Action → VLM** | 只能看到第二个 BOA 之前 | Action 不能偷看"未来动作"的文本描述 | - |
| **VLM → Action** | 完全看不到 | VLM 看不到 Action 的任何信息 | - |

> BOA = Begin of Action（动作开始标记）\
> EOA = End of Action（动作结束标记）

### 为什么 Action → VLM 只能看到第二个 BOA 之前？

VLM 序列的组织通常是：

```
[图像] [文字: "前方路况"] [BOA] [历史动作描述] [EOA] [BOA] [未来动作描述] [EOA]
                              ↑第一个BOA/EOA    ↑第二个BOA/EOA（包含正确答案！）
```

如果 Action 能看到第二个 BOA 之后的内容，就等于**直接看到了未来动作的答案**（作弊）。所以 mask 屏蔽了这部分，强制 Action 只能根据"图像+文字+历史动作"来推理未来。

### Mask 可视化（简化版）

```
         VLM(0~2399)   Action(2400~2408)
        ┌───────────┬─────────────────┐
VLM     │  ▼▼▼▼▼▼   │      ✗✗✗       │  ▼=causal, ✗=看不到
        │   下三角   │                 │
        ├───────────┼─────────────────┤
Action  │  ■■■■■■■   │    ◆◆◆◆◆◆     │  ■=能看到(到第二个BOA), ◆=双向
        │  (部分)    │                 │
        └───────────┴─────────────────┘
```

---

## 4. 逐层 Shared Attention：蓝脑和红脑"边想边聊"

```python
for layer_idx in range(num_layers):  # 32 层
    shared_layer = self.shared_layers[layer_idx]
    current_vlm_h, current_action_h = shared_layer(
        current_vlm_h,      # 上一层的 VLM 输出
        current_action_h,   # 上一层的 Action 输出
        vlm_position_ids_original,
        combined_attention_mask_4d,
        vlm_seq_len,
        action_seq_len,
        batch_size,
    )
```

这是 `Emu3Pi0` 和 `Emu3MoE` **最大的区别**：

| | `Emu3MoE` | `Emu3Pi0` |
|---|---|---|
| 交互方式 | VLM 全部算完 → 一次性传给 Action | **每层都交互** |
| 比喻 | 蓝脑写完整报告，红脑看报告做事 | 蓝脑和红脑**边想边讨论** |

### 单层 Shared Layer 内部细节

```python
# 1. 分别 LayerNorm
normed_vlm_h = vlm_layer.input_layernorm(current_vlm_h)
normed_action_h = action_layer.input_layernorm(current_action_h)

# 2. 分别算 Q/K/V
q_vlm, k_vlm, v_vlm = vlm_layer.self_attn.q_proj/k_proj/v_proj(normed_vlm_h)
q_action, k_action, v_action = action_layer.self_attn.q_proj/k_proj/v_proj(normed_action_h)

# 3. VLM 加 RoPE（旋转位置编码），Action 不加
cos, sin = rotary_emb(v_vlm, seq_len=vlm_seq_len)
q_vlm_rope, k_vlm_rope = apply_rotary_pos_emb(q_vlm, k_vlm, cos, sin, ...)

# 4. 拼接 Q/K/V
combined_q = torch.cat([q_vlm_rope, q_action], dim=2)   # (B, heads, VLM+A, head_dim)
combined_k = torch.cat([k_vlm_rope, k_action], dim=2)
combined_v = torch.cat([v_vlm, v_action], dim=2)

# 5. 一起做注意力（核心！）
attn_output_combined = F.scaled_dot_product_attention(
    combined_q, combined_k, combined_v,
    attn_mask=combined_attention_mask_4d  # ← 前面构造的特殊 mask
)

# 6. 拆回两路
attn_out_vlm = attn_output_combined[:, :vlm_seq_len, :]
attn_out_action = attn_output_combined[:, vlm_seq_len:, :]

# 7. 各自过 o_proj + MLP（标准 Transformer 残差）
current_vlm_h = residual_vlm + o_proj(attn_out_vlm) + mlp(...)
current_action_h = residual_action + o_proj(attn_out_action) + mlp(...)
```

### 为什么 Action 不加 RoPE？

- **VLM 需要 RoPE**：因为 VLM 处理的是文本/图像序列，token 之间有严格的顺序关系（"今天"必须在"明天"前面）
- **Action 不需要 RoPE**：Action 序列的每一帧 already 有明确的时间顺序（state → frame1 → frame2...），而且时间信息已经通过 `tau_emb` 注入。再加 RoPE 反而可能冗余。

---

## 5. 最终输出：从隐状态到预测

### 5.1 过 Final Norm

```python
final_vlm_hidden = self.vlm.model.norm(current_vlm_h)           # VLM 输出归一化
final_action_hidden = self.action_expert.norm(current_action_h)  # Action 输出归一化
```

### 5.2 Action 解码

```python
velo_t_pred = self.action_decoder(final_action_hidden[:, 1:, :], tau_emb_expanded)
```

| 操作 | 说明 |
|------|------|
| `[:, 1:, :]` | **去掉 state_token**，只取 8 帧动作部分 |
| `action_decoder` | `FinalLayer`（adaLN 调制），输入 hidden_size → 输出 action_dim |
| `tau_emb_expanded` | 时间嵌入再次作为条件，调制输出 |

**输出**：`velo_t_pred` 形状 `(B, 8, action_dim)` —— 红脑预测的"速度场"（去噪方向）。

---

## 6. 损失计算：红脑学得好不好？

```python
action_loss = F.mse_loss(noise - action, velo_t_pred)
```

| 项目 | 含义 |
|------|------|
| `noise - action` | 真实速度场（从真实动作到噪声的方向）|
| `velo_t_pred` | 红脑预测的速度场 |
| `MSE` | 两者的均方误差 |

**Flow Matching 原理**：红脑不直接预测动作，而是预测"怎么从带噪动作走回真实动作"。训练目标就是让预测的"方向"和真实方向一致。

### 总损失（当前代码）

```python
self.action_loss_weight = 1.0
self.vlm_loss_weight = 0.0          # ← 硬编码为 0，VLM loss 被关闭

total_loss = action_loss            # 只有动作损失
```

> 如之前分析的，`vlm_loss_weight = 0.0` 导致语言监督被完全关闭，当前只训练 Action Expert。

---

## 7. 完整数据流图

```
输入
├── input_ids ──→ embed_tokens ──→ vlm_h (B, 2400, H)
│                                    │
├── action ──→ 加噪 ──→ noisy_action  │
│      ↑                             │
│      └── τ ~ Beta ──┐              │
│                     ▼              │
│              tau_emb ──→ expand ──┐│
│                                   ▼▼
│                         action_projector ──→ action_h_no_state (B, 8, H)
│
├── pre_action ──┐
│                ├──→ cat ──→ state_projector ──→ state_token (B, 1, H)
└── cmd ─────────┘
                                                │
                    [state_token, action_h_no_state] ──→ cat ──→ action_h (B, 9, H)
                                                │
                                                ▼
        ┌─────────────────────────────────────────────────────┐
        │              逐层 Shared Attention (×32)             │
        │                                                      │
        │   vlm_h ──→ [LayerNorm → Q/K/V → RoPE] ──┐          │
        │                                            ├──→ concat │
        │   action_h ──→ [LayerNorm → Q/K/V] ──────┘          │
        │                                            │          │
        │                                SDP Attention          │
        │                                    │                  │
        │                           拆回 vlm / action            │
        │                                    │                  │
        │                           各自 o_proj + MLP            │
        └─────────────────────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
    vlm.model.norm()          action_expert.norm()
              │                       │
              ▼                       ▼
        (vlm loss 被关闭)      action_decoder[:, 1:, :]
                                      │
                                      ▼
                                velo_t_pred
                                      │
                              MSE(noise-action, velo_t_pred)
                                      │
                                      ▼
                                  action_loss
```

---

## 8. 一句话总结

> `Emu3Pi0.forward()` 是**逐层共享注意力**的训练流程：VLM 和 Action Expert 像两个并肩工作的专家，每层都交换意见（通过拼接 Q/K/V 做联合注意力），最终 Action Expert 输出速度场预测，用 Flow Matching 的 MSE 损失进行训练。当前代码只优化动作损失，语言监督被临时关闭。