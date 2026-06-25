
# `Emu3MoE` — Action Expert 模块文档

---

## 1. 一句话概括

`Emu3MoE` 是一个**有两套大脑**的多模态大模型：

- **蓝脑（VLM 专家）**：看懂摄像头画面、读懂文字指令，输出语义理解
- **红脑（Action Expert）**：根据蓝脑的理解，预测方向盘角度、油门大小等驾驶动作

两个大脑各自有独立的参数，各自干各自擅长的事，这就是 **MoE（专家混合）风格**。

---

## 2. 先澄清一个容易误解的关系

### `Emu3MoE` 和 `Emu3ForCausalLM` 是**兄弟**，不是父子

```python
Emu3PreTrainedModel
        │
        ├── Emu3ForCausalLM      ← 原版 VLM，只输出文字
        │
        └── Emu3MoE              ← 新版 VLM + Action Expert
```

两者**都直接继承 `Emu3PreTrainedModel`**。`Emu3MoE` 并没有调用或继承 `Emu3ForCausalLM`。

那它俩什么关系？—— `Emu3MoE` **复制了 `Emu3ForCausalLM` 的全部蓝脑代码**，然后**在旁边并联了一个红脑**。代码层面体现为：

| 组件 | `Emu3ForCausalLM` | `Emu3MoE` |
|------|-------------------|-----------|
| `self.model` | `Emu3Model(config)` | 完全一样的副本 |
| `self.lm_head` | `Linear(hidden_size, vocab_size)` | 完全一样的副本 |
| `forward()` 的 VLM 部分 | `self.model()` → `lm_head` | **逐行复制**，中间插入了 action 逻辑 |
| `prepare_inputs_for_generation()` | 完整实现 | **逐行复制** |
| `_reorder_cache()` | 完整实现 | **逐行复制** |

> 为什么这么设计？因为 `forward()` 里的 `hidden_states` 是局部变量，子类通过继承拿不到。所以最快的方式就是复制一份再加功能。

---

## 3. 什么是 "MoE 风格"？

### 3.1 传统模型 vs MoE 风格

| | 传统大模型 | MoE 风格（本模型）|
|---|---|---|
| **比喻** | 一个全能博士，什么都做 | 一个专家组，不同专家干不同活 |
| **参数** | 所有参数都参与每次计算 | 只有相关专家的参数参与 |
| **优点** | 简单 | 各司其职，互不干扰，更专业 |

### 3.2 本模型的 MoE 长什么样

```
                    输入：摄像头画面 + 文字指令
                              │
                              ▼
                   ┌─────────────────────┐
                   │   共享输入编码层      │    ← 眼睛/耳朵（共用）
                   │  （图像 + 文本嵌入）   │
                   └─────────────────────┘
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
        ┌───────────────┐         ┌──────────────────┐
        │   蓝脑专家     │         │    红脑专家       │
        │   (VLM 主干)   │         │  (Action Expert)  │
        │               │         │                  │
        │  理解场景语义   │◄───────►│  生成驾驶动作     │
        │  输出文字回答   │  cond   │  方向盘/油门/刹车  │
        │               │         │                  │
        │  参数：θ_vlm  │         │   参数：θ_action  │
        └───────────────┘         └──────────────────┘
                 │                         │
                 ▼                         ▼
            输出文字 logits              输出动作序列
```

**关键设计**：蓝脑和红脑是**两套完全独立的参数**。蓝脑搞语义理解，红脑搞动作预测，互不干扰。但红脑在做动作时会"瞄一眼"蓝脑的理解（通过 `cond`），相当于"前面有红灯"这个语义信息会指导红脑输出"踩刹车"这个动作。

---

## 4. 蓝脑（VLM 专家）详解

### 4.1 蓝脑有哪些模块？

| 代码变量/方法 | 类型 | 通俗解释 |
|--------------|------|---------|
| `self.model` | `Emu3Model` | 蓝脑的主干 Transformer，相当于蓝脑的大脑皮层 |
| `self.lm_head` | `nn.Linear` | 蓝脑的"嘴巴"，把思考结果变成文字概率 |
| `forward()` 中 `self.model(...)` | 方法调用 | 蓝脑开始思考 |
| `forward()` 中 `self.lm_head(hidden_states)` | 方法调用 | 蓝脑输出文字 |
| `prepare_inputs_for_generation()` | 方法 | 自回归生成时的"记忆整理"（配合 KV Cache） |
| `_reorder_cache()` | 静态方法 | Beam Search 时重新排列记忆 |

```python
# 蓝脑初始化（modeling_emu3.py:1446-1447）
self.model = Emu3Model(config)                          # ← 蓝脑主干
self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # ← 蓝脑输出头
```

### 4.2 蓝脑在 `forward()` 里做什么？

```python
# 1. 蓝脑处理输入（和 Emu3ForCausalLM 一模一样）
outputs = self.model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    ...
)
hidden_states = outputs[0]   # ← 蓝脑的"理解结果"，形状 (B, seq_len, hidden_size)

# 2. 蓝脑输出文字 logits（和 Emu3ForCausalLM 一模一样）
logits = self.lm_head(hidden_states)   # ← (B, seq_len, vocab_size)
```

**蓝脑的损失**：如果传了 `labels`，就算交叉熵损失（和 GPT 训练一样）：

```python
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
loss = CrossEntropyLoss()(shift_logits, shift_labels)
```

---

## 5. 红脑（Action Expert）详解

### 5.1 红脑有哪些模块？

| 代码变量/方法 | 类型 | 通俗解释 |
|--------------|------|---------|
| `self.action_projector` | `ActionProjector` | 红脑的"翻译官"，把原始动作+时间信息翻译成红脑能懂的语言 |
| `self.action_layers` | `nn.ModuleList[Emu3DecoderLayer]` | 红脑的独立 Transformer 层堆叠（和蓝脑结构一样，但参数独立） |
| `self.action_decoder` | `FinalLayer` | 红脑的"手"，输出最终动作预测（带 adaLN 调制） |
| `self.rf` | `FlowMatchingScheduler` | 红脑的"加噪调度员"，训练时决定加多少噪声 |
| `self.tau_emb` | `SinusoidalPosEmb` | 红脑的"时钟编码器"，把时间步变成向量 |
| `forward_action(z, t, cond)` | 方法 | 红脑核心前向：接收带噪动作+时间+蓝脑条件，输出速度预测 |
| `generate_action(outputs, ...)` | 方法 | 红脑推理入口：从纯噪声逐步"去噪"生成动作 |

```python
# 红脑初始化（modeling_emu3.py:1458-1468）
if self.action_experts:
    action_config = Emu3Config.from_dict(config.action_config)
    self.action_projector = ActionProjector(config.action_dim, action_config.hidden_size)
    self.action_layers = nn.ModuleList([
        Emu3DecoderLayer(action_config, layer_idx) 
        for layer_idx in range(action_config.num_hidden_layers)
    ])
    self.action_decoder = FinalLayer(action_config.hidden_size, config.action_dim)
    self.rf = FlowMatchingScheduler(sample_method="beta", s=1.0)
    self.tau_emb = SinusoidalPosEmb(action_config.hidden_size)
```

### 5.2 红脑各组件详解

#### 5.2.1 `ActionProjector` — 动作翻译官

**作用**：把原始动作 `z` 和时间嵌入 `tau` 融合，投影到红脑的 hidden size。

**结构**：
```
z (动作) ──→ W1 ──→ (+ pos_embed) ──→ concat[tau] ──→ W2 ──→ SiLU ──→ W3 ──→ out
```

| 层 | 输入 | 输出 |
|---|---|---|
| `W1` | `(B, frames, action_dim)` | `(B, frames, hidden_size)` |
| `W2` | `(B, frames, hidden_size×2)` | `(B, frames, hidden_size)` |
| `W3` | `(B, frames, hidden_size)` | `(B, frames, hidden_size)` |

> `pos_embed` 是可选的帧级位置编码，让红脑知道这是第几帧动作。

#### 5.2.2 `SinusoidalPosEmb` — 时间编码器

**作用**：把标量时间步 `t`（比如 0.3）编码成向量，让红脑知道"现在加噪到啥程度了"。

**原理**：和 Transformer 的位置编码一样，用正弦/余弦函数：

\[
\text{emb}_i = \begin{cases} \sin(t / \theta^{i / (d/2 - 1)}) \\ \cos(t / \theta^{i / (d/2 - 1)}) \end{cases}
\]

| 输入 | 输出 |
|------|------|
| `(B,)` | `(B, hidden_size)` |

#### 5.2.3 `FlowMatchingScheduler` — 加噪调度员

**作用**：训练时负责两件事——**采样时间步**和**加噪声**。

**两种采样策略**：

| 策略 | 原理 | 效果 |
|------|------|------|
| `uniform` | `[0, s]` 均匀采样 | 各阶段训练量一样 |
| `beta`（默认） | `Beta(1.5, 1.0)` 分布 | **偏向小时间步**（更多高噪声样本） |

**加噪公式**：

```
z_τ = (1 - τ) × 真实动作 + τ × 随机噪声
```

- τ = 0：完全真实动作
- τ = 1：完全噪声
- τ = 0.3：30% 噪声，70% 真实动作

#### 5.2.4 `action_layers` — 红脑的 Transformer

**作用**：红脑的独立思考层。结构和蓝脑的 `Emu3DecoderLayer` **一模一样**（自注意力 + MLP），但**参数完全独立**。

**关键设计——条件拼接**：

```python
# 蓝脑理解 + 红脑动作，沿序列维度拼接
action_hidden_states = torch.cat([cond, action_hidden_states], dim=1)
#      ↑蓝脑输出              ↑红脑投影后的动作

# 一起过红脑的 Transformer
for layer in self.action_layers:
    action_hidden_states = layer(action_hidden_states)[0]

# 再拆开
hidden_states_refine = action_hidden_states[:, :seq_len, :]      # ← 蓝脑部分（被动作信息精炼过）
action_hidden_states = action_hidden_states[:, seq_len:, :]      # ← 红脑部分（继续往下走）
```

> **通俗解释**：红脑做动作时，把蓝脑的理解"贴"在自己前面，一起过注意力层。这样红脑每预测一个动作，都能"瞄一眼"场景理解。同时蓝脑的理解也会被动作信息反向精炼。

#### 5.2.5 `FinalLayer` (adaLN) — 红脑的输出手

**作用**：把红脑的思考结果变成最终的速度预测。

**结构**：
```
x ──→ LayerNorm ──→ ×(1+scale) + shift ──→ Linear ──→ velo_pred
              ↑
         tau ─┘ (通过 adaLN_modulation 生成 shift, scale)
```

- `adaLN_modulation`：`SiLU → Linear`，从时间嵌入生成 **shift** 和 **scale** 两个调制参数
- `modulate(x, shift, scale) = x × (1 + scale) + shift`

**初始化技巧**：`Linear` 的权重和偏置初始化为 **0**。这样训练初期红脑输出全 0，不会影响蓝脑的训练稳定性。

| 输入 | 输出 |
|------|------|
| `x`: `(B, frames, hidden_size)` | `velo_pred`: `(B, frames, action_dim)` |
| `c` (tau_emb): `(B, frames, hidden_size)` | |

---

## 6. 两脑交互：蓝脑 → 红脑

### 6.1 交互时机

```
┌─────────────────────────────────────────────────────────────┐
│                        forward() 训练流程                     │
├─────────────────────────────────────────────────────────────┤
│  1. 蓝脑处理输入                                              │
│     input_ids ──→ self.model() ──→ hidden_states (cond)      │
│                                         │                    │
│                                         ▼                    │
│  2. 红脑处理动作（仅在 training 且 action 不为 None 时执行）    │
│     action ──→ 加噪 ──→ z_τ                                   │
│     z_τ + τ + cond ──→ forward_action() ──→ velo_pred        │
│                                         │                    │
│                                         ▼                    │
│  3. 损失合并                                                  │
│     loss = loss_lm + loss_action × vision_loss_weight        │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 `forward_action()` 详细流程

```python
def forward_action(self, z, t, cond):
    # 1. 时间编码
    tau_emb = self.tau_emb(t).to(z.dtype)           # (B, hidden_size)
    tau_emb = tau_emb.repeat(1, z.shape[1], 1)      # (B, frames, hidden_size)

    # 2. 动作投影（融合时间信息）
    action_hidden = self.action_projector(z, tau_emb)  # (B, frames, hidden_size)

    # 3. 拼接蓝脑理解和红脑动作
    combined = torch.cat([cond, action_hidden], dim=1)  # (B, seq_len+frames, hidden_size)

    # 4. 红脑 Transformer 独立思考
    for layer in self.action_layers:
        combined = layer(combined)[0]

    # 5. 拆分
    hidden_states_refine = combined[:, :seq_len, :]     # 蓝脑理解被动作信息精炼
    action_hidden = combined[:, seq_len:, :]            # 红脑继续

    # 6. 最终输出（adaLN 调制）
    velo_pred = self.action_decoder(action_hidden, tau_emb)  # (B, frames, action_dim)

    return velo_pred, hidden_states_refine
```

---

## 7. 红脑的训练：Flow Matching（流匹配）

### 7.1 核心思想

不直接预测动作，而是预测**"从噪声走向真实动作的方向"**（速度场）。

```
噪声（随机乱猜） ←────────────────────── 真实动作（人类驾驶）
     ↑                                     │
     └──────── 红脑要学会"逆流而上" ────────┘
```

### 7.2 训练步骤

| 步骤 | 做什么 | 代码对应 |
|------|--------|---------|
| 1 | 拿到真实动作 `action` | 数据标签 |
| 2 | 随机采样时间步 `τ` | `self.rf.sample_t(B)` |
| 3 | 生成随机噪声 | `torch.randn_like(action)` |
| 4 | 加噪：`z_τ = (1-τ)·action + τ·noise` | `self.rf.add_noise()` |
| 5 | 红脑预测方向 `v_θ` | `self.forward_action(z_τ, τ, hidden_states)` |
| 6 | 目标方向 = `noise - action`（真实速度场）| 理论推导 |
| 7 | 损失 = MSE(目标方向, 红脑预测) | `F.mse_loss(noise - action, velo_pred)` |

### 7.3 损失合并

```python
loss = loss_lm                          # 蓝脑：文字预测损失（CrossEntropy）
if action is not None and self.action_experts:
    loss += loss_action * self.vision_loss_weight   # 红脑：流匹配损失（MSE）
```

> `vision_loss_weight`（默认 0.1）控制红脑损失占总损失的比例，避免红脑"喧宾夺主"。

---

## 8. 红脑的推理：逐步去噪

训练好的红脑不会直接输出动作，而是通过 **欧拉离散化** 从噪声中"提炼"出动作：

```python
def generate_action(self, outputs, sample_steps=20, frames=8, action_dim=7):
    # 1. 先用蓝脑理解输入
    outputs = self.model(input_ids=input_ids, ...)
    hidden_states = outputs[0]

    # 2. 初始化纯随机噪声
    z = torch.randn((batch_size, frames, action_dim)).to(hidden_states.device)
    dt = 1.0 / sample_steps   # 每步走 5%

    # 3. 逐步去噪（反向欧拉）
    for i in range(sample_steps, 0, -1):
        t = i / sample_steps          # 当前时间点，从 1.0 走到 0.0
        t = torch.tensor([t] * batch_size).to(device)

        velo_pred, _ = self.forward_action(z, t, cond=hidden_states)
        z = z - dt * velo_pred        # ← 按红脑指的方向走一步

    return z   # 最终动作序列
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sample_steps` | `20` | 去噪步数，越大越平滑但越慢 |
| `frames` | `8` | 预测未来多少帧的动作 |
| `action_dim` | `7` | 每帧动作有多少个维度（转向、油门等）|

**通俗比喻**：就像雕塑家从一块大理石（噪声）开始，红脑每步说"这里凿掉一点"、"那里磨平一点"，20 步后雕出了最终的动作形状。

---

## 9. KV Cache：让蓝脑说话更快的"记忆便签本"

### 9.1 问题：Transformer 生成文字为什么越来越慢？

Transformer 生成是一个字一个字往外蹦：

```
输入："今天天气"
生成"很"时，要计算 4 个 token 的注意力
生成"好"时，要计算 5 个 token 的注意力
生成"，"时，要计算 6 个 token 的注意力
...
生成第 1000 个字时，要算 1003 个 token！
```

**不优化的话，生成长度每增加 1，计算量就大一截**，越来越慢。

### 9.2 解决方案：缓存 K 和 V

Transformer 注意力需要三个东西：

| | 是什么 | 每生成新字是否变化 | 能缓存吗？ |
|---|---|---|---|
| **Q (Query)** | "我现在想问什么" | 每个新字都要重新问 | ❌ 不能 |
| **K (Key)** | "每个位置的关键词" | 老字不变 | ✅ **可以缓存** |
| **V (Value)** | "每个位置的内容" | 老字不变 | ✅ **可以缓存** |

**KV Cache 做的事**：

```
生成"很"时：算出 K_今, K_天, K_气, K_很, V_今...V_很 → 存进"便签本"
生成"好"时：只需算 K_好, V_好 → 从便签本里直接拿之前的 K, V
生成"，"时：只需算 K_，, V_， → 便签本越来越厚，但每步只算新字
```

**效果**：
- 不缓存：生成第 N 个字要算 O(N²)
- 有缓存：生成第 N 个字只要 O(N)，**快到飞起**

### 9.3 本模型中的 KV Cache

在 `Emu3MoE` 中，`past_key_values` 参数就是 KV Cache：

```python
outputs = self.model(
    input_ids=input_ids,
    past_key_values=past_key_values,   # ← 上次存的 K, V 便签本
    ...
)
```

- **训练时**：`past_key_values=None`，不需要缓存（一次性算完所有字）
- **推理时**：传入上次缓存，只算新 token

`prepare_inputs_for_generation()` 就是专门管理这个便签本的——每次生成新字后，更新便签本，裁剪掉不需要的旧记忆。

---

## 10. 完整数据流总览

### 训练阶段

```
输入画面+文字 ──→ [蓝脑] self.model() ──→ hidden_states (cond)
                                              │
                                              │
真实动作 ──→ [红脑] self.rf.sample_t() ──→ τ
            self.rf.add_noise() ──→ z_τ       │
                                              ▼
                                    [红脑] forward_action(z_τ, τ, cond)
                                              │
                                    ┌─────────┴─────────┐
                                    ▼                   ▼
                              velo_pred          hidden_states_refine
                                    │
                    MSE(噪声-真实动作, velo_pred) = loss_action
                                    │
                                    ▼
总损失 = CrossEntropy(logits, labels) + loss_action × weight
                ↑蓝脑                        ↑红脑
```

### 推理阶段

```
输入画面+文字 ──→ [蓝脑] self.model() ──→ hidden_states
                                              │
                                              ▼
                                    随机噪声 z ~ N(0, I)
                                              │
                                    ┌─────────┴─────────┐
                                    │   循环 20 步去噪    │
                                    │  z = z - dt × 红脑 │
                                    │                   │
                                    │  红脑看(z, t, cond) │
                                    │  说"往这边走"        │
                                    └───────────────────┘
                                              │
                                              ▼
                                    最终动作序列 z
```

---

## 11. 组件速查表

| 组件 | 属于 | 比喻 | 代码位置 |
|------|------|------|---------|
| `self.model` | 蓝脑 | 大脑皮层 | `__init__:1446` |
| `self.lm_head` | 蓝脑 | 嘴巴（输出文字） | `__init__:1471` |
| `self.action_projector` | 红脑 | 翻译官 | `__init__:1461` |
| `self.action_layers` | 红脑 | 独立思考层 | `__init__:1462` |
| `self.action_decoder` | 红脑 | 手（输出动作） | `__init__:1465` |
| `self.rf` | 红脑 | 加噪调度员 | `__init__:1467` |
| `self.tau_emb` | 红脑 | 时钟编码器 | `__init__:1468` |
| `forward()` | 蓝脑+红脑 | 总指挥 | `line:1496` |
| `forward_action()` | 红脑 | 红脑核心 | `line:1667` |
| `generate_action()` | 红脑 | 红脑推理入口 | `line:1692` |
| `past_key_values` | 蓝脑 | 记忆便签本（KV Cache）| `forward 参数` |

有的。在 `Emu3MoE` 的训练中，**语言（`labels`）对蓝脑起到明确的监督作用**。

---

## Language参与MoE训练监督

看 `Emu3MoE.forward()` 的损失计算部分（`modeling_emu3.py:1567-1588`）：

```python
loss = None
if labels is not None:
    # 1. 语言建模损失（蓝脑监督）
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, self.vocab_size)
    shift_labels = shift_labels.view(-1)
    loss = loss_fct(shift_logits, shift_labels)   # ← 蓝脑的语言监督损失

    # 2. 动作损失（红脑监督）
    if action is not None and self.action_experts:
        loss += loss_action * self.vision_loss_weight   # ← 红脑的动作监督损失
```

**训练时总损失**：

$$
[
\text{Loss}_{\text{total}} = \underbrace{\text{CrossEntropy}(\text{logits}, \text{labels})}_{\text{蓝脑：语言监督}} + \underbrace{\text{MSE}(v_{\text{true}}, v_{\text{pred}}) \times w}_{\text{红脑：动作监督}}
]
$$
---

## Language 监督具体起什么作用？

### 1. 让蓝脑学会"看懂 + 说人话"

`labels` 通常是多模态对话数据中的**文本回复**。例如：

| 输入 (`input_ids`) | 目标输出 (`labels`) |
|---|---|
| `<image> 前方路况如何？` | `前方有行人正在过马路，建议减速。` |
| `<image> 描述当前场景` | `十字路口，红灯亮起，前方车辆已停止。` |

蓝脑通过 `CrossEntropyLoss` 学习：看到这幅图 + 这个问题，应该输出这段文字。

### 2. 蓝脑的理解质量直接影响红脑

红脑接收的是蓝脑输出的 `hidden_states` 作为 `cond`。如果蓝脑的语言监督不到位：

- 蓝脑看不懂图 → `hidden_states` 语义错误 → 红脑接收到错误的场景理解 → 动作预测错误
- 蓝脑理解模糊 → `hidden_states` 信息不足 → 红脑缺乏足够的条件指导

**所以 language 监督是"根基"**——它确保蓝脑能正确理解场景，红脑才能基于此做出正确动作。

### 3. 联合训练 vs 单独训练

在 `Emu3MoE` 中，蓝脑和红脑是**联合训练**的：

```
一个 batch 同时包含：
├── 图像 + 文本问题 ──→ 蓝脑 ──→ 预测文本回答（与 labels 算 CE Loss）
│                              │
│                              ▼ hidden_states
│                         [红脑] ──→ 预测动作（与 action 算 MSE Loss）
│
└── 两个损失反向传播，同时更新蓝脑和红脑的参数
```

**关键设计**：红脑的梯度会**反向流回蓝脑**。因为红脑用了蓝脑的 `hidden_states` 作为 `cond`，所以红脑的动作预测误差会通过 `cond` 这条路径，一直反向传播到蓝脑的各层参数。

这意味着：
- 红脑预测错了 → 不只是红脑自己改，蓝脑也要调整自己的理解方式
- **Language 监督和 Action 监督相互影响**：蓝脑为了帮红脑更好地预测动作，会主动学习对驾驶决策更有用的视觉-语言表征

---

## 补充：`labels` 的构造方式

在实际训练数据中，`labels` 通常这样构造：

```python
# 多模态对话格式
conversation = [
    {"role": "user", "content": "<image>\n前方路况如何？"},
    {"role": "assistant", "content": "前方有行人过马路，建议减速停车。"}
]

# 构造 input_ids 和 labels
input_ids = tokenizer(prompt).input_ids
labels = input_ids.copy()

# 用户说的话不计算损失（mask 掉）
labels[:user_prompt_length] = -100

# 只有 assistant 的回复部分计算 CE Loss
# labels[assistant_start:] 保持原 token id
```

**-100 的 token 在 CrossEntropyLoss 中被忽略**，所以蓝脑只学习"如何回复"，不学习"如何理解用户问题"。

---

## 一句话总结

> **Language 通过 `labels` 对蓝脑施加直接的监督（CrossEntropy Loss），是蓝脑学习视觉-语言理解的核心信号。同时，由于红脑依赖蓝脑的 `hidden_states` 作为条件，Language 监督间接也影响了红脑的动作预测质量——两者是联合训练、相互促进的。**

