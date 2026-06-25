# Emu3Pi0

**Emu3Pi0** 与 **Emu3MoE** 是一个**不同的模型类**。

### 先看两者的架构区别

| | `Emu3MoE` | `Emu3Pi0` |
|---|---|---|
| **Action Expert 位置** | VLM 算完之后，"外挂"在后面 | VLM 和 Action **逐层并行**，共享注意力 |
| **数据流** | `VLM → hidden_states → Action Layers` | `VLM Layer 1` ↔ `Action Layer 1` → `VLM Layer 2` ↔ `Action Layer 2`... |
| **交互深度** | 浅：只在 Action Expert 入口处交互一次 | 深：**每层都交互** |
| **代码体现** | `self.action_layers = [Emu3DecoderLayer...]` | `self.action_expert = Emu3Model(...)` |

### `Emu3MoE` 的交互方式（浅交互）

```
输入 ──→ [VLM Layer 1] ──→ [VLM Layer 2] ──→ ... ──→ [VLM Layer N]
                                                          │
                                                          ▼ hidden_states
                                              ┌──────────────────────┐
                                              │  Action Layer 1      │
                                              │  Action Layer 2      │
                                              │  ...                 │
                                              │  Action Layer M      │
                                              └──────────────────────┘
                                                          │
                                                          ▼
                                                      输出动作
```

- VLM 全部算完，输出 `hidden_states`
- Action Expert 接收 `hidden_states` 作为 `cond`，再算自己的层
- **只交互一次**：在 `forward_action()` 里把 `cond` 和 action 拼接

### `Emu3Pi0` 的交互方式（深交互，Pi0 风格）

```
输入 ──→ [VLM Layer 1] ──┬──→ [VLM Layer 2] ──┬──→ ...
          ↑               │      ↑             │
          └──←────←───────┘      └──←────←─────┘
       [Action Layer 1]        [Action Layer 2]
```

- VLM 和 Action **逐层配对**
- 每层都把两路的 Q/K/V **拼接起来一起做注意力**
- **交互 N 次**（N = 层数）

这就是 `Emu3Pi0SharedLayer` 做的事（第1726行）：

```python
# 把 VLM 和 Action 的 Q/K/V 拼接
q_combined = torch.cat([q_vlm, q_action], dim=2)
k_combined = torch.cat([k_vlm, k_action], dim=2)
v_combined = torch.cat([v_vlm, v_action], dim=2)

# 一起做注意力（共享注意力）
attn_output = F.scaled_dot_product_attention(
    q_combined, k_combined, v_combined, ...
)

# 再拆分回两路
attn_vlm = attn_output[:, :, :vlm_seq_len, :]
attn_action = attn_output[:, :, vlm_seq_len:, :]
```

### 那为什么 `Emu3Pi0` 需要 `Emu3Model` 作为 Action Expert？

因为 Pi0 风格的**逐层共享注意力**要求：

1. **Action Expert 必须和 VLM 层数相同**（逐层配对）
2. **每层都需要完整的 Self-Attention + MLP**（不只是 `Emu3DecoderLayer` 的列表）

`Emu3Model` 内部正好包含：
- `embed_tokens`（嵌入层）
- `layers`（`Emu3DecoderLayer` 的列表）
- `norm`（最终的 LayerNorm）

```python
self.action_expert = Emu3Model(self.action_config)
```

这样 `self.action_expert.layers[i]` 就能和 `self.vlm.model.layers[i]` 逐层配对：

```python
self.shared_layers = [
    Emu3Pi0SharedLayer(vlm_layer, action_layer)
    for vlm_layer, action_layer in zip(self.vlm.model.layers, self.action_expert.layers)
]
```

而在 `Emu3MoE` 中，`self.action_layers` 只是一个 `nn.ModuleList`，没有完整的 `Emu3Model` 包装，因为不需要逐层共享注意力。

---

## 一句话总结

| | `Emu3MoE` | `Emu3Pi0` |
|---|---|---|
| **类比** | VLM 先"看完"场景，再告诉 Action Expert 该怎么做 | VLM 和 Action Expert **边想边交流**，每想一层就交换一次意见 |
| **Action Expert 形态** | `nn.ModuleList([DecoderLayer...])`（外挂） | `Emu3Model(...)`（完整模型，逐层配对） |
| **为什么另外定义** | 不需要，`action_layers` 就够了 | 需要完整的 `layers` 列表来和 VLM 逐层共享注意力 |

> `Emu3Pi0` 加载 `Emu3MoE` 只是借用它**预训练好的 VLM 权重**（`self.vlm`），然后**丢弃 `Emu3MoE` 自带的 Action Expert**（不加载那些权重），重新创建一个独立的 `Emu3Model` 作为新的 Action Expert，用 Pi0 的共享注意力方式重新训练。