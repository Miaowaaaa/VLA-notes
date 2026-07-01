
# 从噪声中还原未来Action

## 初始化（2255~2259）

```python
z = torch.randn(B, 8, 7)                           # 起点：纯噪声
state_token = self.state_projector([pre_action, cmd]).unsqueeze(1)  # 历史+指令
```

---

## 去噪循环（2268~2316）

```python
while current_time >= -dt / 2:   # t 从 1 退到 0，共 _num_inference_steps 步
```

**每步只做四件事：**

| 代码 | 在做什么 |
|------|---------|
| `tau_emb = self.tau_emb(t)` | 编码当前时间步，告诉模型"进度多少了" |
| `action_h = self.action_projector(z, tau_emb)` | 噪声+时间 → Action 隐空间 |
| `current_action_h = cat([state_token, action_h], dim=1)` | 状态token + 8帧动作，拼成完整输入 |
| `for 32 layers: shared_layer(vlm_h, action_h, mask)` | VLM 和 Action 做共享注意力，Action 看 VLM 的语义理解 |
| `velo = self.action_decoder(norm(action_h)[:, 1:, :], tau_emb)` | 从隐向量解码出"速度场"：噪声该往哪变 |
| `z = z + dt * velo` | **Euler 更新**：沿速度场方向走一步（`dt` 为负，实际是做减法去噪） |
| `t += dt` | 时间前进一步 |

---

## 返回

```python
return z   # t≈0，噪声已收敛为未来 8 帧的驾驶动作 [B, 8, 7]
```

---

## 训练测试过程对比

| | 训练 | 测试 |
|------|------|------|
| 输入 | [VLM_state] + [pre_action + cmd, noisy_gt_action] ----> [velo_pre] | [VLM_state] + [pre_action + cmd, pure_noisy] ----> [velo_pre] |
| 输出 | [velo_pre] | [velo_pre] |
| 监督 | [noisy_gt_action] - [gt_action]<-> [velo_pre] | [pure_noisy] - [velo_pre] |



## 一句话总结

> 把随机噪声 `z` 和当前时间 `t` 喂给模型，模型输出"该怎么变"（`velo`），然后 `z` 朝这个方向挪一点；重复 20 次，`z` 就从噪声变成了可执行的动作轨迹。