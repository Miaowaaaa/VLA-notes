# FlowMatchingScheduler 计算逻辑与工作原理

## 模块概述

`FlowMatchingScheduler` 是 Flow Matching 模型的核心调度器，负责在训练和推理过程中管理**噪声添加**和**时间步采样**。它实现了从干净数据到噪声数据的连续变换过程。

## 核心概念

### Flow Matching 基本原理

Flow Matching 是一种**连续时间生成模型**，与 Diffusion Model 类似但更简洁：

```
干净数据 (x₀) ←──────────→ 噪声数据 (x₁)
         去噪过程          加噪过程
      (推理/生成)        (训练采样)
```

**关键思想**：
- 学习一个**速度场（velocity field）**，指导噪声数据流向干净数据
- 通过**常微分方程（ODE）**描述数据从噪声到干净的连续变换
- 时间步 \(t \in [0, 1]\) 控制变换进度

## 类结构与参数

### 初始化参数

```python
def __init__(self, sample_method="beta", s=0.999):
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sample_method` | str | `"beta"` | 时间步采样方法：`"uniform"` 或 `"beta"` |
| `s` | float | `0.999` | 时间步阈值，控制采样范围 `[0, s]` |

### 两种采样分布

#### 1. Beta 分布（推荐）

```python
if self.sample_method == "beta":
    # Beta(1.5, 1.0) 分布
    self.distribution = Beta(torch.tensor([1.5]), torch.tensor([1.0]))
```

**Beta(1.5, 1.0) 的特性**：

```
概率密度
  ↑
  |     /\
  |    /  \
  |   /    \
  |  /      \
  | /        \
  |/__________\___→ t
  0          1
  
特点：偏向采样较大的 t 值（更多噪声）
```

**为什么使用 Beta 分布？**

- **偏向高噪声区域**：α=1.5, β=1.0 使分布向右倾斜
- **训练效率**：模型更多地在噪声较大的状态下学习，提高鲁棒性
- **经验验证**：在 Flow Matching 中表现优于均匀采样

#### 2. 均匀分布

```python
elif self.sample_method == "uniform":
    # [0, 1] 均匀分布
    self.distribution = Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
```

**均匀分布的特性**：

```
概率密度
  ↑
  |_____________
  |             |
  |             |
  |             |
  |_____________|___→ t
  0          1
  
特点：所有时间步等概率采样
```

## 核心方法详解

### 方法 1: sample_t - 采样时间步

```python
def sample_t(self, num_samples):
```

#### 功能

采样 `num_samples` 个时间步 \(t\)，范围在 \([0, s]\)

#### 计算流程

##### Beta 采样

```python
if self.sample_method == "beta":
    # Step 1: 从 Beta(1.5, 1.0) 采样
    samples = self.distribution.sample((num_samples,))
    # samples ∈ [0, 1]
    
    # Step 2: 反转并缩放到 [0, s]
    timesteps = self.s * (1 - samples)
    # timesteps ∈ [0, s]
```

**详细过程**：

```python
# 示例：采样 5 个时间步
num_samples = 5

# 1. Beta 采样（假设结果）
samples = [0.2, 0.5, 0.7, 0.3, 0.8]  # ∈ [0, 1]

# 2. 反转（1 - samples）
reversed = [0.8, 0.5, 0.3, 0.7, 0.2]

# 3. 缩放（* s，s=0.999）
timesteps = [0.7992, 0.4995, 0.2997, 0.6993, 0.1998]
```

**为什么要 `1 - samples`？**

- Beta(1.5, 1.0) 偏向**大值**（靠近 1）
- `1 - samples` 反转后偏向**小值**（靠近 0）
- 但乘以 `s` 后，实际采样范围是 \([0, 0.999]\)
- **效果**：更多采样靠近 0（干净数据端），符合 Flow Matching 的训练需求

##### 均匀采样

```python
else:
    # 直接从 [0, s] 均匀采样
    timesteps = self.distribution.sample((num_samples,)) * self.s
```

**示例**：

```python
# Uniform(0, 1) 采样
samples = [0.1, 0.4, 0.6, 0.8, 0.3]

# 缩放
timesteps = [0.0999, 0.3996, 0.5994, 0.7992, 0.2997]
```

#### 输出形状

```python
return timesteps.squeeze(1)
# 输入: (num_samples, 1)
# 输出: (num_samples,)
```

### 方法 2: add_noise - 添加噪声

```python
def add_noise(self, original_samples, noise, timesteps):
```

#### 功能

根据时间步 \(t\) 将噪声添加到干净样本中，生成带噪声的样本。

#### 核心公式

这是 Flow Matching 的**线性插值公式**：

\[
x_t = (1 - t) \cdot x_0 + t \cdot \epsilon
\]

其中：
- \(x_0\)：干净数据（original_samples）
- \(\epsilon\)：噪声（noise），通常从标准正态分布采样
- \(t\)：时间步（timesteps），范围 \([0, s]\)
- \(x_t\)：带噪声的样本

#### 计算流程

**Step 1: 调整时间步维度**

```python
# 确保 timesteps 维度与 noise 匹配
while len(timesteps.shape) < len(noise.shape):
    timesteps = timesteps.unsqueeze(-1)
```

**维度扩展示例**：

```python
# 假设
timesteps.shape = (batch_size,)  # 例如 (32,)
noise.shape = (batch_size, seq_len, action_dim)  # 例如 (32, 8, 10)

# 第 1 次扩展
timesteps.unsqueeze(-1)  # (32, 1)

# 第 2 次扩展
timesteps.unsqueeze(-1)  # (32, 1, 1)

# 现在 len(timesteps.shape) == len(noise.shape) == 3
```

**Step 2: 广播到完整形状**

```python
timesteps = timesteps.expand_as(noise)
```

**广播示例**：

```python
# 扩展前
timesteps = [[0.5], [0.3], [0.7]]  # shape: (3, 1)

# 扩展后（假设 noise shape 是 (3, 2)）
timesteps = [[0.5, 0.5], 
             [0.3, 0.3], 
             [0.7, 0.7]]  # shape: (3, 2)
```

**Step 3: 线性插值加噪**

```python
return (1 - timesteps) * original_samples + timesteps * noise
```

**逐元素计算**：

```python
# 假设
original_samples = [[1.0, 2.0],   # 干净数据
                    [3.0, 4.0]]
                    
noise = [[0.1, 0.2],            # 噪声
         [0.3, 0.4]]
         
timesteps = [[0.5, 0.5],        # 时间步
             [0.3, 0.3]]

# 计算
noised = (1 - timesteps) * original_samples + timesteps * noise

# 样本 0, 维度 0:
# (1 - 0.5) * 1.0 + 0.5 * 0.1 = 0.5 * 1.0 + 0.5 * 0.1 = 0.55

# 样本 0, 维度 1:
# (1 - 0.5) * 2.0 + 0.5 * 0.2 = 0.5 * 2.0 + 0.5 * 0.2 = 1.1

# 样本 1, 维度 0:
# (1 - 0.3) * 3.0 + 0.3 * 0.3 = 0.7 * 3.0 + 0.3 * 0.3 = 2.19

# 样本 1, 维度 1:
# (1 - 0.3) * 4.0 + 0.3 * 0.4 = 0.7 * 4.0 + 0.3 * 0.4 = 2.92

# 结果
noised = [[0.55, 1.1],
          [2.19, 2.92]]
```

## 工作原理详解

### 训练流程

```python
# 1. 准备数据
original_samples = ...  # 真实的动作数据 x₀
noise = torch.randn_like(original_samples)  # 采样噪声 ε

# 2. 采样时间步
scheduler = FlowMatchingScheduler(sample_method="beta", s=0.999)
timesteps = scheduler.sample_t(batch_size)  # 采样 t ∈ [0, 0.999]

# 3. 添加噪声
noisy_samples = scheduler.add_noise(original_samples, noise, timesteps)
# 得到 x_t = (1-t) * x₀ + t * ε

# 4. 模型预测速度场
predicted_velocity = model(noisy_samples, timesteps)

# 5. 计算损失（目标：预测的速度应该等于 ε - x₀）
target_velocity = noise - original_samples
loss = MSE(predicted_velocity, target_velocity)

# 6. 反向传播
loss.backward()
optimizer.step()
```

### 推理流程（生成新动作）

```python
# 1. 从纯噪声开始
x_t = torch.randn(batch_size, seq_len, action_dim)  # x₁ ≈ 噪声

# 2. 逐步去噪（ODE 求解）
num_steps = 50
timesteps = torch.linspace(1.0, 0.0, num_steps)

for i in range(num_steps - 1):
    t = timesteps[i]
    
    # 预测速度场
    velocity = model(x_t, t)
    
    # ODE 步：x_{t-Δt} = x_t - Δt * velocity
    dt = timesteps[i] - timesteps[i+1]
    x_t = x_t - dt * velocity

# 3. 输出生成的动作
generated_actions = x_t  # x₀
```

## 数学原理

### Flow Matching 的核心思想

Flow Matching 学习一个**速度场** \(v_\theta(x, t)\)，满足：

\[
\frac{dx_t}{dt} = v_\theta(x_t, t)
\]

其中：
- \(x_t\) 是时间 \(t\) 的状态
- \(v_\theta\) 是神经网络预测的速度
- 边界条件：\(x_1 \sim \mathcal{N}(0, I)\)（噪声），\(x_0 \sim p_{data}\)（数据）

### 线性插值路径

Flow Matching 使用简单的**线性路径**：

\[
x_t = (1 - t) \cdot x_0 + t \cdot \epsilon
\]

对时间求导得到**目标速度**：

\[
\frac{dx_t}{dt} = \epsilon - x_0
\]

**训练目标**：

\[
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| v_\theta(x_t, t) - (\epsilon - x_0) \|^2 \right]
\]

### 为什么这样设计？

1. **简单高效**：线性路径比 Diffusion 的复杂路径更简单
2. **直接预测**：直接学习速度场，不需要复杂的噪声调度
3. **ODE 可逆**：推理时可以通过 ODE 求解器从噪声生成数据

## 参数 s 的作用

### s = 0.999 的意义

```python
self.s = 0.999  # 时间步阈值
```

**为什么不是 1.0？**

1. **数值稳定性**：
   - 当 \(t = 1.0\) 时，\(x_t = \epsilon\)（纯噪声）
   - 完全丢失数据信息，可能导致数值问题
   - \(t = 0.999\) 保留极少量数据信息

2. **训练效果**：
   - 避免模型在纯噪声状态下训练
   - 保证始终有微弱的数据信号

3. **实际影响**：
   - 0.999 vs 1.0 的差异极小（0.1%）
   - 主要为了数值稳定性

### 不同 s 值的效果对比

| s 值 | 采样范围 | 特点 | 适用场景 |
|------|----------|------|----------|
| 1.0 | [0, 1] | 包含纯噪声 | 理论完整，但可能不稳定 |
| 0.999 | [0, 0.999] | 保留微弱信号 | **推荐**，平衡稳定与性能 |
| 0.9 | [0, 0.9] | 更强数据信号 | 数据质量好时可用 |
| 0.5 | [0, 0.5] | 偏向干净数据 | 快速训练，但生成质量差 |

## 采样方法对比

### Beta vs Uniform

| 对比项 | Beta(1.5, 1.0) | Uniform(0, 1) |
|--------|----------------|---------------|
| **分布形状** | 右偏（偏向大值） | 均匀 |
| **反转后** | 左偏（偏向小值） | 均匀 |
| **采样偏好** | 更多靠近 0（干净数据） | 所有时间步等概率 |
| **训练效果** | 更稳定，收敛更快 | 可能需要更多训练步 |
| **推荐度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

### 可视化对比

```python
import torch
import matplotlib.pyplot as plt
from torch.distributions import Beta, Uniform

# 采样
beta_dist = Beta(torch.tensor([1.5]), torch.tensor([1.0]))
uniform_dist = Uniform(torch.tensor([0.0]), torch.tensor([1.0]))

beta_samples = 1 - beta_dist.sample((10000,))  # 反转
uniform_samples = uniform_dist.sample((10000,))

# 绘制直方图（示意）
# Beta:     ████▇▇▅▅▃▃▂▂ （偏向 0）
# Uniform:  ████▇▇▇▇▅▅▅▅ （均匀）
```

## 实际使用示例

### 完整训练循环

```python
import torch
from noise_schedulers import FlowMatchingScheduler

# 初始化调度器
scheduler = FlowMatchingScheduler(sample_method="beta", s=0.999)

# 训练数据
batch_size = 32
seq_len = 8
action_dim = 10

original_samples = torch.randn(batch_size, seq_len, action_dim)  # 真实动作

# 训练步骤
for step in range(num_steps):
    # 1. 采样噪声
    noise = torch.randn_like(original_samples)
    
    # 2. 采样时间步
    timesteps = scheduler.sample_t(batch_size)  # shape: (32,)
    
    # 3. 添加噪声
    noisy_samples = scheduler.add_noise(original_samples, noise, timesteps)
    # noisy_samples[i] = (1-t[i]) * original[i] + t[i] * noise[i]
    
    # 4. 模型预测
    predicted_velocity = model(noisy_samples, timesteps)
    
    # 5. 计算损失
    target_velocity = noise - original_samples
    loss = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)
    
    # 6. 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 与 ActionProjector 的配合

```
训练数据 (真实动作)
    ↓
[FlowMatchingScheduler.add_noise]
    ↓
噪声动作 x_t + 时间步 t
    ↓
[ActionProjector]
    ↓
投影到隐藏空间 → 输入 Transformer
    ↓
预测速度场 → 计算损失
```

## 与 Diffusion Scheduler 的对比

| 特性 | Flow Matching | Diffusion (DDPM) |
|------|---------------|------------------|
| **噪声路径** | 线性插值 | 复杂方差调度 |
| **预测目标** | 速度场 \(\epsilon - x_0\) | 噪声 \(\epsilon\) |
| **时间步** | 连续 \(t \in [0, 1]\) | 离散 \(t \in \{0, 1, ..., T\}\) |
| **推理方式** | ODE 求解 | 逐步去噪 |
| **实现复杂度** | 简单 | 较复杂 |
| **生成质量** | 好 | 非常好 |
| **生成速度** | 快（可大步长） | 慢（需多步） |

## 总结

FlowMatchingScheduler 的核心要点：

1. **线性插值加噪**：\(x_t = (1-t) \cdot x_0 + t \cdot \epsilon\)
2. **时间步采样**：Beta 分布优于均匀分布
3. **参数 s**：控制采样范围，0.999 保证数值稳定
4. **训练目标**：预测速度场 \(v_\theta(x_t, t) \approx \epsilon - x_0\)
5. **推理方式**：通过 ODE 从噪声逐步生成干净数据

这种设计使得 Flow Matching 比传统 Diffusion Model 更简洁高效，同时保持了强大的生成能力。
