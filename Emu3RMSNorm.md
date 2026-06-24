## Emu3RMSNorm
**核心思想**： **RMSNorm** (Root Mean Square Normalization) 是对传统 LayerNorm 的简化，去除了均值中心化步骤，只保留方差归一化。

##### 第 1 步：保存原始数据类型
```python
input_dtype = hidden_states.dtype
```
- 保存输入的数据类型（如 float16、bfloat16）
- 用于最后恢复原始精度

##### 第 2 步：转换为 float32（提升数值稳定性）
```python
hidden_states = hidden_states.to(torch.float32)
```
**为什么？**
- 归一化计算涉及除法和小数值
- float16 容易出现溢出或精度丢失
- float32 提供更高的数值精度

##### 第 3 步：计算均方根 (RMS)
```python
variance = hidden_states.pow(2).mean(-1, keepdim=True)
```

**详细展开**：
```python
# 假设有 3 个 token，每个 token 维度为 4
hidden_states = [[1.0, 2.0, 3.0, 4.0],      # token 1
                 [0.5, 1.5, 2.5, 3.5],      # token 2
                 [2.0, 4.0, 6.0, 8.0]]      # token 3

# 1. 平方
hidden_states.pow(2) = [[1.0, 4.0, 9.0, 16.0],
                        [0.25, 2.25, 6.25, 12.25],
                        [4.0, 16.0, 36.0, 64.0]]

# 2. 沿最后一个维度求平均
variance = [[7.5],              # (1+4+9+16)/4 = 7.5
            [5.25],             # (0.25+2.25+6.25+12.25)/4 = 5.25
            [30.0]]             # (4+16+36+64)/4 = 30.0
```

##### 第 4 步：归一化
```python
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
```

**等价于数学公式**：
$$\bar{x}_i = \frac{x_i}{\sqrt{\text{RMS}(x)^2 + \epsilon}}$$

其中：
$$\text{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2}$$

**详细计算**：
```python
# 假设 variance = 7.5, eps = 1e-6
rsqrt(7.5 + 1e-6) = 1/sqrt(7.5) ≈ 0.3651

# 归一化
normalized = [1.0, 2.0, 3.0, 4.0] * 0.3651
           = [0.3651, 0.7303, 1.0954, 1.4605]
```

##### 第 5 步：可学习的缩放
```python
return self.weight * hidden_states.to(input_dtype)
```

```python
# self.weight 是可学习参数，形状为 [hidden_size]
# 初始化为 1.0，训练过程中会学习最优的缩放因子

# 恢复原始数据类型（如 float16）
output = weight * normalized.to(input_dtype)
```

##### 完整数学公式

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma$$

其中：
- $\text{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}$
- $\gamma$ 是可学习的缩放参数 (`self.weight`)
- $\odot$ 表示逐元素相乘

##### RMSNorm vs LayerNorm 对比

| 特性 | LayerNorm | RMSNorm |
|------|-----------|---------|
| **计算** | 减去均值 + 除以标准差 | 只除以 RMS |
| **公式** | $\frac{x-\mu}{\sigma} \odot \gamma + \beta$ | $\frac{x}{\text{RMS}} \odot \gamma$ |
| **可学习偏置** | 有 ($\beta$) | 无 |
| **计算量** | 较大 | 较小（快 ~7-64%） |
| **效果** | 基准 | 几乎相同 |

**LayerNorm 计算**：
```python
mean = x.mean(-1, keepdim=True)
var = x.var(-1, keepdim=True)
x = (x - mean) / sqrt(var + eps)
x = x * weight + bias
```

**RMSNorm 计算**：
```python
var = (x ** 2).mean(-1, keepdim=True)
x = x / sqrt(var + eps)
x = x * weight
```

##### 为什么大模型都用 RMSNorm？

1. **效率更高**：少了均值计算和减法操作
2. **效果相当**：论文证明性能几乎相同
3. **更简洁**：少一个可学习参数 (`bias`)
4. **数值稳定**：配合 float32 计算，避免精度问题

##### 实际应用示例

```python
# 配置
hidden_size = 4096
eps = 1e-6

# 创建 RMSNorm 层
norm = Emu3RMSNorm(hidden_size=4096, eps=1e-6)

# 输入：batch_size=2, seq_len=100, hidden=4096
x = torch.randn(2, 100, 4096, dtype=torch.float16)

# 输出：相同形状，但已归一化
output = norm(x)  # shape: [2, 100, 4096]
```

##### 在 Transformer 中的位置

```
Input → RMSNorm → Attention → Residual → RMSNorm → MLP → Residual → Output
        ↑                              ↑
     归一化输入                     归一化输入
```

RMSNorm 通常应用在：
- **Attention 之前**（Pre-Norm）
- **MLP 之前**（Pre-Norm）