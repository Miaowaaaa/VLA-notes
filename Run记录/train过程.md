# DriveVLA-W0 Flow Matching 训练流程完整解析

## 一、训练流程概览

```
数据准备 → 配置加载 → 模型初始化 → 数据集构建 → 分布式训练 → 模型保存
```

---

## 二、关键步骤详细分解

### 阶段 1：环境配置与启动

#### 1.1 启动脚本
**文件**: `scripts/scripts_train/train_navsim_flow_matching.sh`

**核心配置**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `NGPUS` | 8 | 使用 8 张 GPU |
| `DATAPATH` | `navsim_emu_vla_256_144_trainval_pre_1s.pkl` | 训练数据路径 |
| `MODEL_NAME_OR_PATH` | `train_nuplan_6va_v0.2_multi_node` | 预训练 VLM 权重 |
| `MODEL_CONFIG_PATH` | `pi0_fast_video.json` | 模型架构配置 |
| `ACTION_TOKENIZER_PATH` | `pretrained_models/fast` | 动作分词器 |
| `EXP_NAME` | `train_navsim_flow_matching` | 实验名称 |

#### 1.2 分布式启动命令

```bash
torchrun \
  --nproc_per_node=8 \          # 8 卡并行
  --nnodes=1 \                  # 单节点
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=23456 \
  train/train_pi0.py \          # 训练入口
  [所有训练参数...]
```

**作用**：启动 8 个进程，每个进程绑定一张 GPU，协同训练。

---

### 阶段 2：配置加载与解析

#### 2.1 参数解析
**文件**: `utils/train_pi0.py` (第 172-173 行)

```python
parser = tf.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_into_dataclasses()
```

**三大参数类**：

| 类 | 作用 | 关键字段 |
|---|------|---------|
| `ModelArguments` | 模型路径配置 | `model_name_or_path`, `model_config_path` |
| `DataArguments` | 数据配置 | `data_path`, `actions`, `driving`, `action_frames` |
| `TrainingArguments` | 训练超参 | `learning_rate`, `max_steps`, `batch_size` |

#### 2.2 模型配置加载
**文件**: `configs/moe_fast_video.json`

```json
{
  "architectures": ["Emu3MoE"],
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "action_config": {
    "hidden_size": 4096,
    "num_hidden_layers": 2,
    "action_dim": 7
  }
}
```

**关键配置**：
- VLM: 32 层 Transformer
- Action Expert: 2 层 Transformer
- 动作维度：3 (dx, dy, dθ)

---

### 阶段 3：模型初始化

#### 3.1 加载 Pi0 配置
**文件**: `utils/train_pi0.py` (第 179-185 行)

```python
pi0_config = Emu3Pi0Config.from_pretrained(model_args.model_config_path)
update_configs(pi0_config, training_args, 
    ["image_area", "max_position_embeddings", 
     "action_loss_weight", "action_sample_steps", "freeze_vlm"])
```

#### 3.2 模型构建
**文件**: `utils/train_pi0.py` (第 108-136 行)

```python
model = load_model(model_args, pi0_config, training_args)
```

**内部流程**（`reference/Emu3/emu3/mllm/modeling_emu3.py`）：

```python
class Emu3Pi0:
    def __init__(self, config, pretrain_vlm_path):
        # 1. 加载预训练 VLM (Emu3MoE)
        self.vlm = Emu3MoE.from_pretrained(pretrain_vlm_path)
        
        # 2. 创建 Action Expert
        self.action_expert = Emu3Model(self.action_config)
        
        # 3. 状态投影器 (pre_action + cmd → hidden)
        self.state_projector = nn.Sequential(...)
        
        # 4. 动作投影器 (action → hidden)
        self.action_projector = ActionProjector(...)
        
        # 5. Flow Matching 调度器
        self.rf = FlowMatchingScheduler(sample_method="beta")
        
        # 6. 动作解码器 (hidden → action)
        self.action_decoder = FinalLayer(...)
        
        # 7. 创建共享层 (VLM + Action Expert 联合注意力)
        self.shared_layers = [
            Emu3Pi0SharedLayer(vlm_layer, action_layer)
            for vlm_layer, action_layer in zip(...)
        ]
```

**模型架构**：

```
┌──────────────────────────────────────┐
│         Emu3Pi0 Model               │
├──────────────────────────────────────┤
│  VLM (Emu3MoE - 32 层)              │
│  ↓                                   │
│  Shared Layers (联合注意力)          │
│  ↓                                   │
│  Action Expert (2 层)               │
│  ↓                                   │
│  Flow Matching Head                 │
│  ↓                                   │
│  Action Decoder (8 帧 × 3 维)       │
└──────────────────────────────────────┘
```

---

### 阶段 4：分词器与数据集构建

#### 4.1 加载分词器
**文件**: `utils/train_pi0.py` (第 192-197 行)

```python
tokenizer = Emu3Tokenizer.from_pretrained(
    model_args.model_name_or_path,
    model_max_length=1400,
    padding_side="right",
    use_fast=False,
)
```

#### 4.2 构建数据集
**文件**: `utils/train_pi0.py` (第 200 行)

```python
train_dataset, eval_dataset = get_dataset_split(data_args, tokenizer)
```

**数据集类**: `Emu3DrivingVAVADataset` (`utils/datasets.py` 第 602-750 行)

**数据加载流程**：

```python
class Emu3DrivingVAVADataset(Emu3SFTDataset):
    def __init__(self, args, tokenizer):
        # 1. 加载 pickle 数据
        with open(args.data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # 2. 加载动作分词器
        self.action_tokenizer = AutoProcessor.from_pretrained(
            args.action_tokenizer_path
        )
        
        # 3. 创建驾驶指令映射
        self.prompt2vec = {
            "go left": [1, 0, 0, 0],
            "go straight": [0, 1, 0, 0],
            "go right": [0, 0, 1, 0],
            "unknown": [0, 0, 0, 1]
        }
    
    def __getitem__(self, index):
        scene = self.data[index]
        
        # 1. 加载视觉 token (VQ codes)
        image_tokens = np.load(scene["image"][cur_idx])
        
        # 2. 加载动作并分词
        action = scene["action"]
        action_ids = self.action_tokenizer(action)
        
        # 3. 加载历史动作
        pre_action = scene["pre_1s_action"]
        
        # 4. 获取驾驶指令
        cmd = self.prompt2vec[scene["text"][cur_idx]]
        
        # 5. 构建输入
        return {
            "input_ids": ...,
            "attention_mask": ...,
            "action": action,
            "pre_action": pre_action,
            "cmd": cmd
        }
```

**数据划分**：
- **训练集**: 95%
- **验证集**: 5%

---

### 阶段 5：Trainer 设置

#### 5.1 创建 Trainer
**文件**: `utils/train_pi0.py` (第 211-217 行)

```python
trainer = tf.Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)
```

#### 5.2 DeepSpeed 配置
**文件**: `scripts/sft/zero3_offload.json`

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"}
  },
  "bf16": {"enabled": "auto"}
}
```

**作用**：
- **ZeRO-3**: 分布式优化器，节省显存
- **CPU Offload**: 将优化器和参数卸载到 CPU
- **BF16**: 混合精度训练

---

### 阶段 6：训练循环

#### 6.1 启动训练
**文件**: `utils/train_pi0.py` (第 221-224 行)

```python
if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    trainer.train(resume_from_checkpoint=True)  # 断点续训
else:
    trainer.train()  # 从头训练
```

#### 6.2 前向传播（核心）

**文件**: `reference/Emu3/emu3/mllm/modeling_emu3.py`

```python
def forward(self, input_ids, attention_mask, action, pre_action, cmd):
    # 1. VLM 处理视觉和文本
    vlm_hidden = self.vlm(input_ids, attention_mask)
    
    # 2. 投影历史动作和指令
    state_embedding = self.state_projector(
        torch.cat([pre_action.flatten(), cmd])
    )
    
    # 3. Flow Matching 训练
    # 采样时间步 t
    t = self.rf.sample_t(batch_size)
    
    # 添加噪声到真实动作
    z_t = (1 - t) * action + t * noise
    
    # 预测速度场
    v_pred = self.action_decoder(
        shared_layers(z_t, vlm_hidden, state_embedding)
    )
    
    # 计算损失
    loss = MSE(v_pred, noise - action)
    
    return loss
```

**Flow Matching 算法**：

```
真实动作 x₁
    ↓
添加噪声: z_t = (1-t)·x₁ + t·ε    (t ∈ [0, 1])
    ↓
VLM + Action Expert 预测 v_θ(z_t, t)
    ↓
损失: L = ||v_θ(z_t, t) - (ε - x₁)||²
```

#### 6.3 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `learning_rate` | 5e-5 | 学习率 |
| `max_steps` | 10000 | 最大训练步数 |
| `batch_size` | 12 | 每张卡 batch size |
| `gradient_accumulation` | 1 | 梯度累积 |
| `warmup_steps` | 50 | 预热步数 |
| `lr_scheduler` | cosine_with_min_lr | 余弦衰减 |
| `min_learning_rate` | 1e-6 | 最小学习率 |
| `action_loss_weight` | 1.0 | 动作损失权重 |
| `action_sample_steps` | 10 | Flow Matching 采样步数 |

---

### 阶段 7：模型保存

#### 7.1 保存检查点
**文件**: `utils/train_pi0.py` (第 227-229 行)

```python
trainer.save_state()          # 保存训练状态
torch.cuda.synchronize()      # 同步 GPU
trainer.save_model(output_dir)  # 保存模型权重
```

#### 7.2 保存内容

```
logs/train_navsim_flow_matching/
├── checkpoint-5000/
│   ├── model.safetensors      # 模型权重
│   ├── optimizer.pt           # 优化器状态
│   ├── scheduler.pt           # 学习率调度器
│   └── trainer_state.json     # 训练状态
├── config.json                # 模型配置
├── training_args.bin          # 训练参数
└── logs/                      # TensorBoard 日志
```

---

## 三、关键文件索引

| 文件 | 作用 | 代码行 |
|------|------|--------|
| `scripts/scripts_train/train_navsim_flow_matching.sh` | 训练启动脚本 | 1-82 |
| `utils/train_pi0.py` | 训练主入口 | 1-233 |
| `utils/datasets.py` | 数据集实现 | 602-750 |
| `reference/Emu3/emu3/mllm/modeling_emu3.py` | Emu3Pi0 模型 | 1799-2435 |
| `models/policy_head/flow_matching.py` | Flow Matching 调度器 | 1-64 |
| `models/tokenizer/action_tokenizer.py` | 动作分词器 | 1-140 |
| `scripts/sft/zero3_offload.json` | DeepSpeed 配置 | 1-37 |
| `configs/moe_fast_video.json` | 模型架构配置 | 1-57 |

---

## 四、训练数据流

```
Pickle 文件 (navsim_emu_vla_256_144_trainval_pre_1s.pkl)
    ↓
Emu3DrivingVAVADataset
    ↓
┌─────────────────────────────┐
│ 每个样本包含：               │
│ - image: VQ codes 路径列表  │
│ - action: (8, 3) 归一化动作 │
│ - pre_action: (3, 3) 历史   │
│ - text: 驾驶指令            │
│ - pre_1s_*: 前 1 秒数据     │
└─────────────────────────────┘
    ↓
ActionTokenizer → 离散 token IDs
    ↓
Emu3Tokenizer → 文本 token IDs
    ↓
合并 → input_ids + attention_mask
    ↓
Emu3Pi0 Model (Flow Matching)
    ↓
Action Loss (MSE)
    ↓
Backprop + DeepSpeed ZeRO-3
```

---

## 五、训练时间估算

| 配置 | 估算时间 |
|------|---------|
| **硬件** | 8× L20 (40GB) |
| **数据量** | ~100K 场景 |
| **Batch Size** | 12 × 8 = 96 |
| **Max Steps** | 10,000 |
| **预计时间** | ~2-3 天 |

---

## 六、关键创新点

### 6.1 Flow Matching vs 自回归

| 特性 | Flow Matching | 自回归 |
|------|---------------|--------|
| **生成方式** | 并行去噪 | 序列生成 |
| **推理速度** | 快（10 步） | 慢（逐 token） |
| **多样性** | 高 | 低 |
| **训练稳定性** | 好 | 一般 |

### 6.2 Pi0 架构

```
VLM 理解场景 → 提取特征
    ↓
联合注意力机制
    ↓
Action Expert 处理动作
    ↓
Flow Matching 生成连续动作
```

---

## 七、Flow Matching 训练方法详解

### 7.1 核心思想

Flow Matching 是一种生成模型训练方法，通过学习从噪声分布到数据分布的**速度场**来生成样本。

### 7.2 训练过程

```python
# 1. 采样时间步 t ~ Beta(1.5, 1.0)
t = self.rf.sample_t(batch_size)

# 2. 插值：在真实动作和噪声之间
z_t = (1 - t) * action + t * noise

# 3. 模型预测速度场
v_pred = model(z_t, t, condition)

# 4. 目标：预测噪声与真实动作的差值
v_target = noise - action

# 5. MSE 损失
loss = MSE(v_pred, v_target)
```

### 7.3 推理过程

```python
# 从纯噪声开始
z = torch.randn(batch_size, action_frames, action_dim)

# 逐步去噪（10 步）
dt = 1.0 / 10
for i in range(10, 0, -1):
    t = i / 10
    v_pred = model(z, t, condition)
    z = z - dt * v_pred  # 沿速度场移动

return z  # 最终生成的动作
```

---

## 八、分布式训练优化

### 8.1 DeepSpeed ZeRO-3

| 优化项 | 说明 |
|--------|------|
| **Optimizer States** | 分布式存储，节省 8 倍显存 |
| **Gradients** | 分片存储 |
| **Parameters** | 按需加载到 GPU |
| **CPU Offload** | 优化器和参数卸载到 CPU |

### 8.2 显存优化效果

| 配置 | 显存占用 |
|------|---------|
| 无优化 | ~80GB |
| ZeRO-3 | ~25GB |
| ZeRO-3 + Offload | ~15GB |

---

## 九、数据集处理细节

### 9.1 数据预处理流程

1. **VQ 编码**: Emu3 Vision Tokenizer 将图像编码为离散 token
2. **动作归一化**: 使用 q01/q99 分位数归一化到 [-1, 1]
3. **时间窗口**: 选取当前帧 + 前 3 帧历史
4. **动作离散化**: ActionTokenizer 将连续动作映射到 token space

### 9.2 数据增强

```python
# 随机翻转（数据增强）
if do_flip:
    action = action.copy()
    action[:, 1:] *= -1  # 翻转 y 和 yaw
    image_path = image_path.replace("/trainval_vq_codes/", "/trainval_vq_codes_flip/")
```

---

## 十、常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| **OOM** | 显存不足 | 减小 batch_size 或启用 CPU offload |
| **训练慢** | I/O 瓶颈 | 增加 num_workers 或使用 SSD |
| **loss 不降** | 学习率过大 | 降低 learning_rate 或增加 warmup |
| **断点续训失败** | 路径问题 | 检查 checkpoint 目录权限 |
| **分布式卡住** | NCCL 问题 | 设置 `NCCL_DEBUG=INFO` 调试 |

---

## 十一、训练监控

### 11.1 TensorBoard 日志

```bash
# 启动 TensorBoard
tensorboard --logdir logs/train_navsim_flow_matching
```

**监控指标**：
- `loss`: 总损失
- `action_loss`: 动作预测损失
- `learning_rate`: 当前学习率
- `grad_norm`: 梯度范数

### 11.2 验证指标

每 400 步在验证集上评估：
- **Action MSE**: 动作预测均方误差
- **Action MAE**: 动作预测平均绝对误差

---

## 十二、从训练到推理的完整流程

```
1. 训练阶段 (train_pi0.py)
   ↓
2. 模型保存 (logs/train_navsim_flow_matching/)
   ↓
3. 推理阶段 (inference_action_navsim_flow_matching_vava.py)
   ↓
4. 输出 JSON (json_output/*.json)
   ↓
5. 评估指标 (PDMS, Collision Rate, etc.)
```

---

## 附录：完整训练命令示例

```bash
#!/usr/bin/env bash

# 环境变量设置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO

# 训练参数
WORLD_SIZE=1
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=23456
NGPUS=8

DATAPATH='data/navsim/processed_data/meta/navsim_emu_vla_256_144_trainval_pre_1s.pkl'
ACTION_TOKENIZER_PATH="configs/fast"
EXP_NAME=train_navsim_flow_matching
MODEL_NAME_OR_PATH="/path/to/pretrained/vlm"
MODEL_CONFIG_PATH="configs/moe_fast_video.json"

# 启动训练
torchrun \
  --nproc_per_node=${NGPUS} \
  --nnodes=1 \
  --node_rank=${RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train/train_pi0.py \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --model_config_path ${MODEL_CONFIG_PATH} \
  --actions_format fast \
  --action_tokenizer_path ${ACTION_TOKENIZER_PATH} \
  --deepspeed scripts/sft/zero3_offload.json \
  --output_dir logs/${EXP_NAME} \
  --learning_rate 5e-5 \
  --null_prompt_prob 0.15 \
  --weight_decay 0.1 \
  --min_learning_rate 1e-6 \
  --max_grad_norm 5.0 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-6 \
  --bf16 True \
  --tf32 True \
  --data_path ${DATAPATH} \
  --freeze_vlm False \
  --max_steps 10000 \
  --dataloader_num_workers 12 \
  --lr_scheduler_type cosine_with_min_lr \
  --warmup_steps 50 \
  --per_device_train_batch_size 12 \
  --frames 1 \
  --action_frames 8 \
  --max_position_embeddings 1400 \
  --seed 0 \
  --logging_steps 10 \
  --gradient_checkpointing True \
  --gradient_accumulation_steps 1 \
  --save_strategy steps \
  --save_steps 5000 \
  --eval_strategy no \
  --apply_loss_on_only_vision True \
  --apply_loss_on_only_action False \
  --actions True \
  --use_gripper False \
  --driving True \
  --evaluation_strategy steps \
  --eval_steps 400 \
  --per_device_eval_batch_size 4 \
  --eval_accumulation_steps 1 \
  --use_previous_actions True \
  --action_dim 3 \
  --train_action_only False \
  --action_loss_weight 1.0 \
  --action_sample_steps 10 \
  --report_to tensorboard
```

---

**文档版本**: v1.0  
**更新日期**: 2026-05-23  
**项目**: DriveVLA-W0
