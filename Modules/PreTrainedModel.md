## PreTrainedModel 数据结构
 PreTrainedModel 是 Hugging Face Transformers 库中所有预训练模型的基类（Base Class）。从代码中可以看到：
> class PreTrainedModel(nn.Module, EmbeddingAccessMixin, ModuleUtilsMixin, PushToHubMixin, PeftAdapterMixin):
- 1. 本质
它继承自 torch.nn.Module（PyTorch 的神经网络模块基类）
同时混入了多个 Mixin 类，提供额外功能
- 2. 核心功能
PreTrainedModel 提供了所有 Transformer 模型通用的功能：
- 📦 模型加载/保存
```
from_pretrained() - 从预训练权重加载模型
save_pretrained() - 保存模型权重和配置
push_to_hub() - 推送到 Hugging Face Hub
```
- ⚙️ 配置管理
  - 存储和管理模型配置（config 属性）
  - 自动加载对应的 PretrainedConfig
- 🔧 模型操作
```
resize_token_embeddings() - 调整词表大小
prune_heads() - 剪枝注意力头
tie_weights() - 绑定权重（如输入输出embedding共享）
```
- 🚀 训练支持
  - 梯度检查点（gradient checkpointing）
  - 设备映射（device_map）支持
  - 量化支持（quantization）
  - 分布式训练支持