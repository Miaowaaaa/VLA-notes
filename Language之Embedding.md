## NLP 前置阶段：Token ID 的确定

在文本喂给模型之前，必须由 **Tokenizer（分词器）** 将字符串转化为数字 ID。

### 1. 第一步：构建词表（静态阶段）

在模型训练前，Tokenizer 统计海量文本语料，生成一个固定的词表（Vocabulary）字典。主流切分算法有：

* **单词级（Word-based）：** 按空格切分（缺点：词表太大，易遇到未登录词）。
* **字符级（Character-based）：** 按单字母切分（缺点：丢失词汇语义）。
* **子词级（Subword-based）：** 如 **BPE / WordPiece**（**现代大模型核心**）。高频词保持完整（如 `highest`），罕见词拆分为子词（如 `huggy` $\rightarrow$ `hug` + `gy`），完美平衡了词表大小与未知词问题。

### 2. 第二步：查表映射（动态阶段）

当输入一句文本时，Tokenizer 执行以下三步：

1. **切分（Tokenize）：** 将 `"I love you"` 切分为 Token 列表。
2. **查表（Mapping）：** 在词表中匹配每个 Token 的编号（如 `"I"` $\rightarrow 102$）。
3. **输出（Output）：** 输出整数列表 `[102, 3421, 1805]`，即为 **Token ID**，作为 `nn.Embedding` 的直接输入。

---

## nn.Embedding() 嵌入层

`nn.Embedding` 是将离散的整数 Token ID 转化为连续、稠密的低维物理向量的核心层。

### 1. 工作原理：可训练的查找表（Lookup Table）

* **底层结构：** 内部维护一个巨大的二维浮点数矩阵，形状为 `(num_embeddings, embedding_dim)`。
* **前向传播：** 接收到 Token ID 后，底层**不做任何矩阵乘法**，而是直接通过 C++ 指针偏移，快速“查表”抓取对应行的向量返回。
* **数学等价性：** 在数学原理上，它完全等价于 $\text{One-Hot 编码} \times \text{权重矩阵 } W$，但查表极大地节省了算力和显存。
* **语义学习：** 矩阵内部的数值是**可学习的参数**。随着下游任务的训练，语义相近的词（如“猫”和“狗”）在空间中的向量距离会被自动拉近。

### 2. 标准工程写法与参数约束

```python
nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, padding_idx=self.padding_idx)

```

* **`config.vocab_size`（行数）：** 词表总大小。限制了输入的 Token ID 范围必须在 `[0, vocab_size - 1]` 之间。
* **`config.hidden_size`（列数）：** * **等价关系：** 纯层结构中叫 `embedding_dim`，Transformer 全局架构中叫 `hidden_size`。为了无缝对接后续的 Transformer Block，两者必须**完全等价**。
* **设计规范（以词表大小 1024 为例）：**
* *上限约束：* 维度不宜过大（如 2048），防止小词表参数量爆炸导致过拟合。
* *下限约束：* 维度不宜过小（如 4），否则表达能力不足导致欠拟合。经验推荐 **`256`** 或 **`512`**。
* *硬件优化：* 强烈建议选择 **2 的幂次**（$64, 128, 256, 512$），有利于 GPU 硬件加速。
* *架构对齐：* 必须能被 Transformer 的多头注意力头数（`num_attention_heads`）**整除**。


* **`self.padding_idx`（填充特殊位）：** 指定变长文本对齐时用的占位符 ID（通常为 0）。该行向量会被**硬编码清零**，且在训练中**梯度冻结（不更新）**，确保无意义的填充不干扰模型的注意力机制。


SiLU（Swish）激活函数的最小值约等于 $-0.28$，这个值是通过对函数求导并寻找**极小值点**精确计算出来的。