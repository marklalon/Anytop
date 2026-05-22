# AnyTop Diffusion 模型架构详解

## 模型配置 (train_tiny)

| 参数 | 值 |
|------|-----|
| 层数 (layers) | 4 |
| 隐藏维度 (latent_dim) | 128 |
| 前馈维度 (ff_size) | 1024 |
| 注意力头数 (num_heads) | 4 |
| 文本编码 | t5-base (t5_out_dim=768) |
| 关节数 (njoints) | 143 (max_joints) |
| 特征数 (nfeats) | 13 |

---

## 网络结构图

```
输入: [Batch, 143关节, 13特征, Frames] + Timestep: [Batch]
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    InputProcess                              │
│                    参数量: ~107K                             │
│                                                             │
│  ┌─────────────────────┐  ┌─────────────────────────────┐  │
│  │ T-pose 位置编码      │  │ 当前动作编码                 │  │
│  │ root_embedding       │  │ root_embedding              │  │
│  │ (13→128)             │  │ (13→128)                    │  │
│  │ tpos_root_embedding  │  │ tpos_joint_embedding        │  │
│  │ (13→128)             │  │ joint_embedding             │  │
│  │ tpos_joint_embedding │  │ (13→128)                    │  │
│  │ (13→128)             │  └─────────────────────────────┘  │
│  └─────────────────────┘                                   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ T5文本编码                                            │  │
│  │ text_embedding (768→128)                              │  │
│  │ 98,432 params                                         │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  + Sinusoidal Positional Embedding                         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                 Topology/Edge Embeddings                     │
│  (GraphMotionDecoder的内部组件，被每一层共享使用)             │
│         参数量: 3,072 (在每层Spatial Attention中使用)        │
│                                                             │
│  4×Embedding(6, 128):                                       │
│  ├─ topology_query_emb: 拓扑距离投影 Q侧                    │
│  ├─ topology_key_emb: 拓扑距离投影 K侧                      │
│  ├─ edge_query_emb: 关系类型投影 Q侧                        │
│  └─ edge_key_emb: 关系类型投影 K侧                          │
│                                                             │
│  ↓↓↓↓ 被传入每一层的Spatial Attention使用 ↓↓↓↓              │
└─────────────────────────────────────────────────────────────┘
    ↓                                  ↑
    │    (每层中使用这些嵌入)            │ (从父Decoder获取)
    │                                  │
│              GraphMotionDecoder (4层堆叠)                    │
│              总参数量: ~1.85M + 3.072K = ~1.85M              │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Layer 1 (每层 ~462K params)                          │  │
│  │                                                      │  │
│  │  ┌──────────────────────────────────────────────┐   │  │
│  │  │ Spatial Attention (GraphMultiHeadAttn)       │   │  │
│  │  │ 66,048 params                                │   │  │
│  │  │ 4×Linear(128→128) + Topology/Edge偏置       │   │  │
│  │  │ ← 使用上面的4个嵌入表作为注意力logits偏置   │   │  │
│  │  │                                              │   │  │
│  │  │ 结果: 显式编码父子、兄弟关系                  │   │  │
│  │  │      让肩膀運动更容易影响手臂                 │   │  │
│  │  └──────────────────────────────────────────────┘   │  │
│  │           ↓ LayerNorm(128) [256 params]              │  │
│  │           ↓ Residual                                 │  │
│  │                                                      │  │
│  │  ┌──────────────────────────────────────────────┐   │  │
│  │  │ Temporal Attention (MultiheadAttn)           │   │  │
│  │  │ 66,048 params                                │   │  │
│  │  │ 标准4头注意力 (不使用Topology/Edge嵌入)      │   │  │
│  │  │ 处理时间维度的连贯性                         │   │  │
│  │  │ 当前帧可看到过去和未来帧的信息               │   │  │
│  │  └──────────────────────────────────────────────┘   │  │
│  │           ↓ LayerNorm(128) [256 params]              │  │
│  │           ↓ Residual                                 │  │
│  │                                                      │  │
│  │  ┌──────────────────────────────────────────────┐   │  │
│  │  │ Feed Forward Network                         │   │  │
│  │  │ 263,168 params                               │   │  │
│  │  │ Linear(128→1024→128)                         │   │  │
│  │  │ GELU激活 + Dropout(0.1)                      │   │  │
│  │  │ 中间维度1024 = 128×8                         │   │  │
│  │  └──────────────────────────────────────────────┘   │  │
│  │           ↓ LayerNorm(128) [256 params]              │  │
│  │           ↓ Residual                                 │  │
│  │                                                      │  │
│  │  ┌──────────────────────────────────────────────┐   │  │
│  │  │ Reference Attention (可选)                   │   │  │
│  │  │ 66,048 params                                │   │  │
│  │  │ 标准MHA (不使用Topology/Edge嵌入)            │   │  │
│  │  │ Stage1: 禁用 (disable_reference_branch=True) │   │  │
│  │  │ Stage2: 启用，从参考动作提取信息             │   │  │
│  │  └──────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  Layer 2, 3, 4: 相同结构，都使用同一组Topology/Edge嵌入   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    OutputProcess                             │
│                    参数量: ~3.3K                             │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ root_dembedding: 128 → 13 (根关节) 1,677 params      │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ joint_dembedding: 128 → 13 (其他关节) 1,677 params   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
输出: [Batch, 143关节, 13特征, Frames] (去噪后的动作)
```

---

## 参数量统计表格

### InputProcess (~107K)

| 子组件 | 计算方式 | 参数量 |
|--------|---------|--------|
| root_embedding | Linear(13→128) | 1,792 |
| tpos_root_embedding | Linear(13→128) | 1,792 |
| joint_embedding | Linear(13→128) | 1,792 |
| tpos_joint_embedding | Linear(13→128) | 1,792 |
| text_embedding (T5) | Linear(768→128) | 98,432 |
| **小计** | | **107,200** |

### GraphMotionDecoder (4层) - 总 ~1.85M

包含：
- 4层GraphMotionDecoderLayer: 1,848,368
- Topology/Edge Embeddings (共享): 3,072
  └─ 这些嵌入表属于GraphMotionDecoder，4层都使用同一份，不会重复计算

| 组件 | 计算方式 | 参数量 |
|------|---------|--------|
| **每层 (~462K)** | | |
| Spatial Attn (GraphMultiHeadAttn) | 4×Linear(128→128) | 66,048 |
| Norm1 | LayerNorm(128) | 256 |
| Temporal Attn (MultiheadAttn) | MultiheadAttn(4 heads) | 66,048 |
| Norm2 | LayerNorm(128) | 256 |
| FFN | Linear(128→1024→128) | 263,168 |
| Norm3 | LayerNorm(128) | 256 |
| Ref Attn (可选) | MultiheadAttn | 66,048 |
| **Layer单层总计** | | **~462,092** |
| **4层总计** | | **1,848,368** |
| **Topology/Edge嵌入 (4层共享)** | 4×Embedding(6,128) | 3,072 |
| **GraphMotionDecoder总计** | | **1,851,440** |

### OutputProcess - 3,354

| 子组件 | 计算方式 | 参数量 |
|--------|---------|--------|
| root_dembedding | Linear(128→13) | 1,677 |
| joint_dembedding | Linear(128→13) | 1,677 |
| **小计** | | **3,354** |

### 总参数量

| 组件 | 参数量 |
|------|--------|
| InputProcess | 107,200 |
| GraphMotionDecoder (4层 + 共享嵌入) | 1,851,440 |
| OutputProcess | 3,354 |
| **总计 (不含ReferencePriorEncoder)** | **~1,961,994 (~2M)** |

---

## 架构关键点澄清

### Topology/Edge Embeddings的位置和数据流

**错误理解** ❌ (之前的结构图):
```
InputProcess → GraphMotionDecoder → Topology/Edge Embeddings → OutputProcess
                                    (下游，应该反馈？)
```

**正确理解** ✅ (实际代码):
```
                 ┌─ Topology/Edge Embeddings (4张表)
                 │                              
InputProcess → GraphMotionDecoder ─────────┐    OutputProcess
                │ Layer1 ┌────────────────┘
                │        ├─ Spatial Attn ← 使用这些嵌入作为Q,K的偏置
                │        ├─ Temporal Attn (不使用)
                │        ├─ FFN
                │        └─ Ref Attn (不使用)
                │ Layer2-4: 同样的结构，都使用相同的嵌入表
```

**关键点**：
1. **Topology/Edge Embeddings是GraphMotionDecoder的属性** (在__init中定义)
2. **每一层都共享这4张表** (不重复创建)
3. **只在Spatial Attention的前向计算中使用** (作为Q,K投影的偏置)
4. **没有"反馈"关系**，是平行的信息流

---

### 三种注意力机制与Topology/Edge Embeddings的关系

| 注意力类型 | 实现方式 | 使用Topology/Edge Embeddings | 作用 |
|-----------|--------|----------------------------|-----|
| **Spatial Attention** | GraphMultiHeadAttention (自定义) | ✅ 是 | 作为Q和K投影的偏置，修改节点间的注意力权重 |
| **Temporal Attention** | MultiheadAttention (标准PyTorch) | ❌ 否 | 标准注意力，只看特征值，不涉及骨骼拓扑 |
| **Reference Attention (Stage2可选)** | MultiheadAttention (标准PyTorch) | ❌ 否 | 标准注意力，只关注参考动作的信息 |

**关键结论**：
- **只有Spatial Attention消费Topology/Edge Embeddings**
- 这些嵌入通过**修改注意力logits**来引导骨骼约束
- Temporal Attention维持动作的时间连贯性，但不需要骨骼结构知识

---

### 1. InputProcess (~107K) - 输入编码

**输入形状**: [Batch, 143关节, 13特征, Frames]

**处理流程**:
```
├─ T-pose 位置编码 (T-pose 参考帧)
│  ├─ root_embedding: 13 → 128
│  └─ tpos_joint_embedding: 13 → 128
├─ 当前动作编码 (Current motion)
│  ├─ root_embedding: 13 → 128
│  └─ joint_embedding: 13 → 128
└─ T5文本编码 (骨骼名称)
   └─ text_embedding: 768 → 128 (来自T5-base)
```

**关键差异**: 为什么要双路径编码？
- **T-pose 分支**: 提供骨骼的绝对参考位置（对理解关节关系重要）
- **当前帧分支**: 编码实际的动作数据
- **T5 分支**: 通过文本嵌入传输骨骼语义信息（如"Arm", "Hand"等）

### 2. GraphMotionDecoder (4层) - 核心去噪

每层 **462K 参数**，堆叠 4 次 = **1.85M**

#### 2.1 Spatial Attention (空间注意) - 66K

```
处理骨骼本身的关节关系，使用Topology/Edge Embeddings:
├─ GraphMultiHeadAttention (自定义，不是标准MultiheadAttention)
│  ├─ linear_q: 128 → 128 (q投影)
│  ├─ linear_k: 128 → 128 (k投影)
│  ├─ linear_v: 128 → 128 (v投影)
│  └─ output_layer: 128 → 128 (输出投影)
│
└─ 关键: 将Topology/Edge Embeddings作为注意力偏置
   (这是与标准Transformer最大的区别)
```

**数据流详解 (Topology/Edge Embeddings是如何被消费的)**:

```
输入数据 (来自y字典):
├─ y['graph_dist']: [bs, njoints, njoints] 
│  └─ 值域: 0-5 (拓扑距离，表示节点间的最短路径长度)
│     0: self (同一个关节)
│     1: 父子或兄弟关系 (1跳)
│     2-5: 更远的距离
│     6: 无关系 ('far')
│
└─ y['joints_relations']: [bs, njoints, njoints]
   └─ 值域: 0-5 (关系类型)
      0: self (自己)
      1: parent (父关节)
      2: child (子关节)
      3: sibling (兄弟关节)
      4: no_relation (无直接关系)
      5: end_effector (末梢节)

Topology/Edge Embedding表:
├─ topology_query_emb: Embedding(6, 128)  [距离查询]
├─ topology_key_emb: Embedding(6, 128)    [距离键值]
├─ edge_query_emb: Embedding(6, 128)      [关系查询]
└─ edge_key_emb: Embedding(6, 128)        [关系键值]

在GraphMultiHeadAttention前向传播中的使用:

第1步: 投影Q、K、V到多头空间
  Q = linear_q(x)           // [bs, njoints, 128] → [bs, njoints, 128]
  K = linear_k(x)           // [bs, njoints, 128] → [bs, njoints, 128]
  V = linear_v(x)           // [bs, njoints, 128] → [bs, njoints, 128]
  重塑为多头: [batch, num_heads, seq_len, d_k]

第2步: 计算拓扑距离偏置 (使用 topology_query_emb & topology_key_emb)
  query_hop = matmul(Q, topology_query_emb.T)  
             // [batch, nheads, njoints, njoints] x [nheads, ??, d_k]
             // 结果: [batch, nheads, njoints, 6]
             
  query_hop = gather(query_hop, distance_matrix)
             // 使用y['graph_dist']作为索引
             // 从6个距离值中为每一对关节选择对应的嵌入
             // 结果: [batch, nheads, njoints, njoints, 1]
  
  同样计算 key_hop (使用 topology_key_emb)
  
  spatial_bias = query_hop + key_hop
                // [batch, nheads, njoints, njoints]

第3步: 计算关系类型偏置 (使用 edge_query_emb & edge_key_emb)
  query_edge = matmul(Q, edge_query_emb.T)
              // 同样的流程，但基于边类型
  
  edge_bias = query_edge + key_edge
             // [batch, nheads, njoints, njoints]

第4步: 将偏置融入注意力计算
  attention_logits = matmul(Q, K.T) + spatial_bias + edge_bias
                    // 这是关键！偏置被直接加到Q@K的结果上
                    
  attention_weights = softmax(attention_logits * scale)
                     // 使用修改后的logits计算注意力权重

第5步: 应用注意力到V
  output = matmul(attention_weights, V)
```

**具体例子**：
假设关节关系矩阵：
```
     关节0(肩) 关节1(上臂)  关节2(手肘)   ...
关节0    0        1           2       ...   (0=self, 1=child, 2=grandchild)
关节1    1        0           1       ...   (1=parent, 0=self, 1=child)
关节2    2        1           0       ...
...
```

当计算**肩→上臂**的注意力权重时：
1. 提取distance[肩, 上臂] = 1 (拓扑距离)
2. 提取edge[肩, 上臂] = 2 (child关系)
3. 使用索引1从topology_query_emb和topology_key_emb中选择对应的128维向量
4. 使用索引2从edge_query_emb和edge_key_emb中选择对应的128维向量
5. 这些偏置值被加到(Q[肩] @ K[上臂].T)上，修改最终的注意力权重
6. 结果：**肩关节的特征可以通过修改后的更高权重来影响上臂**

#### 2.2 Temporal Attention (时间注意) - 66K

```
处理动作在时间维度的连贯性:
├─ MultiheadAttention (标准4头，不使用拓扑/边嵌入)
│  ├─ in_proj: 128 → 384 (Q,K,V三合一)
│  └─ out_proj: 128 → 128
│
└─ 目的: 关键帧应互相注意
   (当前帧可看到过去和未来帧的信息)
   
⚠️  重要: Temporal Attention使用标准的MultiheadAttention，
   与Spatial Attention不同，它不会消费Topology/Edge Embeddings
```

#### 2.3 FFN (前馈网络) - 263K

```
逐位置的非线性变换:
├─ Linear(128 → 1024): 128×1024 + 1024 = 132,096
└─ Linear(1024 → 128): 1024×128 + 128 = 131,072
   
为什么1024这么大？
- 中间维度 = d_model × (ff_size/d_model) = 128 × 8 = 1024
- 这个8倍扩展是标准Transformer设计
- 在128维隐空间中"暂时展开"到1024维进行非线性处理
```

#### 2.4 Reference Attention (可选) - 66K

```
Stage1中**禁用** (disable_reference_branch=True)
Stage2中启用:
└─ 从参考动作(真实动作)中提取高置信区域的信息
```

### 3. Topology/Edge Embeddings (3K) - 骨骼图编码

**属于 GraphMotionDecoder 的内部组件，4层共享使用**

```
4个Embedding表（只被Spatial Attention使用）:
├─ topology_query_emb: Embedding(6 距离值, 128维)
│  └─ 在Spatial Attention中时，Q向量投影到拓扑距离空间
├─ topology_key_emb: Embedding(6 距离值, 128维)
│  └─ 在Spatial Attention中时，K向量投影到拓扑距离空间
├─ edge_query_emb: Embedding(6 关系类型, 128维)
│  └─ 在Spatial Attention中时，Q向量投影到关系类型空间
└─ edge_key_emb: Embedding(6 关系类型, 128维)
   └─ 在Spatial Attention中时，K向量投影到关系类型空间

关系类型 (6种):
0: self (自己)
1: parent (父关节)
2: child (子关节)
3: sibling (兄弟关节)
4: none (无直接关系)
5: end_effector (末梢节)

⚠️  使用说明:
- 这些嵌入在Spatial Attention中用作**注意力权重的偏置** (见上一节详细流程)
- Stage1 (train_tiny): value_emb=False → 只有4个表（query和key嵌入）
- 这是GraphMotionDecoder.__init()定义的属性
- 在forward()中，这4张表被传给每一层的forward()调用
- Layer1-4都使用相同的嵌入表，不会重复创建参数

### 4. OutputProcess (3.3K) - 输出解码

```
将隐空间映射回动作特征:
├─ root_dembedding: 128 → 13 (根关节)
└─ joint_dembedding: 128 → 13 (其他关节)

输出形状: [Batch, 143关节, 13特征, Frames]
(与输入相同)
```

---

## 总体参数量对比

```
Stage1 (train_tiny 配置):
├─ InputProcess:                     107,200
├─ GraphMotionDecoder (4层):       1,848,368
│  └─ 包含Topology/Edge嵌入 (4张表,共享):    3,072
├─ OutputProcess:                     3,354
├─ Total:                      ~1,961,994 (~2M parameters)

详细参数分布:
├─ 4层×(线性投影 + LayerNorm + Attention):  1,848,368
│  ├─ Spatial Attention (4层): 4×66,048 = 264,192
│  ├─ Temporal Attention (4层): 4×66,048 = 264,192
│  ├─ FFN (4层): 4×263,168 = 1,052,672
│  └─ LayerNorm等其他: 剩余
├─ Topology/Edge Embeddings (共享，不重复): 3,072
└─ 文本编码+输入输出: 110,554

对比其他模型:
├─ BERT-base: 110M (百倍大)
├─ ViT-base: 86M (四十倍大)
└─ 小型CNN: 5-10M (2-5倍大)

所以AnyTop是一个相对轻量级的模型！
```

---

## 关键设计特点

| 特点 | 说明 |
|------|------|
| **双路径输入** | T-pose + 当前帧 + T5语义，提供多角度上下文 |
| **图感知注意** | Spatial Attn 显式编码骨骼父子、兄弟关系 |
| **时空解耦** | 分别用 Spatial 和 Temporal Attn 处理，高效且可解释 |
| **轻量级** | 2M参数，易于训练和部署 |
| **模块化** | 各个组件独立，便于消融研究 |

---

## 数据流总结

```
原始骨骼数 (1-143)
    ↓
[padding to max_joints=143]
    ↓
InputProcess: 动作编码 + 位置编码 + 语义编码
             [Batch, 143, 13, Frames]
    ↓
GraphMotionDecoder (4层, 内含Topology/Edge嵌入表):
  ├─ Layer1:
  │  ├─ Spatial Attn: ← 使用topology_*_emb + edge_*_emb作为偏置
  │  ├─ Temporal Attn: 处理时间维度
  │  ├─ FFN: 非线性变换
  │  └─ Ref Attn (可选): 参考动作约束
  │
  ├─ Layer2-4: 相同结构
  │
  └─ 共享的Topology/Edge嵌入:
     ├─ topology_query/key_emb: 拓扑距离投影 (4×768参数)
     └─ edge_query/key_emb: 关系类型投影 (4×768参数)
    ↓
OutputProcess: 解码回动作空间
             [Batch, 143, 13, Frames]
    ↓
输出: 去噪后的动作
```

---

*生成时间: 2026-04-17 (已于2026-04-17修正架构关键点)*
