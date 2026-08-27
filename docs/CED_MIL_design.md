# CED-MIL: Class-Conditional Evidence Decomposition for WSI Learning

## 完整方法设计文档

---

## 1. 问题定义

给定一张全切片图像 $S$，经 patch encoder 提取后得到实例集合：

$$\mathcal{X} = \{x_1, x_2, \ldots, x_N\}, \quad x_i \in \mathbb{R}^d$$

切片级标签 $y \in \{0, 1\}$（COAD=0, READ=1）。

**现有方法（ABMIL）** 学一个标量注意力 $a_i \in [0,1]$，聚合为：

$$z = \sum_{i=1}^{N} a_i x_i$$

然后 $\hat{y} = \text{Classifier}(z)$。

**问题**：单一 attention 无法区分"证据类型"，所有高 attention patch 被混为一谈。

---

## 2. 核心思想

**不要让 patch 只争一个 attention 分数，而是让每个 patch 回答"我属于哪类证据"。**

定义四种证据角色（Evidence Role）：

| 角色 | 符号 | 含义 |
|------|------|------|
| Class-0 Evidence | $r=0$ | 支持 COAD 的形态证据 |
| Class-1 Evidence | $r=1$ | 支持 READ 的形态证据 |
| Shared Evidence | $r=S$ | 两类共享的肿瘤/恶性证据 |
| Nuisance | $r=\varnothing$ | 背景、伪相关、噪声 |

每个 patch 被软分配到这四种角色，然后分角色聚合形成四个"证据原型"，最终基于原型之间的关系做分类。

---

## 3. 网络结构

### 总览

```
patch features x_i (N × d)
        │
   ┌────▼────┐
   │ Shared   │  (Linear → GELU → Linear → d')
   │ Encoder  │
   └────┬────┘
        │  h_i (N × d')
        │
   ┌────▼────────────────┐
   │ Evidence Role Gate  │  4-way softmax/sparsemax
   │ g(h_i) → (p0, p1,  │
   │           pS, pN)   │
   └────┬────────────────┘
        │
   ┌────▼──────────────────────────────────┐
   │ Role-Conditional Attention (×4)        │
   │                                        │
   │  For each role r ∈ {0, 1, S, ∅}:      │
   │    α_i^r = softmax(w_r · tanh(V_r h_i))│
   │    z_r = Σ (p_i^r · α_i^r · h_i)      │
   └────┬──────────────────────────────────┘
        │
        │  z_0, z_1, z_S, z_∅  (4 × d')
        │
   ┌────▼────────────────────────┐
   │ Evidence Contrast Classifier │
   │                              │
   │  h = [z_0, z_1, z_S,        │
   │       z_0 - z_1,            │
   │       z_S - z_∅]            │
   │                              │
   │  ŷ = MLP(h) → 2 classes     │
   └──────────────────────────────┘
```

### 3.1 Shared Encoder

将 patch 特征映射到工作空间：

$$h_i = \text{GELU}(W_1 x_i + b_1) W_2 + b_2, \quad h_i \in \mathbb{R}^{d'}$$

$d'$ 可取 256 或 384。这不是 patch encoder 本身，而是一个浅层投影。

### 3.2 Evidence Role Gate

对每个 patch 输出四维角色分配：

$$g_i = \text{softmax}(W_g h_i + b_g) \in \mathbb{R}^4$$

其中 $g_i = (p_i^0, p_i^1, p_i^S, p_i^\varnothing)$，$\sum_r p_i^r = 1$。

**含义**：每个 patch 对四种角色的隶属程度。

### 3.3 Role-Conditional Attention

对每种角色 $r$，独立计算 attention 并加权聚合：

$$\alpha_i^r = \frac{\exp(w_r^\top \tanh(V_r h_i))}{\sum_j \exp(w_r^\top \tanh(V_r h_j))}$$

$$z_r = \sum_{i=1}^{N} p_i^r \cdot \alpha_i^r \cdot h_i$$

注意两重权重：
- $p_i^r$：patch 属于角色 $r$ 的概率（gate 输出）
- $\alpha_i^r$：在该角色内部的重要性排序（attention）

这确保了：只有"确实属于该角色"的 patch 才对该原型有贡献。

### 3.4 Evidence Contrast Classifier

将四个原型组合成最终表示：

$$h_{slide} = [z_0 \| z_1 \| z_S \| (z_0 - z_1) \| (z_S - z_\varnothing)]$$

其中 $\|$ 为拼接，$(z_0 - z_1)$ 是**类条件对比**，$(z_S - z_\varnothing)$ 是**信号-噪声对比**。

$$\hat{y} = W_c \cdot \text{GELU}(W_h \cdot h_{slide} + b_h) + b_c$$

输出 2 维 logits。

---

## 4. 损失函数

### 4.1 主分类损失

$$\mathcal{L}_{cls} = \text{FocalLoss}(\hat{y}, y)$$

与现有 pipeline 一致。

### 4.2 证据分离损失（Evidence Separation Loss）

鼓励 $z_0$ 和 $z_1$ 在表示空间中可分：

$$\mathcal{L}_{sep} = \max(0, \; \delta - \|z_0 - z_1\|_2)$$

其中 $\delta$ 为 margin（可取 1.0–2.0）。

**含义**：如果两类证据太相似，给一个惩罚。

### 4.3 类条件对齐损失（Class-Conditional Alignment Loss）

对于标签 $y=0$（COAD）的 slide，$z_0$ 应该主导分类：

$$\mathcal{L}_{align} = -\log \frac{\exp(\text{sim}(z_y, z_{cls}))}{\sum_{r \in \{0,1\}} \exp(\text{sim}(z_r, z_{cls}))}$$

其中 $z_{cls}$ 是分类头倒数第二层表示，$\text{sim}$ 为余弦相似度。

**含义**：标签为 COAD 的 slide，COAD evidence 原型应该更接近分类决策。

### 4.4 噪声抑制损失（Nuisance Suppression Loss）

训练一个辅助分类头只看 $z_\varnothing$：

$$\hat{y}_{noise} = W_n z_\varnothing$$

$$\mathcal{L}_{noise} = -H(\hat{y}_{noise})$$

即最大化 $\hat{y}_{noise}$ 的熵（让它接近均匀分布 = 随机猜）。

**含义**：噪声原型不应含有类别信息。如果 $z_\varnothing$ 能分类，说明模型学到了 shortcut。

### 4.5 角色均衡正则（Role Balance Regularizer）

防止所有 patch 都分配到同一角色：

$$\mathcal{L}_{bal} = \text{KL}\left(\bar{g} \; \| \; \mathcal{U}(4)\right), \quad \bar{g} = \frac{1}{N}\sum_i g_i$$

鼓励四种角色的平均分配接近均匀。

### 4.6 总损失

$$\mathcal{L} = \mathcal{L}_{cls} + \lambda_{sep} \mathcal{L}_{sep} + \lambda_{align} \mathcal{L}_{align} + \lambda_{noise} \mathcal{L}_{noise} + \lambda_{bal} \mathcal{L}_{bal}$$

推荐初始值：$\lambda_{sep}=0.1$, $\lambda_{align}=0.1$, $\lambda_{noise}=0.05$, $\lambda_{bal}=0.01$。

---

## 5. 与 ABMIL / TransMIL 的本质区别

| 维度 | ABMIL | TransMIL | **CED-MIL** |
|------|-------|----------|-------------|
| attention 类型 | 单路标量 | 多头 self-attn | **4 路角色条件** |
| bag 表示 | 1 个 embedding | 1 个 embedding | **4 个证据原型** |
| 分类依据 | embedding→logit | embedding→logit | **原型对比关系** |
| 噪声处理 | 无 | 无 | **显式噪声分支+熵约束** |
| shortcut 防御 | 无 | 无 | **nuisance suppression** |
| 可解释性 | attention heatmap | attention map | **4 类角色 heatmap** |
| 参数量(估) | ~0.5M | ~3.9M | **~1.5M** |

---

## 6. 参数量估计

假设 $d=1024$, $d'=256$, num_classes=2：

| 模块 | 参数量 |
|------|--------|
| Shared Encoder | $1024 \times 256 + 256 \times 256 \approx 328K$ |
| Evidence Role Gate | $256 \times 4 \approx 1K$ |
| Role-Conditional Attention (×4) | $4 \times (256 \times 128 + 128) \approx 131K$ |
| Evidence Contrast Classifier | $(256 \times 5) \times 256 + 256 \times 2 \approx 328K$ |
| Aux noise head | $256 \times 2 \approx 0.5K$ |
| **总计** | **~789K** |

比 ABMIL 略大，比 TransMIL 轻得多，比 WSI-HINT 轻两个量级。

---

## 7. 消融实验设计

| 实验 | 配置 | 验证什么 |
|------|------|----------|
| A1 | ABMIL baseline | 基线 |
| A2 | CED-MIL, 2 roles (class0 + class1 only) | 多路 vs 单路 |
| A3 | CED-MIL, 3 roles (+shared) | shared branch 价值 |
| A4 | CED-MIL, 4 roles (+nuisance) | 噪声抑制价值 |
| A5 | A4 without $\mathcal{L}_{sep}$ | 分离损失的作用 |
| A6 | A4 without $\mathcal{L}_{noise}$ | 噪声损失的作用 |
| A7 | A4 without $\mathcal{L}_{bal}$ | 均衡正则的作用 |
| A8 | A4 + multi-encoder ensemble | 与集成兼容 |

---

## 8. 可解释性分析

CED-MIL 天然提供四类 heatmap：

对每个 patch $i$，可以可视化：
- $p_i^0 \cdot \alpha_i^0$：COAD 证据强度
- $p_i^1 \cdot \alpha_i^1$：READ 证据强度
- $p_i^S \cdot \alpha_i^S$：共享肿瘤证据
- $p_i^\varnothing \cdot \alpha_i^\varnothing$：噪声区域

这比 ABMIL 的单一 heatmap 更丰富，且可以分角色叠加在 WSI 上。

---

## 9. 实现计划

### 文件改动

| 文件 | 动作 |
|------|------|
| `src/wsi_hint/model/ced_mil.py` | **新建** — CED-MIL 模型 |
| `src/wsi_hint/training.py` | 添加 CED-MIL 辅助损失 |
| `src/wsi_hint/cli.py` | `_build_model` 加 "ced-mil" 分支 |
| `configs/ced_mil.yaml` | 新建配置 |

### MVP（最小可行版）

先实现模块 1-4 + 主分类损失，不加辅助损失，验证结构可行。  
再逐步加入 $\mathcal{L}_{sep}$、$\mathcal{L}_{noise}$、$\mathcal{L}_{bal}$。

---

## 10. 论文贡献声明（草稿）

1. 我们提出 **CED-MIL**，一种面向弱监督 WSI 分类的类条件证据分解学习机制，将切片内实例组织为四类语义角色并分别聚合。
2. 我们设计了 **噪声抑制损失**，显式防止模型依赖伪相关背景区域。
3. 我们在 TCGA COAD vs READ 这一高难度细粒度任务上，验证了证据分解优于单路注意力聚合，并提供了多角色可解释性分析。
