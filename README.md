# ↑↓（updown）— 泛模因数据建模引擎

基于**泛模因理论（Pan-Meme Theory）**的信息-结构建模系统。将任意结构化文本映射为数学上可逆的五维模因表示，并通过完整的形式化管线实现从原始信息到可验证密码学凭证的闭合转换。

> **核心假说**：在足够丰富的跨领域数据输入下，自适应的结构提取与演化系统将收敛到一组确定的动力学方程。若收敛失败，则该假说被证伪。

---

## 理论基础

泛模因理论将一切可复制、可演化的信息-结构模式统摄于统一本体论框架。其思想渊源可追溯至三条学术脉络：

- **道金斯 (1976) 的模因论**：将观念、技能、行为视为文化演化的复制因子，通过模仿与学习在群体中传播
- **香农 (1948) 的信息论**：信息是消除不确定性的量度，但本理论补充了一个关键缺口——信息的"质"（结构、意义、演化）不应被悬置
- **阿瑟 (2009) 的技术演化理论**：技术是已有技术的组合，技术演化是自创生的，与生物演化共享同一逻辑结构

### 八个核心命题

| 命题 | 核心主张 |
|------|----------|
| **信息本体论** | 信息是现实的基本构成维度，与物质-能量并列；信息不可还原为物质或能量，且具有独立因果效力 |
| **泛模因实在论** | 一切可复制、可演化、可保持同一性的信息-结构模式皆为泛模因——基因、观念、制度、协议都是泛模因的不同实例 |
| **结构现实主义** | 真正持久存在的不是实体，而是实体之间的稳定关系模式——即结构；结构具有独立的因果效力 |
| **认知-建模统一性** | 认知本身就是建模活动；本理论将隐式认知建模转化为可显式执行的方法论步骤 |
| **可逆性认识论** | 模型可逆性是模型可信度的根本判据——Φ⁻¹(Φ(I)) ≡ I |
| **计算作为模因属性** | 图灵完备性是泛模因系统的逻辑延伸，而非外部引入的计算能力 |
| **演化普遍性** | 任何满足复制-变异-选择三要素的泛模因系统必然表现出演化行为，不依赖具体载体 |
| **层级涌现** | 低层泛模因通过组合形成高层泛模因；各层级之间没有本体论特权 |

### 数学保证（附录 D 严格证明）

| 定理 | 内容 |
|------|------|
| 定理 1-4 | 四阶段各自为双射（信息无损） |
| 定理 5 | 完整转换链 Φ = ΦD ∘ ΦC ∘ ΦB ∘ ΦA 为双射复合 |
| 推论 5.1 | Φ⁻¹(Φ(I)) = I（往返一致性，**完全可逆**） |
| 定理 6-8 | 5D ODE 系统解的存在唯一性、Lipschitz 连续、收敛性 |
| 定理 9 | 信息守恒：H(I) = H(Ψ) = H(M) = H(G) = H(Q) |
| 定理 10 | 九类演化原型的分类与收敛判定 |
| 假设 0 | H = T × F × Θ × N 全局优化：紧致度量空间上损失函数最小化解存在 |

---

## 系统架构

```
                    ┌──────────┐
                    │  原始输入  │ (文本、词列表)
                    └────┬─────┘
                         │
    ╔════════════════════╧════════════════════╗
    ║    Phase 1  浮现 (Emergence)           ║
    ║    I → Ψ → ↑↓循环 → 推理 → M=(S,F,C)  ║
    ║    词提取 · Jaccard 共现 · 伽罗瓦循环   ║
    ║    五类推理 · 概念层级 · 规则/约束推导   ║
    ╚════════════════════╤════════════════════╝
                         │
    ╔════════════════════╧════════════════════╗
    ║    Phase 2  编码 (Encoding)            ║
    ║    M → G = (K, g, Γ, R)               ║
    ║    CW 胞腔复形构造 · 离散梯度场          ║
    ║    标量场/向量场 · 几何不变量 · 可逆记录  ║
    ╚════════════════════╤════════════════════╝
                         │
    ╔════════════════════╧════════════════════╗
    ║    Phase 3  分解 (Decomposition)       ║
    ║    G → Q = ({Xᵢ}, Θ, C)               ║
    ║    Betti 连通分量分解 · 五维状态映射     ║
    ║    11 参数闭式解 · ξᵢ 扩展维度 · 耦合矩阵 ║
    ╚════════════════════╤════════════════════╝
                         │
    ╔════════════════════╧════════════════════╗
    ║    Phase 4  固化 (Binding)             ║
    ║    Q → 凭证 (SHA-256 + Merkle 树)       ║
    ║    分层哈希 · Merkle 证明 · 篡改定位     ║
    ╚════════════════════╤════════════════════╝
                         │
    ╔════════════════════╧════════════════════╗
    ║    Phase 5  演化 (Evolution)           ║
    ║    Q → ODE 轨迹 → 收敛分类 → 九类原型    ║
    ║    RKF45 自适应步长 · H=T×F×Θ×N 全局优化 ║
    ║    Louvain 多层社区检测 · 多层级结构域 S   ║
    ╚═════════════════════════════════════════╝
```

### 五维状态空间

每个模因由五维状态向量 M(t) = [D, B, ρ, R, S] 描述其在时间中的演化：

| 维度 | 符号 | 含义 | 方程核心 |
|------|------|------|----------|
| 内禀度 | D | 模因内部结构的复杂度与信息深度 | dD/dt = α₁·ΦD(D) - α₂·D·B² |
| 关联度 | B | 与其他模因的连接密度与耦合强度 | dB/dt = β₁·B·(1-B) - β₂·D·B |
| 能流密度 | ρ | 流经该模因的信息-结构能量 | dρ/dt = γ₁·D·B - γ₂·ρ |
| 演化速率 | R | 结构变化的速度与适应能力 | dR/dt = δ₁·ρ·R·(1-R) - δ₂·D·R - δ₃·R |
| 结构韧度 | S | 抵抗外部扰动的结构稳定性 | dS/dt = ε₁·D·ρ - ε₂·S |

### 九类演化原型

| 原型 | D | B | ρ | R | S | 典型特征 |
|------|---|---|---|---|---|------|
| 基石型 (Stone) | ↑ | ↑ | ↑ | → | ↑ | 深度稳固的核心结构 |
| 稳定核 (StableCore) | → | → | → | → | → | 长期稳态平衡 |
| 爆发型 (Burst) | ↓ | ↑↑ | ↑↑ | ↑↑ | ↓ | 快速扩张后收缩 |
| 衰减型 (Decay) | ↓ | ↓ | ↓ | → | ↓ | 渐近消亡的结构 |
| 振荡型 (Oscillatory) | ↻ | ↻ | ↻ | ↻ | ↻ | 周期性波动 |
| 瞬变型 (Transient) | ↑↓ | ↑↓ | ↑↓ | ↑↓ | ↑↓ | 短暂出现后消失 |
| 韧态 (Resilient) | → | ↓ | ↓ | → | ↑↑ | 低活动高稳定 |
| 源型 (Source) | → | ↑ | ↑↑ | ↑ | → | 信息-结构净产出者 |
| 汇型 (Sink) | → | ↓ | ↑ | ↓ | → | 信息-结构净吸收者 |

---

## 项目结构

```
模因/
├── README.md                         # 本文件
├── .gitignore
│
├── docs/                             # 项目文档
│   ├── 泛模因理论.md                    #   ~2500 行完整理论论文
│   ├── formal-concept-analysis-proof.md #   形式概念分析证明
│   └── ROSE-PAN-MEME-迁移规划.md        #   迁移路线图
│
├── scripts/                          # Python 运行脚本
│   ├── build_background.py           #   背景层级树构建
│   ├── run_dictionary.py             #   字典词 → pan-meme 管线
│   ├── run_dictionary_full.py        #   全集 ↑↓ 循环
│   ├── run_pipeline.py               #   完整四阶段管线
│   ├── rose_ingest.py                #   ROSE-SCA 一键摄入
│   └── tsv_bridge.py                 #   pan_meme → WikiLine TSV
│
├── data/                             # 数据文件
│   └── background_tree.json          #   新华字典背景层级树 (~3.5 MB)
│
├── pan_meme/                         # Python 原型工具包
│   ├── module1_input/                #   浮现：词→关系网络
│   ├── module2_geo/                  #   几何化：关系→CW复形
│   ├── module3_meme/                 #   模因化：CW→五维状态
│   ├── module4_bind/                 #   绑定：SHA-256 + 凭证
│   ├── core/                         #   核心类型与管线引擎
│   ├── engines/                      #   加速引擎 (LSH/ODE/GP)
│   ├── plugins/                      #   函数族与模态插件
│   ├── rust_native/                  #   Rust 原生扩展
│   ├── tests/                        #   端到端 + 模块测试
│   ├── PB_ARCHITECTURE.md            #   PB 级工业架构白皮书
│   └── TASKS.md                      #   任务清单
│
└── updown_rs/                        # Rust ↑↓ 引擎
    ├── Cargo.toml
    ├── README.md
    └── src/
        ├── main.rs                   #   主入口 (CLI + 五阶段流水线)
        ├── emergence/                #   Phase 0-1: 浮现
        │   ├── extractor.rs          #     词→关系网络 Ψ=A(I)
        │   ├── cycle.rs              #     ↑↓ 伽罗瓦循环
        │   ├── relations.rs          #     推理 · 概念层级 · 规则/约束
        │   └── louvain.rs            #     Louvain 多层社区检测
        ├── encoding/                 #   Phase 2-3: 编码
        │   ├── geometry.rs           #     CW 复形 · 梯度场 · 不变量
        │   └── decomposition.rs      #     Betti 分解 · 五维映射 · ξᵢ
        ├── sealing/                  #   Phase 4-5: 固化与演化
        │   ├── binding.rs            #     SHA-256 绑定 · 凭证
        │   ├── merkle.rs             #     分层 Merkle 树 · 篡改定位
        │   ├── ode.rs                #     RKF45 求解器 · 收敛分类
        │   └── optimizer.rs          #     H=T×F×Θ×N 全局优化
        └── infra/                    #   基础设施
            ├── tsv.rs                #     TSV 格式输出
            └── plugins.rs            #     多模态 Tokenizer 插件
```

---

## 快速开始

### 环境要求

- **Python** ≥ 3.10（pan_meme 工具包）
- **Rust** Toolchain ≥ 1.75（updown_rs 引擎）
- **依赖**：NumPy, NetworkX（Python）；零外部依赖（Rust，SHA-256 可选启用）

### 编译 Rust 引擎

```bash
cd updown_rs
cargo build --release

# 启用 SHA-256（可选）：
# 1. Cargo.toml: 取消注释 sha2 = "0.10"
# 2. src/sealing/binding.rs: 启用标注的 use sha2 行
```

### 运行

```bash
# 完整五阶段流水线
./target/release/updown input.txt

# 仅运行 Phase 1（浮现）
./target/release/updown input.txt --phase 1

# 指定输出目录和 Jaccard 阈值
./target/release/updown input.txt -o output/ -T 0.3

# 固定概念层级（5 层）
./target/release/updown input.txt --fixed 5

# Python 原型（字典词管线）
python scripts/run_dictionary.py
```

### 测试

```bash
cd updown_rs
cargo check --tests     # 类型检查（沙箱兼容）
cargo test              # 64 个单元测试（需本地编译环境）
```

---

## 技术栈

| 组件 | 语言 | 依赖 | 定位 |
|------|------|------|------|
| **updown_rs** | Rust | 零外部（sha2 可选） | 高性能原生引擎，五阶段全流程 |
| **pan_meme** | Python 3.10+ | NumPy, NetworkX | 原型验证工具包，含 PB 级架构白皮书 |
| **scripts** | Python | pan_meme | 一键运行脚本，新华字典管线 |

---

## 测试覆盖

| 模块 | 测试数 | 覆盖内容 |
|------|--------|----------|
| emergence/extractor | 0 | 依赖上游输入（main.rs 往返验证通过） |
| emergence/cycle | 1 | ↑↓ 循环统计 |
| emergence/relations | 11 | 关系自组织 · 五类推理 · 概念层级 · 规则/约束 · 自洽性 · 往返 |
| emergence/louvain | 6 | 模块度 · 单层检测 · 多层压缩 · 断开图 |
| encoding/geometry | 7 | CW复形构造 · 标量场 · 向量场 · 不变量 · 熵 · 可逆解码 |
| encoding/decomposition | 5 | Betti 分解 · 五维映射 · 11 参数 · 耦合 · ξᵢ 双射 |
| sealing/binding | 4 | SHA-256 哈希 · 凭证 · 指纹唯一性 |
| sealing/merkle | 10 | 空树 · 单叶 · 多叶 · 证明 · 层验证 · 篡改定位 · 大树 |
| sealing/ode | 9 | 5 种函数族 · RKF45 · 不变集 Ω=[0,1]⁵ · 平衡点 · 原型 |
| sealing/optimizer | 10 | H 空间搜索 · 5 族遍历 · 结果转 OdeConfig |
| infra/plugins | 2 | Tokenizer 注册与命名 |

---

## 设计原则

1. **零外部依赖（Rust）**：核心引擎不依赖任何第三方库（SHA-256 可选），保证完全可控性与可审计性
2. **可逆性至上**：建模的每一个步骤都有明确定义的逆操作——不存在信息丢弃步骤
3. **证明驱动开发**：每个算法直接对应论文附录 D 中的定理，确保实现与理论一致
4. **紧致优化**：所有自适应参数（阈值、函数族、参数、模因数）由假设 0 的全局优化模型统一确定，非手工调参

---

## 引用

本项目的理论基础来自泛模因理论（Pan-Meme Theory），完整论文见 `docs/泛模因理论.md`（约 2500 行，含附录 A-E 数学证明）。

相关学术背景：
- Dawkins, R. (1976). *The Selfish Gene*. Oxford University Press.
- Shannon, C. E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*.
- Arthur, W. B. (2009). *The Nature of Technology*. Free Press.
- Blondel, V. D. et al. (2008). "Fast unfolding of communities in large networks." *JSTAT*.
