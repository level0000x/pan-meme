# Related Work 文献综述

本文的理论基础横跨七个领域。以下按与本文核心主张的逻辑距离排序，从最直接的理论祖先到方法论工具。

---

## 1. 普适达尔文主义 (Universal Darwinism)

**Dawkins, R. (1983). "Universal Darwinism." In *Evolution from Molecules to Men*, ed. D. S. Bendall, Cambridge University Press.**

Dawkins 首次提出：如果宇宙中存在其他生命形式，其演化必然遵循达尔文主义的三条原则——复制、变异、选择。本文的核心主张直接继承于此：基因、观念、代码、技术——任何可复制的信息结构——都遵循同一套演化动力学。我们的贡献是将这一哲学主张操作化为可证伪的数学模型。

**Dennett, D. C. (1995). *Darwin's Dangerous Idea: Evolution and the Meanings of Life*. Simon & Schuster.**

Dennett 将自然选择提升为"通用酸"（universal acid）——一种可以腐蚀任何传统观念的算法过程。他认为演化是一个底层的、领域无关的算法。本文的 ODE 系统可以视为对这一哲学立场的数学化尝试。

**Campbell, D. T. (1960). "Blind Variation and Selective Retention in Creative Thought as in Other Knowledge Processes." *Psychological Review*, 67(6), 380–400.**

Campbell 的 BVSR（盲变异-选择性保留）模型是普适达尔文主义在认知科学中的奠基之作。本文的 Phase 1（FCA 形式背景生成）可视为 BVSR 的变异阶段，Phase 5（ODE 演化）可视为选择阶段。

**Hodgson, G. M. & Knudsen, T. (2010). *Darwin's Conjecture: The Search for General Principles of Social and Economic Evolution*. University of Chicago Press.**

Hodgson 和 Knudsen 将普适达尔文主义系统化，提出演化经济学中的"达尔文猜想"：社会-经济演化在更一般的意义上也是达尔文式的。他们区分了"一般达尔文主义"（general Darwinism）和"普适达尔文主义"（universal Darwinism）。本文的野心与他们一致，但数学工具不同——我们使用 CW 复形和 ODE 而非博弈论和制度分析。

---

## 2. 模因学 (Memetics)

**Dawkins, R. (1976). *The Selfish Gene*, Chapter 11. Oxford University Press.**

模因（meme）概念的原始出处。Dawkins 提出：文化演化单位（曲调、观念、口号、服装时尚）与基因类似，通过模仿在脑间传播，经历变异和选择。本文的"模因"定义扩展了 Dawkins 的概念——从"文化传播单位"扩展为"信息自组织单元"，由 CW 复形的拓扑结构而非语义内容定义。

**Blackmore, S. (1999). *The Meme Machine*. Oxford University Press.**

Blackmore 将模因学从隐喻推向理论框架，提出模因驱动了人类大脑容量、语言和宗教的演化。她认为模仿是模因传播的核心机制。本文与 Blackmore 的关键分歧在于：我们放弃"模仿"作为必要条件，转而用拓扑自组织（Louvain 社区检测）来定义模因边界。

**Dennett, D. C. (2017). *From Bacteria to Bach and Back: The Evolution of Minds*. W. W. Norton.**

Dennett 在此书中将模因重新定义为"文化演化中的信息单元"，并强调模因不需要"理解"——它们只需要被复制。本文的 Phase 1 n-gram 切分正是这一思想的工程实现：我们没有"理解"输入文本，只是切分和统计共现。

**Aunger, R. (2002). *The Electric Meme: A New Theory of How We Think*. Free Press.**

Aunger 尝试将模因学从类比推向科学，提出模因的神经基础假说。本文的路径不同：我们不寻找模因的物理基础，而是寻找模因的数学结构——拓扑不变量和 ODE 动力学。

---

## 3. 信息哲学 (Philosophy of Information)

**Floridi, L. (2011). *The Philosophy of Information*. Oxford University Press.**

Floridi 提出信息哲学作为"第一哲学"，主张信息是比知识、意义、真理更基础的概念。他区分了"环境信息"（environmental information）和"语义信息"（semantic information）。本文的 Phase 1 形式背景生成将文本转为纯结构信息（环境信息），而不依赖语义理解——这与 Floridi 的信息层级理论一致。

**Floridi, L. (2010). *Information: A Very Short Introduction*. Oxford University Press.**

Floridi 在此书中提出了信息的三层结构：数据 → 信息 → 知识。本文的管线可以映射为：Phase 1（数据 → 形式背景），Phase 2-3（形式背景 → 拓扑结构），Phase 4-5（拓扑结构 → 动力学知识）。但我们的终点不同——最终产物是 ODE 系统而非命题知识。

**Shannon, C. E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*, 27(3), 379–423.**

信息论的奠基之作。Shannon 的信息定义是纯统计的——信息量只与概率分布有关，与语义无关。本文的 Phase 1 复用了 Shannon 的哲学立场：结构优先于语义。我们使用 Shannon 熵来验证 Phase 3 的熵守恒（定理 9），但实验结果表明熵不守恒——这是对 Shannon 框架的一个有趣偏离。

---

## 4. 文化演化理论 (Cultural Evolution)

**Boyd, R. & Richerson, P. J. (1985). *Culture and the Evolutionary Process*. University of Chicago Press.**

文化演化理论的奠基之作。Boyd 和 Richerson 用种群遗传学模型（如 Price 方程）来描述文化传播，建立了"双重遗传理论"（dual inheritance theory）。本文的 Phase 5 ODE 系统与他们的种群模型在数学精神上一致——都是用微分方程描述演化——但我们的状态空间是五维的（D, B, ρ, R, S）而非基于种群频率。

**Cavalli-Sforza, L. L. & Feldman, M. W. (1981). *Cultural Transmission and Evolution: A Quantitative Approach*. Princeton University Press.**

文化传播的数学建模先驱。他们提出了水平传播、垂直传播和斜向传播的定量模型。本文的 n-gram 共现网络可以视为一种隐式的水平传播模型——共现的 n-gram 在信息空间中"传播"给对方。

**Laland, K. N., Odling-Smee, J., & Feldman, M. W. (2000). "Niche Construction, Biological Evolution, and Cultural Change." *Behavioral and Brain Sciences*, 23(1), 131–146.**

生态位构建理论认为，生物不是被动适应环境，而是主动改造环境，从而改变自身的选择压力。这与本文的 Phase 5 耦合 ODE 系统一致：每个模因的演化不仅受自身动力学驱动，还受其他模因的耦合影响——模因间的竞争-协作关系构成了彼此的"信息生态位"。

**Mesoudi, A. (2011). *Cultural Evolution: How Darwinian Theory Can Explain Human Culture and Synthesize the Social Sciences*. University of Chicago Press.**

Mesoudi 系统总结了文化演化领域的实证研究，包括实验室实验、历史分析和考古数据。本文可以视为向这一领域提供一种新的数学工具——CW 复形建模——来补充现有的统计和实验方法。

---

## 5. 演化动力学 (Evolutionary Dynamics)

**Nowak, M. A. (2006). *Evolutionary Dynamics: Exploring the Equations of Life*. Harvard University Press.**

Nowak 的演化动力学框架将复制子方程（replicator equation）和进化博弈论统一为数学框架，应用于癌症、语言和合作演化。本文的 Phase 5 ODE 系统直接继承了 Nowak 的数学风格——五个耦合的一阶微分方程描述信息结构的演化——但我们的变量（D, B, ρ, R, S）不是频率而是拓扑量。

**Hofbauer, J. & Sigmund, K. (1998). *Evolutionary Games and Population Dynamics*. Cambridge University Press.**

演化博弈论的标准教材。复制子动力学（replicator dynamics）是本文 ODE 系统的数学原型。关键区别：复制子动力学中的变量是种群频率（总和恒为 1），而本文的五维变量是拓扑量（无守恒约束）。

**Page, S. E. (2010). *Diversity and Complexity*. Princeton University Press.**

Page 研究了多样性如何影响复杂系统的鲁棒性和适应性。本文的 Phase 3 模因分解（Louvain 社区检测）天然产生多样性——不同模因代表不同的信息子结构，它们的耦合 ODE 描述了多样性的动力学后果。

---

## 6. 复杂系统与自组织 (Complex Systems & Self-Organization)

**Prigogine, I. & Stengers, I. (1984). *Order Out of Chaos: Man's New Dialogue with Nature*. Bantam Books.**

耗散结构理论：开放系统通过能量流维持有序状态。本文的 Phase 5 ODE 系统是一个耗散系统——值函数 L 驱动参数选择，系统在 t→∞ 收敛到 Stone 原型（S→1, D→1, B→0）。换言之，Stone 原型是五维状态空间中的耗散结构吸引子。

**Bak, P. (1996). *How Nature Works: The Science of Self-Organized Criticality*. Oxford University Press.**

自组织临界性（SOC）理论：复杂系统倾向于在有序与混沌之间达到平衡。本文的 Phase 2 梯度场 |∇φ| 可以视为系统"临界性"的度量——|∇φ| 越接近 1，系统越远离临界点。

**Holland, J. H. (1995). *Hidden Order: How Adaptation Builds Complexity*. Basic Books.**

Holland 的复杂适应系统（CAS）理论：大量主体通过简单规则交互产生涌现行为。本文的 Phase 1 n-gram 共现矩阵 + FCA 闭包就是一个 CAS 的实例——简单规则（共现统计 + 闭包运算）在规模效应下产生不可预测的全局结构（信息深度 depth）。

**Kauffman, S. A. (1993). *The Origins of Order: Self-Organization and Selection in Evolution*. Oxford University Press.**

Kauffman 提出自组织和自然选择共同驱动演化。本文的框架与这一立场一致：Phase 1-3 是自组织（FCA 闭包 + Louvain 分解），Phase 4-5 是选择（优化器 + ODE 积分）。

---

## 7. 拓扑数据分析 (Topological Data Analysis)

**Carlsson, G. (2009). "Topology and Data." *Bulletin of the American Mathematical Society*, 46(2), 255–308.**

TDA 的奠基性综述。Carlsson 提出持久同调（persistent homology）作为从数据中提取拓扑特征的工具。本文的 Phase 2 使用 CW 复形（而非持久同调）来编码信息结构，但共享核心哲学：拓扑不变量（Betti 数、欧拉示性数）比统计量携带更本质的信息。

**Edelsbrunner, H. & Harer, J. (2010). *Computational Topology: An Introduction*. American Mathematical Society.**

计算拓扑的标准教材。本文的 Phase 2 胞腔复形构建和欧拉示性数计算直接受益于计算拓扑的理论基础。

**Zomorodian, A. & Carlsson, G. (2005). "Computing Persistent Homology." *Discrete & Computational Geometry*, 33(2), 249–274.**

持久同调算法的经典论文。本文当前未使用持久同调（Phase 2 使用静态 CW 复形），但未来版本可能引入持久同调来捕捉信息结构在不同分辨率下的鲁棒性。

**Singh, G., Mémoli, F., & Carlsson, G. (2007). "Topological Methods for the Analysis of High Dimensional Data Sets and 3D Object Recognition." *Eurographics Symposium on Point-Based Graphics*.**

Mapper 算法的原始论文。Mapper 用透镜函数（lens function）和部分覆盖（partial clustering）将高维数据投影到低维拓扑图。本文的 Phase 1-3 管线（共现 → FCA 闭包 → Louvain 分解）在精神上类似于 Mapper 的多尺度拓扑分析。

---

## 本文与现有工作的定位

| 维度 | 现有工作 | 本文 |
|------|---------|------|
| 演化范围 | 单个领域（生物 OR 文化 OR 经济） | 跨领域统一建模 |
| 数学工具 | 种群遗传学 / 博弈论 / 统计 | CW 复形 + ODE |
| 模因定义 | 类比（"文化基因"） | 操作化（Louvain 社区） |
| 预测性 | 定性解释 | 定量可证伪 |
| 信息角色 | 认识论概念 | 物理量（熵守恒） |

本文的独特贡献在于将上述七个领域的核心洞见整合为一个可证伪的数学框架：普适达尔文主义提供哲学立场，信息哲学提供信息层级，模因学提供复制单元概念，文化演化提供定量建模传统，演化动力学提供 ODE 工具，复杂系统提供自组织视角，拓扑数据分析提供结构提取方法。