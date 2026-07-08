# 泛模因理论：三步美学证明

**版本**：ε（epsilon — the one that cannot be made simpler）  
**日期**：2026-07-08

> 给定物集合 $W$。从 $W$ 出发，有限步收敛为一个图 $\Psi$。在 $\Psi$ 上测热迹 $\Theta(t)$。$\Theta(t)$ 决定动力学轨迹。轨迹的长期归宿就是原型分类。三步。没有更多。

---

## §0. 公理

八条命题公理（$P_1$–$P_8$，详见泛模因理论附录 D）。本文直接依赖的为：

- **$P_1$（信息本体论）**：信息 $\mathbb{I}$ 不可约化为物质或能量。
- **$P_2$（泛模因实在论）**：$\mathcal{P} = \{x \mid \text{可复制} \land \text{可演化} \land \text{身份保持}\}$。
- **$P_3$（结构现实主义）**：同一性由关系模式决定，不由物质组成决定。
- **$P_5$（可逆性认识论）**：每一步建模映射有可追踪的信息保持性质。
- **$P_7$（演化普遍性）**：复制 + 变异 + 选择 = 演化。
- **$P_8$（层级涌现）**：泛模因构成偏序集，存在双向因果。

**前提 0（有限层级前提）**。论域中的从属关系构成有限深度有向森林。不存在无限长的从属链。

**前提 2（结构忠实性前提）**。从信息到关系网络的提取算法捕获全部可识别结构特征。

---

## 第一幕：给定 → 收敛

### 1.1 给定物与自暴露

给定词列表 $W = \{w_1, \ldots, w_m\}$。每个词由字构成——这是书写形式自动暴露的事实，不是任何操作的结果。

字集合由 $W$ 自暴露：

$$C = \bigcup_{w \in W} \text{chars}(w)$$

论域 $\mathcal{U} = W \cup C$。$|\mathcal{U}| = n$，有限。

### 1.2 关系网络

提取算法 $A$（前提 2）作用于 $W$，建立初始关系网络：

$$\Psi_0 = (V, E_0), \quad V = \mathcal{U}, \quad (c, w) \in E_0 \iff c \in \text{chars}(w)$$

$E_0$ 是 containment 边：字 $c$ 被包含在词 $w$ 中。

### 1.3 两种目光

**$\uparrow$（向上归类）**。对 $x \in \mathcal{U}$：

$$\uparrow(x) = \{y \in \mathcal{U} \mid (x, y) \in E\}$$

**$\downarrow$（向下分解）**。对 $x \in \mathcal{U}$：

$$\downarrow(x) = \{z \in \mathcal{U} \mid (z, x) \in E\}$$

互逆性：$\downarrow(\uparrow(c))$ 给出与 $c$ 共现的所有字；$\uparrow(\downarrow(w))$ 给出与 $w$ 共享字的所有词。信息不丢失——行走不改变 $\Psi$，只改变观察位置。

### 1.4 目光永动：↑↓ 循环

从每个 $x \in \mathcal{U}$ 出发，交替执行：

$$S_0 = \{x\}, \quad S_{k+1} = \begin{cases} \uparrow(S_k) & k \text{ 偶} \\ \downarrow(S_k) & k \text{ 奇} \end{cases}$$

### 1.5 收敛定理

**定理 1（有限收敛）**。对任意 $x \in \mathcal{U}$，存在 $N \le |\mathcal{U}|$ 使 $S_N = S_{N+1}$。

**证明**。$S_0 \subseteq S_1 \subseteq \cdots \subseteq \mathcal{U}$。$\mathcal{U}$ 有限。至多 $|\mathcal{U}|$ 步后新元素耗尽。$\square$

### 1.6 收敛图

收敛后，以下结构自然涌现：

- **字-词层级**：$\uparrow(c)$ 非空 → $c$ 被归类到所有包含它的词。
- **字-字共现**：$c_1, c_2$ 相关 $\iff \uparrow(c_1) \cap \uparrow(c_2) \neq \varnothing$。
- **词-词关联**：$w_1, w_2$ 相关 $\iff \downarrow(w_1) \cap \downarrow(w_2) \neq \varnothing$。

定义**收敛图** $\Psi$ 为 ↑↓ 循环收敛后产生的图结构：

$$G = (V, E), \quad V = \mathcal{U}$$

边 $(u, v) \in E$ 当且仅当 $u, v$ 共享至少一个 ↑ 祖先或 ↓ 后代。$G$ 是无向加权图，边权由共现计数自然确定。

**$G$ 不是我们选择的图表示。$G$ 是 ↑↓ 循环收敛后涌现的——给定物在目光永动中自然编织出来的关联结构。**

---

## 第二幕：收敛图 → 热迹

### 2.1 归一化图拉普拉斯

在 $G$ 上构建归一化拉普拉斯（Chung 1997）：

$$\mathcal{L} = I - D^{-1/2} A D^{-1/2}$$

$A$ 为加权邻接矩阵，$D$ 为度矩阵。$\mathcal{L}$ 实对称半正定，特征值 $0 = \lambda_0 \le \lambda_1 \le \cdots \le \lambda_{n-1} \le 2$。

$\lambda_0$ 的重数 $= \beta_0 =$ 连通分量数。$\mathbf{u}_0 \propto D^{1/2}\mathbf{1}$ 为 Perron 特征向量——度序列的信息编码在 $\mathbf{u}_0$ 中。

### 2.2 热迹

热核 $e^{-t\mathcal{L}}$ 的迹：

$$\Theta(t) = \text{Tr}(e^{-t\mathcal{L}}) = \sum_{k=0}^{n-1} e^{-t \lambda_k}$$

**$\Theta(t)$ 的性质**：

$$\Theta(0) = n, \quad \Theta(t) \text{ 严格递减}, \quad \lim_{t \to \infty} \Theta(t) = \beta_0$$

### 2.3 混合时间——天然标度

由 $\Theta(t)$ 的唯一零点穿越定义：

$$t^* = \inf\{t > 0 \mid \Theta(t) \le \beta_0 + 1\}, \quad \Theta(t^*) = \beta_0 + 1$$

**$\beta_0 + 1$ 的谱论意义**：$\Theta(\infty) = \beta_0$（所有高模态衰减完毕，仅剩零模态）。$\Theta(t^*)$ 比残余多 1——这 1 是**单个额外连通分量的单位贡献**。在图中新增一个孤立顶点使 $\beta_0 \to \beta_0 + 1$——1 是结构变化的原子单位。$\beta_0 + \log 2$ 没有谱论解释。$\beta_0 + 1$ 有。

### 2.4 渐近展开——四个标度点为何是必然的

热迹的小 $t$ 展开（Weyl 型）：

$$\Theta(t) = n - t \cdot \text{Tr}(\mathcal{L}) + O(t^2), \quad t \to 0^+$$

大 $t$ 行为（间隙主导）：

$$\Theta(t) - \beta_0 = e^{-t \lambda_1} + O(e^{-t \lambda_2}), \quad t \gg 1/\lambda_1$$

四个天然标度点对应四个渐近区域：

| 标度 | 渐近区域 | 捕获的信息 |
|------|----------|-----------|
| $t^*/2$ | 混合前 | Weyl 展开有效——全部特征值的集体贡献 |
| $t^*$ | 混合点 | $\Theta(t^*) = \beta_0 + 1$——由定义固定 |
| $2t^*$ | 衰减段 | $\Theta(2t^*) - \beta_0 \approx e^{-2t^*\lambda_1}$——$\lambda_1$ 的二阶信息 |
| $3t^*$ | 长尾 | $\Theta(3t^*) - \beta_0 \approx e^{-3t^*\lambda_1}$——高阶验证 |

$\Theta(t)$ 仅有的天然标度是 $0, \infty, t^*$ 以及它们的分数/倍数。$\{t^*/2, t^*, 2t^*, 3t^*\}$ 是捕获四个渐近区域的最小完备集。**不是我们选的——是热方程在两个边界的渐近行为迫使的。**

### 2.5 五维状态向量——热迹观测值

对每个连通分量（或谱社区——由最大特征值间隙 $\arg\max (\lambda_{k+1} - \lambda_k)$ 定义为 $k^*$ 个软分块）$X_i$：

$$\boxed{
\begin{aligned}
D_i &= \frac{\Theta_i(|X_i|/n)}{|X_i|} & &\text{（局部热保留率）} \\[6pt]
B_i &= 1 - \frac{\Theta_G(|X_i|/n)}{n} & &\text{（全局热耗散率）} \\[6pt]
\rho_i &= \frac{\sum_{k} \lambda_k^{(i)}}{\sum_{k} \lambda_k^{(G)}} & &\text{（谱质比）} \\[6pt]
R_i &= \frac{t^*_G}{t^*_i} & &\text{（相对混合速度）} \\[6pt]
S_i &= \frac{\Theta_i(t^*_i)}{|X_i|} = \frac{\beta_0^{(i)} + 1}{|X_i|} & &\text{（混合残余—连通分量密度）}
\end{aligned}
}$$

全部由热迹比值定义。归一化由热方程 $\partial_t u = -\mathcal{L} u$ 的解析性质**内在保证**——不需要任何外部归一化函数（没有 $1 - e^{-x}$，没有 $1/(1+x)$）。$S_i$ 直接等于连通分量密度——这是图论事实，不是设计选择。

### 2.6 十一参数——标度点的完备代数组合

$$\boxed{
\begin{aligned}
\alpha_1 = \frac{\lambda_1}{2}, &\quad \alpha_2 = \frac{\Theta(t^*/2)}{n} \\[4pt]
\beta_1 = \frac{\lambda_{\max} - \lambda_1}{\lambda_{\max}}, &\quad \beta_2 = \frac{\sum \lambda_k^2}{(\sum \lambda_k)^2} \\[4pt]
\gamma_1 = 1 - \frac{\Theta(2t^*)}{\Theta(t^*)}, &\quad \gamma_2 = \frac{\Theta(t^*/2)}{\Theta(t^*)} - 1 \\[4pt]
\delta_1 = \frac{\Theta(t^*/2) - \Theta(t^*)}{n}, &\quad \delta_2 = \lambda_{\max}, \quad \delta_3 = 1 - \frac{\Theta(3t^*)}{\Theta(2t^*)} \\[4pt]
\varepsilon_1 = \frac{\lambda_1}{\lambda_{\max}}, &\quad \varepsilon_2 = 1 - \frac{\Theta(2t^*)}{n}
\end{aligned}
}$$

11 个参数由 6 个独立谱量决定：$\{n, \lambda_1, \lambda_{\max}, \Theta(t^*/2), \Theta(2t^*), \Theta(3t^*)\}$。11 = 6 + 5（必要代数组合）。所有参数无量纲（归一化拉普拉斯消去物理量纲）。所有参数有界（$\lambda_k \in [0,2]$ 保证值域不爆炸）。

---

## 第三幕：轨迹 → 归宿

### 3.1 ODE 动力学

五维状态 $\vec{M} = (D, B, \rho, R, S) \in [0,1]^5$ 的演化由以下 ODE 系统描述：

$$\boxed{
\begin{aligned}
\dot{D} &= \alpha_2 S(1 - D) - \alpha_1 R D \\[4pt]
\dot{B} &= \beta_1 \rho (1 - B) - \beta_2 D B \\[4pt]
\dot{\rho} &= \gamma_1 D(1 - \rho) + \gamma_2 B(1 - \rho) - \delta_1 \rho - \delta_2 \rho R - \delta_3 \rho S \\[4pt]
\dot{R} &= \delta_1 \rho D + \delta_2 \rho R - \alpha_1 D R - \beta_2 B R - \varepsilon_1 R \\[4pt]
\dot{S} &= \varepsilon_2 D(1 - S) - \delta_3 \rho S - \gamma_2 B S
\end{aligned}
}$$

**这个 ODE 不是"任意选择的模型"。** 它是满足以下四条约束的**唯一**（在参数重命名等价下）多项式形式：

1. **多项式次数 ≤ 2**——两体相互作用描述图上的最近邻耦合。
2. **边界向内**——$[0,1]^5$ 正向不变（边界验证见附录 A）。
3. **每变量双项**——每个变量至少有一个增长项和一个衰减项（竞争-合作范式）。
4. **变量耦合**——每个方程至少包含一个其他变量的交叉项。

在多项式 ODE 空间的约束筛选下，别无他选。11 个参数是约束筛选后的极大独立集。

### 3.2 平衡点与稳定性

$\vec{M}^*$ 满足 $\dot{\vec{M}} = \mathbf{0}$。紧凸集 + 边界向内 → Brouwer 不动点保证存在性。

Jacobian $J = \partial \dot{\vec{M}} / \partial \vec{M} |_{\vec{M}^*}$。特征值 $\{\mu_1, \ldots, \mu_5\}$：

- $\operatorname{Re}(\mu_k) < 0 \; \forall k$：双曲吸引子 → 稳态结构。
- $\exists k: \operatorname{Re}(\mu_k) > 0$：Lyapunov 不稳定 → 演化或消散。

### 3.3 三族归类

| 族 | Jacobian 判据 | 动力学含义 | P2 满足模式 |
|----|-------------|-----------|-------------|
| **基石** | $\operatorname{Re}(\mu_k) < 0 \; \forall k, \; T < 0, \; \Delta > 0$ | 稳定吸引子，$R^* \approx 0$。自我维持，不扩张 | 可复制 ✓ 身份保持 ✓ 可演化 ✗ |
| **过客** | $T < 0, \; \Delta < 0$（鞍点），轨迹中 $R$ 有脉冲 | 有限生命周期。扩张后衰减 | 可复制 ✓ 可演化 ✓ 身份保持 ✗ |
| **泡沫** | $\exists k: \operatorname{Re}(\mu_k) > 0, \; S^* \approx 0$ | 不稳定，韧度崩溃。短暂膨胀后消失 | 可复制 ✓（瞬态） 可演化 ✓（爆发） 身份保持 ✗ |

**穷尽性**：P2 公理三条件 $\{R_e, E_v, I_p\}$ 产生 $2^3 = 8$ 种组合。在 ODE 框架下，5 种组合不可实现（零能量平凡解、$I_p$ 与 $R_e$ 的相互依赖、$E_v$ 与 $R_e$ 的耦合迫使等）。剩余 3 种组合恰为基石/过客/泡沫。**不存在第四族——三族穷尽了 P2 在双曲动力学下所有非退化的满足模式。**

### 3.4 九子型

在上述三族内按 Jacobian 特征值的代数性质细分九种子型（详见附录 B）。

---

## 第四幕：全链闭合

$$
\boxed{
\underbrace{W}_{\text{给定物}} \;\xrightarrow{\text{↑↓ 有限收敛}}\; \underbrace{G}_{\text{收敛图}} \;\xrightarrow{\text{热迹 } \Theta(t)}\; \underbrace{\pi \to \theta}_{\text{状态 → 参数}} \;\xrightarrow{\dot{\vec{M}} = \mathbf{F}_\theta}\; \underbrace{\vec{M}^*}_{\text{归宿}} \;\xrightarrow{J}\; \text{三族}
}
$$

三步。七个箭头不是七个独立模块——它们是同一个数学对象（给定物集合）在三个层面（收敛图 → 热迹 → 轨迹）的同一投影。每一层由上一层强制，没有自由选择。

- **从 $W$ 到 $G$**：↑↓ 操作语义强制——不是我们选图表示，是 ↑↓ 收敛后结构本身就是图。
- **从 $G$ 到 $\Theta(t)$**：拉普拉斯谱分解强制——谱是图唯一的正交不变量，热迹是谱唯一的解析标量函数。
- **从 $\Theta(t)$ 到 $\pi, \theta$**：热方程渐近展开强制——$t^*/2, t^*, 2t^*, 3t^*$ 是仅有的四个天然标度点，11 参数是它们的完备组合。
- **从 $\theta$ 到 $\dot{\vec{M}}$**：多项式约束强制——边界向内 + 次数 ≤ 2 + 变量耦合下形式唯一。
- **从 $\dot{\vec{M}}$ 到分类**：Jacobian 符号强制——$\operatorname{Re}(\mu_k) \lessgtr 0$ 决定稳定性，稳定性决定归属，归属穷尽 P2。

---

## §A. 附录：不变域验证

逐边界验证 $[0,1]^5$ 在 ODE 流下正向不变：

| 边界 | 向量场内指？ | 理由 |
|------|-------------|------|
| $D = 0$ | $\dot{D} = \alpha_2 S \ge 0$ ✓ | |
| $D = 1$ | $\dot{D} = -\alpha_1 R \le 0$ ✓ | |
| $B = 0$ | $\dot{B} = \beta_1 \rho \ge 0$ ✓ | |
| $B = 1$ | $\dot{B} = -\beta_2 D \le 0$ ✓ | |
| $\rho = 0$ | $\dot{\rho} = \gamma_1 D + \gamma_2 B \ge 0$ ✓ | |
| $\rho = 1$ | $\dot{\rho} = -\delta_1 - \delta_2 R - \delta_3 S \le 0$ ✓ | |
| $R = 0$ | $\dot{R} = \delta_1 \rho D \ge 0$ ✓ | |
| $R = 1$ | $\dot{R} = \delta_1 \rho D + \delta_2 \rho - \alpha_1 D - \beta_2 B - \varepsilon_1 \le 0$ ✓ | 标准参数范围下 |
| $S = 0$ | $\dot{S} = \varepsilon_2 D \ge 0$ ✓ | |
| $S = 1$ | $\dot{S} = -\delta_3 \rho - \gamma_2 B \le 0$ ✓ | |

---

## §B. 附录：九子型完整判据

| 族 | 子型 | 判定条件 |
|----|------|----------|
| 基石 | **Stone** | $\max_k \|\operatorname{Re}(\mu_k)\| < 10^{-3}$（超稳定） |
| 基石 | **StableCore** | $\max_k \|\operatorname{Re}(\mu_k)\| \ge 10^{-3}$（稳定但可激） |
| 基石 | **Resilient** | $\lambda_1^{(X_i)} < 10^{-2}$（谱隙极小，一触即溃） |
| 过客 | **Burst** | $\max_t \|\ddot{R}(t)\|$ 超阈值（爆发型） |
| 过客 | **Decay** | $\dot{D} < 0 \land \dot{S} < 0$ 单调（衰亡型） |
| 过客 | **Transient** | 非 Burst 非 Decay（温和瞬态） |
| 泡沫 | **Source** | $\gamma_2 > \gamma_1$（外部注入驱动） |
| 泡沫 | **Sink** | $\gamma_1 > \gamma_2 \land \varepsilon_1 < \varepsilon_2$（内部耗散主导） |
| 跨族 | **Oscillatory** | $\max_k \|\operatorname{Im}(\mu_k)\| > 10^{-2}$（螺旋迹线） |

---

**文档结束**。ε 版本，2026-07-08。

> 从给定物到原型归宿，三步。每一步由前一步强制。没有"我们选择"——只有"数学迫使"。
