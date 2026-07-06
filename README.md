# 模因 — 泛模因数据建模与知识体系化

基于泛模因理论的完整数据建模管线，从原始文本到数学可逆的模因表示，实现知识的形式化与体系化。

## 项目结构

```
模因/
├── README.md                    # 本文件
├── .gitignore
├── scripts/                     # 运行脚本
│   ├── build_background.py      #   构建背景层级树
│   ├── run_dictionary.py        #   字典词 → pan-meme 管线
│   ├── run_dictionary_full.py   #   全集 ↑↓ 循环
│   ├── run_pipeline.py          #   完整四阶段管线
│   ├── rose_ingest.py           #   ROSE-SCA 一键摄入
│   └── tsv_bridge.py            #   pan_meme → WikiLine TSV
├── data/                        # 数据文件
│   └── background_tree.json     #   新华字典背景层级树
├── docs/                        # 文档
│   ├── formal-concept-analysis-proof.md
│   └── ROSE-PAN-MEME-迁移规划.md
├── pan_meme/                    # Python 泛模因工具包
│   ├── module1_input/           #   浮现：词→关系网络
│   ├── module2_geo/             #   几何化：关系→CW复形
│   ├── module3_meme/            #   模因化：CW→五维状态
│   ├── module4_bind/            #   绑定：SHA-256+凭证
│   ├── core/                    #   核心类型/管线引擎
│   ├── engines/                 #   加速引擎(LSH/ODE/GP)
│   ├── plugins/                 #   函数族/模态插件
│   ├── rust_native/             #   Rust 原生扩展
│   └── tests/                   #   端到端+模块测试
└── updown_rs/                   # Rust ↑↓ 引擎
    ├── src/
    │   ├── emergence/           #   Phase 0-1: 浮现
    │   ├── encoding/            #   Phase 2-3: 编码
    │   ├── sealing/             #   Phase 4-5: 固化演化
    │   └── infra/               #   基础设施
    └── Cargo.toml
```

## 五阶段数据建模管线

| 阶段 | 输入 → 输出 | 数学对应 |
|------|-------------|----------|
| Phase 1 浮现 | 词列表 → M = (S, F, C) | §3.2 ↑↓ 循环、关系自组织 |
| Phase 2 编码 | M → G = (K, g, Γ, R) | §3.3 CW 复形、离散梯度 |
| Phase 3 分解 | G → Q = ({Xᵢ}, Θ, C) | §3.4 Betti 分解、五维映射 |
| Phase 4 固化 | Q → 凭证 (SHA-256) | §3.5 Merkle 树绑定 |
| Phase 5 演化 | Q → ODE 轨迹 | §D.3 动力系统仿真 |

## 运行

```bash
# 构建背景知识树
python scripts/build_background.py

# 字典词 → 完整管线
python scripts/run_dictionary.py

# 全集涌现结构
python scripts/run_dictionary_full.py

# Rust 引擎
cd updown_rs && cargo build --release
```

## 技术栈

| 组件 | 语言 |
|------|------|
| pan_meme 工具包 | Python + NumPy/NetworkX |
| updown_rs 引擎 | Rust |
