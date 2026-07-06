"""
模块3 — 模因化层 (Memefication)

数学对应: 定义4 — Q = {X₁, X₂, ..., Xₙ, Θ, C}
  - 定理4: Φ_D: G → Q 双射 + 连通分量分解.
  - 前提5: K_i → X_i = (m_i, ξ_i) 双射 — 五维核心 + 扩展维度.

子模块:
  - geometry_split: 几何分解 (连通分量分解 + 合并).
  - mapping_5d:     五维映射 (K_i → MemeState + 逆映射).

PB 级并行模式 (PB_ARCHITECTURE.md 第3.4节 + 第4节):

  [Mapper 持久同调模式] — geometry_split.GeometrySplit:
    启用 use_mapper=True 后, 用 MapperHomology 持久同调估计 β₀,
    滤除噪声连接, 再执行精确连通分量分解.
    参数:
      - mapper_intervals (int, 默认 10): filter 区间数.
      - mapper_overlap (float, 默认 0.3): 区间重叠比例.
      - mapper_persistence_threshold (float, 默认 0.1): 持久寿命阈值.
    用法:
      splitter = GeometrySplit(use_mapper=True)
      sub_geos = splitter.split(geo)

  [Pregel CC 分布式模式] — geometry_split.GeometrySplit:
    启用 use_pregel=True 后, 用 PregelCC 分布式的 GAS 模型
    计算连通分量, 替代 scipy connected_components.
    用法:
      splitter = GeometrySplit(use_pregel=True)
      sub_geos = splitter.split(geo)

  [并行批处理映射] — mapping_5d.Mapping5D:
    map_batch(sub_geos, parallel=True, n_jobs=-1):
      用 ProcessPoolExecutor 并行执行五维映射.
      对应 PB_ARCHITECTURE.md 第4节 — Ray 并行 map, 10^7 并发.
    inverse_map_batch(memes, parallel=True, n_jobs=-1):
      并行逆映射, 从模因状态批量重建胞腔复形.

  [并行 ODE 求解池] — engines.ode_solver.ParallelODESolver:
    每模因独立积分, ThreadPoolExecutor (thread) /
    ProcessPoolExecutor (process) 两种模式.
    solve_streaming(...): 流式批量处理, 适合 PB 级数据集.
    对应 PB_ARCHITECTURE.md 第4节 — Ray Actor 池, 50K 并行.
"""
