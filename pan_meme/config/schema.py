"""pan-meme 配置常量与校验"""

DEFAULT_CONFIG = {
    "module1": {
        "cycle_mode": "converge",
        "cycle_max_rounds": 20,
        "threshold_candidates": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "threshold_default": 0.5,
        "transitive_decay": 0.9,
        "symmetric_decay": 0.85,
        "cooccurrence_min_neighbors": 3,
        "concept_max_levels": 10,
        "max_component_threshold": 0.8,
        "max_isolated_ratio": 0.2,
    },
    "module3": {
        "split_strategies": ["connected_components"],
        "cell_count_max": 10000,
        "level_depth_max": 20,
        "invariant_count_max": 10,
    },
    "module4": {"hash_algo": "sha256"},
    "optimizer": {
        "lambda_reg": 0.01,
        "function_families": ["power", "exp", "sigmoid", "log", "piecewise"],
        "theta_bounds": {
            "alpha_1": [0.0, 1.0], "alpha_2": [0.0, 1.0],
            "beta_1": [0.0, 2.0], "beta_2": [0.0, 2.0],
            "gamma_1": [0.0, 2.0], "gamma_2": [0.0, 2.0],
            "delta_1": [0.0, 2.0], "delta_2": [0.0, 5.0], "delta_3": [0.0, 1.0],
            "epsilon_1": [0.0, 2.0], "epsilon_2": [0.0, 5.0],
        },
    },
    "ode": {
        "method": "RK45",
        "atol": 1e-8,
        "rtol": 1e-8,
        "max_step": 0.1,
        "t_span": [0.0, 100.0],
    },
    # ================================================================
    # PB级新增: PB 级优化配置段
    # ================================================================
    "pb_scale": {
        "enabled": False,                      # 是否启用 PB 级优化
        "relation_extractor": {
            "use_lsh": False,                  # LSH 近似加速
            "lsh_bands": 20,                   # LSH band 数
            "lsh_rows": 5,                     # 每 band 行数
            "lsh_window": 100,                 # 滑动窗口大小
        },
        "reasoner": {
            "use_star_closure": False,         # Star 传递闭包
            "star_max_iter": 20,               # 最大迭代次数
            "star_tol": 0.0001,                # 收敛容差
        },
        "geometry_split": {
            "use_mapper": False,               # Mapper 持久同调
            "mapper_intervals": 10,            # filter 区间数
            "mapper_overlap": 0.3,             # 区间重叠比
            "mapper_persistence_threshold": 0.1,  # 持久寿命阈值
            "use_pregel": False,               # Pregel CC
        },
        "mapping_5d": {
            "parallel": False,                 # 并行映射
            "n_workers": -1,                   # 工作线程数
        },
        "ode_solver": {
            "parallel": False,                 # 并行 ODE 求解
            "n_workers": -1,                   # 工作线程数
            "mode": "thread",                  # 并行模式 thread|process
        },
        "merkle_tree": {
            "use_cas": False,                  # CAS 内容寻址存储
            "partition_size": 1000000,         # 叶子分区大小
            "storage_backend": "memory",       # 存储后端 memory|s3|ipfs
        },
        "pipeline": {
            "checkpoint_enabled": False,       # 检查点保存
            "checkpoint_dir": None,            # 检查点目录
            "mode": "sequential",              # sequential|parallel|distributed
            "n_workers": -1,                   # 并行数
        },
        "gp_optimizer": {
            "n_initial": 20,                   # 初始采样点数
            "n_iterations": 100,               # 最大迭代次数
            "multi_fidelity": False,           # 多保真度优化
            "convergence_threshold": 1e-06,    # EI 收敛阈值
        },
    },
}


def validate(config: dict) -> list:
    """校验配置文件, 返回错误列表"""
    errors = []
    # 阈值候选集非空
    tc = config.get("module1", {}).get("threshold_candidates", [])
    if not tc:
        errors.append("threshold_candidates must be non-empty")
    for t in tc:
        if not (0.0 <= t <= 1.0):
            errors.append(f"threshold {t} out of [0,1]")
    # theta_bounds 格式
    bounds = config.get("optimizer", {}).get("theta_bounds", {})
    for k, v in bounds.items():
        if len(v) != 2 or v[0] >= v[1]:
            errors.append(f"theta_bounds.{k}: invalid range {v}")

    # ================================================================
    # PB级新增: PB 级配置校验
    # ================================================================
    pb = config.get("pb_scale", {})
    if pb.get("enabled", False):
        # LSH 参数
        re_cfg = pb.get("relation_extractor", {})
        if re_cfg.get("lsh_bands", 1) <= 0:
            errors.append("pb_scale.relation_extractor.lsh_bands must be > 0")
        if re_cfg.get("lsh_rows", 1) <= 0:
            errors.append("pb_scale.relation_extractor.lsh_rows must be > 0")

        # Mapper 参数
        gs_cfg = pb.get("geometry_split", {})
        if gs_cfg.get("mapper_intervals", 1) <= 0:
            errors.append("pb_scale.geometry_split.mapper_intervals must be > 0")
        mo = gs_cfg.get("mapper_overlap", 0.3)
        if not (0.0 <= mo <= 1.0):
            errors.append(
                f"pb_scale.geometry_split.mapper_overlap {mo} not in [0, 1]"
            )

        # Star closure 参数
        rsn_cfg = pb.get("reasoner", {})
        if rsn_cfg.get("star_max_iter", 1) <= 0:
            errors.append("pb_scale.reasoner.star_max_iter must be > 0")

        # Merkle 树参数
        mt_cfg = pb.get("merkle_tree", {})
        if mt_cfg.get("partition_size", 1) <= 0:
            errors.append("pb_scale.merkle_tree.partition_size must be > 0")
        sb = mt_cfg.get("storage_backend", "memory")
        if sb not in ("memory", "s3", "ipfs"):
            errors.append(
                f"pb_scale.merkle_tree.storage_backend '{sb}' 无效, "
                "必须为 memory|s3|ipfs"
            )

        # 管线模式
        pl_cfg = pb.get("pipeline", {})
        pipe_mode = pl_cfg.get("mode", "sequential")
        if pipe_mode not in ("sequential", "parallel", "distributed"):
            errors.append(
                f"pb_scale.pipeline.mode '{pipe_mode}' 无效, "
                "必须为 sequential|parallel|distributed"
            )

        # ODE 求解器并行模式
        ode_cfg = pb.get("ode_solver", {})
        ode_mode = ode_cfg.get("mode", "thread")
        if ode_mode not in ("thread", "process"):
            errors.append(
                f"pb_scale.ode_solver.mode '{ode_mode}' 无效, "
                "必须为 thread|process"
            )

    return errors


# ================================================================
# PB级新增: build_pipeline_config — 构造兼容 PipelineConfig 的配置字典
# ================================================================


def build_pipeline_config(config: dict) -> dict:
    """PB级新增: 从 DEFAULT_CONFIG 的 pb_scale 段提取扁平化管线配置.

    将嵌套的 pb_scale 配置转换为一维配置字典, 可直接传递给
    Pipeline.__init__ 的 pipeline_config 参数或用于构造 PipelineConfig dataclass.

    映射规则:
      pb_scale.relation_extractor.use_lsh       → use_lsh
      pb_scale.reasoner.use_star_closure        → use_star_closure
      pb_scale.geometry_split.use_mapper        → use_mapper
      pb_scale.merkle_tree.use_cas              → use_cas
      pb_scale.mapping_5d.parallel              → parallel_mapping
      pb_scale.ode_solver.parallel              → parallel_ode
      pb_scale.pipeline.mode                    → mode
      pb_scale.pipeline.n_workers               → n_workers
      pb_scale.pipeline.checkpoint_dir          → checkpoint_dir

    Args:
        config: 完整配置字典 (通常来自 DEFAULT_CONFIG).

    Returns:
        dict — 扁平的管线配置字典, 各键值可直接用于 PipelineConfig(...).
    """
    pb = config.get("pb_scale", {})

    re_cfg = pb.get("relation_extractor", {})
    rsn_cfg = pb.get("reasoner", {})
    gs_cfg = pb.get("geometry_split", {})
    mt_cfg = pb.get("merkle_tree", {})
    map_cfg = pb.get("mapping_5d", {})
    ode_cfg = pb.get("ode_solver", {})
    pl_cfg = pb.get("pipeline", {})

    return {
        "mode": pl_cfg.get("mode", "sequential"),
        "checkpoint_dir": pl_cfg.get("checkpoint_dir", None),
        "use_lsh": re_cfg.get("use_lsh", False),
        "use_star_closure": rsn_cfg.get("use_star_closure", False),
        "use_mapper": gs_cfg.get("use_mapper", False),
        "use_cas": mt_cfg.get("use_cas", False),
        "parallel_mapping": map_cfg.get("parallel", False),
        "parallel_ode": ode_cfg.get("parallel", False),
        "n_workers": pl_cfg.get("n_workers", -1),
    }
