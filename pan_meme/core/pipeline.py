"""管线编排器 — 定理5实现 + 推论5.1 逆映射
数学对应: Phi = Phi_D \\circ Phi_C \\circ Phi_B \\circ Phi_A (定理5: 复合双射)
"""

import json
import os
import pickle
import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from pan_meme.core.types import PipelineData, Credential
from pan_meme.module1_input.adapter import InputAdapter, InputConfig
from pan_meme.module2_geo.geometrizer import Geometrizer
from pan_meme.module3_meme.decomposer import MemeDecomposer
from pan_meme.module4_bind.binder import CredentialBinder
from pan_meme.module4_bind.hierarchy_hasher import HierarchyHasher


# ============================================================
# PipelineConfig — PB级新增: 分布式管线配置
# ============================================================


@dataclass
class PipelineConfig:
    """PB级新增: 管线编排配置, 支持 sequential/parallel/distributed 模式.

    数学对应: PB_ARCHITECTURE.md — 分布式管线编排.
    控制管线执行模式和各模块加速策略:
      - LSH 加速关系提取 (模块1)
      - Star closure 加速推理 (模块1)
      - Mapper 持久同调分解 (模块3)
      - CAS Merkle 哈希绑定 (模块4)
      - 并行 5D 映射 / 并行 ODE 求解
      - 检查点保存与恢复

    Attributes:
        mode:              执行模式 ("sequential" | "parallel" | "distributed").
        checkpoint_dir:    检查点目录, 用于保存中间产物 (JSON/Parquet).
        use_lsh:           是否启用 LSH 加速 (模块1 关系提取).
        use_star_closure:  是否启用 Star closure 加速 (模块1 推理).
        use_mapper:        是否启用 Mapper 持久同调 (模块3 分解).
        use_cas:           是否启用 CAS Merkle (模块4 哈希绑定).
        parallel_mapping:  是否并行 5D 映射 (模块2).
        parallel_ode:      是否并行 ODE 求解 (模块3).
        n_workers:         并行工作线程数 (-1 表示自动, 0 表示单线程).
    """
    mode: str = "sequential"
    checkpoint_dir: Optional[str] = None
    use_lsh: bool = False
    use_star_closure: bool = False
    use_mapper: bool = False
    use_cas: bool = False
    parallel_mapping: bool = False
    parallel_ode: bool = False
    n_workers: int = -1


class Pipeline:
    """定理5: Phi: I -> Q 正向管线 + Phi^{-1}: Q -> I 逆向管线"""

    def __init__(self, config: Dict[str, Any] = None, pipeline_config: Optional[PipelineConfig] = None):
        cfg = config or {}
        # ---- PB级新增: 存储管线配置 ----
        self.pipeline_config: PipelineConfig = pipeline_config or PipelineConfig()

        m1 = cfg.get("module1", {})
        # ---- PB级新增: 根据 PipelineConfig 配置 InputAdapter 加速选项 ----
        self.adapter = InputAdapter(InputConfig(
            cycle_mode=m1.get("cycle_mode", "converge"),
            cycle_max_rounds=m1.get("cycle_max_rounds", 20),
            threshold_candidates=m1.get("threshold_candidates",
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
            threshold_default=m1.get("threshold_default", 0.5),
            transitive_decay=m1.get("transitive_decay", 0.9),
            symmetric_decay=m1.get("symmetric_decay", 0.85),
            cooccurrence_min_neighbors=m1.get("cooccurrence_min_neighbors", 3),
            concept_max_levels=m1.get("concept_max_levels", 10),
            max_component_threshold=m1.get("max_component_threshold", 0.8),
            max_isolated_ratio=m1.get("max_isolated_ratio", 0.2),
            # ---- PB级新增: LSH 与 Star closure 标志 ----
            use_lsh=self.pipeline_config.use_lsh,
            use_star_closure=self.pipeline_config.use_star_closure,
        ))
        self.geometrizer = Geometrizer()
        # ---- PB级新增: 配置并行映射标志 ----
        if self.pipeline_config.parallel_mapping:
            self.geometrizer.parallel = True
            self.geometrizer.n_workers = self.pipeline_config.n_workers

        # ---- PB级新增: MemeDecomposer 配置 Mapper 同调 ----
        self.decomposer = MemeDecomposer()
        if self.pipeline_config.use_mapper:
            self.decomposer.use_mapper = True
        if self.pipeline_config.parallel_ode:
            self.decomposer.parallel_ode = True
            self.decomposer.n_workers = self.pipeline_config.n_workers

        # ---- PB级新增: CredentialBinder 配置 CAS Merkle ----
        self.binder = CredentialBinder()
        if self.pipeline_config.use_cas:
            self.binder.use_cas = True

        self.hasher = HierarchyHasher()
        self.config = cfg

    def run_forward(self, raw_input: Any, h_override: Optional[Dict] = None) -> PipelineData:
        """定理5正向: Phi(I) = Q -> Credential"""
        thr = h_override.get("T") if h_override else None

        # ---- PB级新增: 检查点目录创建 ----
        ckpt_dir = self.pipeline_config.checkpoint_dir

        # 模块一: I -> M (定理1+2)
        data = self.adapter.adapt(raw_input, thr=thr)

        # ---- PB级新增: 模块一检查点 {ckpt_dir}/psi_{timestamp}.json ----
        if ckpt_dir and data.psi:
            self._save_checkpoint(
                ckpt_dir, "psi", "json",
                data.psi.to_dict() if hasattr(data.psi, "to_dict") else {"nodes": data.psi.nodes},
            )

        # 累积哈希 — token层
        if data.psi and data.psi.nodes:
            for i, node_text in enumerate(data.psi.nodes):
                data.all_hash_nodes.append(self.hasher.hash_component(
                    layer="token", component_id=f"token_{i}",
                    canonical_data=json.dumps({"text": node_text}, sort_keys=True),
                    children=[], metadata={}
                ))

        # 累积哈希 — math_model层
        if data.math_model:
            data.all_hash_nodes.append(self.hasher.hash_math_model(data.math_model))

        # 模块二: M -> G (定理3)
        data = self.geometrizer.encode(data)

        # ---- PB级新增: 模块二检查点 {ckpt_dir}/geo_{timestamp}.pkl ----
        if ckpt_dir and data.geo:
            self._save_checkpoint(ckpt_dir, "geo", "pkl", data.geo)

        # 模块三: G -> Q (定理4)
        data = self.decomposer.decompose(data)

        # ---- PB级新增: 模块三检查点 {ckpt_dir}/meme_{timestamp}.json ----
        if ckpt_dir and data.meme_state:
            self._save_checkpoint(
                ckpt_dir, "meme", "json",
                data.meme_state.to_dict() if hasattr(data.meme_state, "to_dict") else {"memes": str(data.meme_state)},
            )

        # 累积哈希 — meme + composite层
        if data.meme_state:
            for i, meme in enumerate(data.meme_state.memes):
                data.all_hash_nodes.append(self.hasher.hash_meme(meme, i))
            data.all_hash_nodes.append(self.hasher.hash_composite(
                data.meme_state.Theta, data.meme_state.C))

        # 模块四: Q -> Credential
        meta = {}
        if data.meme_state:
            meta["original_size_bytes"] = -1
            meta["meme_count"] = len(data.meme_state.memes)
            meta["original_type"] = str(type(raw_input).__name__)
        data.metadata_override = meta  # 供binder使用（临时桥接）
        data = self.binder.bind(data)

        # ---- PB级新增: 模块四检查点 {ckpt_dir}/cred_{timestamp}.json ----
        if ckpt_dir and data.credential:
            self._save_checkpoint(
                ckpt_dir, "cred", "json",
                {"data_hash": data.credential.data_hash, "root_hash": data.credential.merkle_tree.root_hash},
            )

        return data

    def run_inverse(self, credential: Credential) -> Any:
        """推论5.1: Phi^{-1}(Phi(I)) = I"""
        meme = credential.meme_state
        geo = self.decomposer.reconstruct(meme)
        math = self.geometrizer.decode(geo)
        tokens = self.adapter.inverse(math.structure)
        return tokens

    # ================================================================
    # PB级新增: 检查点与恢复方法
    # ================================================================

    def _save_checkpoint(self, directory: str, stage: str, fmt: str, data: Any) -> str:
        """PB级新增: 保存管线中间产物检查点.

        每个检查点文件命名格式: {stage}_{timestamp}.{fmt}
        附带 hash 指纹用于完整性校验.

        Args:
            directory: 检查点根目录.
            stage:     阶段标识 (psi|geo|meme|cred).
            fmt:       序列化格式 (json|pkl).
            data:      待序列化的中间产物.

        Returns:
            str — 检查点文件的完整路径.
        """
        os.makedirs(directory, exist_ok=True)

        timestamp: str = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        fname: str = f"{stage}_{timestamp}"
        fpath: str = os.path.join(directory, f"{fname}.{fmt}")

        if fmt == "json":
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif fmt == "pkl":
            with open(fpath, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError(f"不支持的检查点格式: {fmt} (仅支持 json/pkl)")

        # ---- 写入指纹文件 (sha256 hash of checkpoint file) ----
        import hashlib
        with open(fpath, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        hpath: str = os.path.join(directory, f"{fname}.hash")
        with open(hpath, "w", encoding="utf-8") as f:
            f.write(file_hash)

        return fpath

    def run_from_checkpoint(
        self, checkpoint_path: str, start_module: int = 1
    ) -> PipelineData:
        """PB级新增: 从指定检查点恢复管线执行.

        数学对应: PB_ARCHITECTURE.md — 检查点恢复语义.
        支持从任意模块入口继续执行, 跳过已完成的阶段:
          - start_module=1: 从模块1 (输入适配) 开始 (默认, 完整执行)
          - start_module=2: 从模块2 (几何化) 继续, 需加载 psi checkpoint
          - start_module=3: 从模块3 (模因化) 继续, 需加载 psi + geo checkpoint
          - start_module=4: 从模块4 (哈希绑定) 继续, 需加载 psi + geo + meme checkpoint

        当前实现为存根版本: 支持 start_module=1 (完整执行) 和
        start_module=4 (从凭证恢复). 中间模块恢复需配合具体序列化格式.

        Args:
            checkpoint_path: 检查点文件路径或检查点目录.
            start_module:    起始模块编号 (1-4).

        Returns:
            PipelineData — 恢复后继续执行完成的管线数据.

        Raises:
            ValueError: 当 start_module 无效或 checkpoint_path 不存在时.
            NotImplementedError: 当 start_module 所求恢复路径尚未实现时.
        """
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"检查点路径不存在: {checkpoint_path}")

        if start_module not in (1, 2, 3, 4):
            raise ValueError(
                f"start_module 必须在 1-4 之间, 当前值: {start_module}"
            )

        # ---- start_module=1: 回退到完整 run_forward ----
        if start_module == 1:
            # 无需加载 checkpoint, 从原始输入开始
            raise NotImplementedError(
                "run_from_checkpoint: start_module=1 需提供原始输入, "
                "请使用 run_forward(raw_input) 执行完整管线."
            )

        # ---- 中间模块恢复: 存根 ----
        if start_module in (2, 3):
            raise NotImplementedError(
                f"run_from_checkpoint: start_module={start_module} 尚未实现. "
                "需从 {checkpoint_path} 加载中间产物并反序列化到 PipelineData."
            )

        # ---- start_module=4: 从凭证恢复 ----
        if start_module == 4:
            # 尝试从 checkpoint 加载 credential
            if os.path.isfile(checkpoint_path):
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    cred_data = json.load(f)
            else:
                # checkpoint_path 为目录, 查找最新的 cred checkpoint
                cred_files = sorted(
                    [f for f in os.listdir(checkpoint_path) if f.startswith("cred_") and f.endswith(".json")],
                    reverse=True,
                )
                if not cred_files:
                    raise ValueError(
                        f"checkpoint 目录 {checkpoint_path} 中无 cred_*.json 文件"
                    )
                with open(os.path.join(checkpoint_path, cred_files[0]), "r", encoding="utf-8") as f:
                    cred_data = json.load(f)

            # 构造最小 PipelineData 返回
            # 注意: 完整恢复需从其他 checkpoint 重建完整管线数据
            data = PipelineData()
            data.credential = Credential(
                data_hash=cred_data.get("data_hash", ""),
                merkle_tree=MerkleTree(
                    root_hash=cred_data.get("root_hash", ""),
                    nodes={},
                    leaf_index={},
                ),
                meme_state=None,
                metadata={},
            )
            return data

    def run_distributed(
        self, raw_input: Any, spark_session=None
    ) -> PipelineData:
        """PB级新增: Spark 分布式模式存根.

        数学对应: PB_ARCHITECTURE.md — 分布式管线执行.
        当前为存根实现, 回退到 sequential 执行.

        未来实现方向:
          - 使用 Spark RDD 并行化 LSH 关系提取
          - 使用 Spark MLlib 分布式 Star closure
          - 使用 Spark 分区并行 ODE 求解
          - 使用 Spark 广播变量共享 CAS Merkle 分区清单

        Args:
            raw_input:     原始输入数据.
            spark_session: SparkSession 实例 (可选, 未使用时回退).

        Returns:
            PipelineData — 管线执行结果 (回退到 sequential 模式).
        """
        # ---------------- 当前回退到 sequential 执行 ----------------
        return self.run_forward(raw_input)

    # ================================================================
    # 全局优化
    # ================================================================

    def run_with_optimization(self, raw_input: Any):
        """先全局优化再运行 — 假设0 + 定理5"""
        from pan_meme.engines.optimizer import GlobalOptimizer
        opt = GlobalOptimizer(self.config.get("optimizer", {}))
        result = opt.optimize(raw_input, self.run_forward)
        data = self.run_forward(raw_input, h_override=result.h_star)
        return data, result
