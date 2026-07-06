//! pan-meme-rs: Rust 原生扩展模块
//!
//! 热路径:
//!   - graph_ops: Warshall 传递闭包, 连通分量 (petgraph)
//!   - ode_rkf: RKF45 自适应步长 ODE 积分器 (比 scipy 快 3-10x)
//!   - sparse_linalg: 稀疏 CSR 矩阵乘法, 拉普拉斯零特征值 (nalgebra)

use pyo3::prelude::*;

/// Python 可调用的模块入口
#[pymodule]
fn pan_meme_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(warshall_transitive_closure, m)?)?;
    // m.add_function(wrap_pyfunction!(connected_components, m)?)?;
    // m.add_function(wrap_pyfunction!(rkf45_integrate, m)?)?;
    // 函数注册将在各子模块实现时逐步添加
    Ok(())
}
