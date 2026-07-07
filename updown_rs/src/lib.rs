// lib.rs — 库 target，供集成测试引用内部模块。
// binary 入口仍在 main.rs，两者共享同一套源码。

pub mod emergence;
pub mod encoding;
pub mod infra;
pub mod sealing;
