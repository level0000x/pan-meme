//! 插件系统模块 - TokenizerPlugin 注册与调用。
//!
//! 设计要点：
//! - TokenizerPlugin trait 使用关联函数（无 &self），规避 trait object / dyn 动态分发
//! - PluginRegistry 内部以函数指针 fn(&str) -> Vec<String> 存储 tokenize 逻辑，零虚表开销
//! - 插件名使用 &'static str，编译期常量，HashMap key 零分配

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// TokenizerPlugin trait
// ---------------------------------------------------------------------------

/// 分词插件 trait：所有 tokenizer 必须实现此 trait。
///
/// 注意：两个关联函数均不接收 `&self`，因此无需实例化即可调用，
/// 也不依赖虚函数表（vtable）进行动态分发。
pub trait TokenizerPlugin {
    /// 对输入文本进行分词，返回词元列表。
    fn tokenize(text: &str) -> Vec<String>;

    /// 返回插件名称（唯一标识）。
    fn name() -> &'static str;
}

// ---------------------------------------------------------------------------
// 内置 Tokenizer 实现
// ---------------------------------------------------------------------------

/// 逐字切分中文文本的 tokenizer：按单个中文字符拆分，去除非中文字符。
pub struct ChineseCharTokenizer;

impl TokenizerPlugin for ChineseCharTokenizer {
    fn tokenize(text: &str) -> Vec<String> {
        text.chars().filter(|&c| is_chinese(c)).map(|c| c.to_string()).collect()
    }

    fn name() -> &'static str {
        "chinese_char"
    }
}

/// 按空白字符切分的 tokenizer：等价于 `split_whitespace`。
pub struct WhitespaceTokenizer;

impl TokenizerPlugin for WhitespaceTokenizer {
    fn tokenize(text: &str) -> Vec<String> {
        text.split_whitespace().map(|s| s.to_string()).collect()
    }

    fn name() -> &'static str {
        "whitespace"
    }
}

// ---------------------------------------------------------------------------
// CJK 字符判定辅助
// ---------------------------------------------------------------------------

/// 判断字符是否属于 CJK 统一表意文字区间。
fn is_chinese(c: char) -> bool {
    matches!(
        c,
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
        | '\u{3400}'..='\u{4DBF}' // CJK Unified Ideographs Extension A
        | '\u{F900}'..='\u{FAFF}' // CJK Compatibility Ideographs
    )
}

// ---------------------------------------------------------------------------
// PluginRegistry
// ---------------------------------------------------------------------------

/// 插件注册表：以插件名为 key，存储 tokenize 函数指针。
///
/// ## 为什么不用 trait object？
///
/// `Box<dyn TokenizerPlugin>` 会产生 vtable 间接调用开销。
/// 此处将 tokenize 逻辑存储为 `fn(&str) -> Vec<String>` 函数指针，
/// 编译期即可确定调用目标，零运行时虚表开销。
pub struct PluginRegistry {
    tokenizers: HashMap<&'static str, fn(&str) -> Vec<String>>,
}

impl PluginRegistry {
    /// 创建一个空注册表。
    pub fn new() -> Self {
        PluginRegistry {
            tokenizers: HashMap::new(),
        }
    }

    /// 注册一个实现了 `TokenizerPlugin` 的类型。
    ///
    /// 泛型 `T: TokenizerPlugin` 在编译期单态化，调用点无间接跳转。
    pub fn register<T: TokenizerPlugin>(&mut self) {
        self.tokenizers.insert(T::name(), T::tokenize);
    }

    /// 根据插件名执行分词。若名称不存在返回 `None`。
    pub fn tokenize(&self, name: &str, text: &str) -> Option<Vec<String>> {
        self.tokenizers.get(name).map(|f| f(text))
    }

    /// 列出所有已注册的插件名称。
    pub fn list_plugins(&self) -> Vec<&str> {
        self.tokenizers.keys().copied().collect()
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// 内置插件初始化
// ---------------------------------------------------------------------------

/// 向注册表中注册所有内置 tokenizer。
///
/// 当前内置：
/// - `chinese_char`  → `ChineseCharTokenizer`
/// - `whitespace`    → `WhitespaceTokenizer`
pub fn register_builtin_plugins(registry: &mut PluginRegistry) {
    registry.register::<ChineseCharTokenizer>();
    registry.register::<WhitespaceTokenizer>();
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chinese_char_tokenizer() {
        let tokens = ChineseCharTokenizer::tokenize("你好world！中文ABC测试");
        assert_eq!(tokens, vec!["你", "好", "中", "文", "测", "试"]);
    }

    #[test]
    fn test_chinese_char_empty() {
        let tokens = ChineseCharTokenizer::tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_chinese_char_no_chinese() {
        let tokens = ChineseCharTokenizer::tokenize("hello world 123");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_whitespace_tokenizer() {
        let tokens = WhitespaceTokenizer::tokenize("hello  world\trust\nprogramming");
        assert_eq!(tokens, vec!["hello", "world", "rust", "programming"]);
    }

    #[test]
    fn test_whitespace_empty() {
        let tokens = WhitespaceTokenizer::tokenize("   ");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_plugin_registry_register_and_tokenize() {
        let mut reg = PluginRegistry::new();
        register_builtin_plugins(&mut reg);

        let tokens = reg.tokenize("chinese_char", "Rust编程语言")
            .expect("chinese_char should be registered");
        assert_eq!(tokens, vec!["编", "程", "语", "言"]);

        let tokens = reg.tokenize("whitespace", "hello world")
            .expect("whitespace should be registered");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_plugin_registry_unknown_name() {
        let mut reg = PluginRegistry::new();
        register_builtin_plugins(&mut reg);
        assert!(reg.tokenize("unknown", "text").is_none());
    }

    #[test]
    fn test_list_plugins() {
        let mut reg = PluginRegistry::new();
        register_builtin_plugins(&mut reg);

        let mut names = reg.list_plugins();
        names.sort();
        assert_eq!(names, vec!["chinese_char", "whitespace"]);
    }

    #[test]
    fn test_plugin_names() {
        assert_eq!(ChineseCharTokenizer::name(), "chinese_char");
        assert_eq!(WhitespaceTokenizer::name(), "whitespace");
    }
}
