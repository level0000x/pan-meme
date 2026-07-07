//! WikiLine TSV 输出 — 泛模因理论 §6.2.2
//!
//! 每行 (entity, predicate, target) 三元组

use std::fs::File;
use std::io::{Write, BufWriter};

/// TSV 写入器
pub struct TsvWriter {
    writer: BufWriter<File>,
}

impl TsvWriter {
    /// 创建 TSV 写入器
    pub fn new(path: &str) -> Result<Self, std::io::Error> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "entity\tpredicate\ttarget")?;
        Ok(TsvWriter { writer })
    }

    /// 写入一行
    pub fn write_line(&mut self, entity: &str, predicate: &str, target: &str) -> Result<(), std::io::Error> {
        writeln!(self.writer, "{}\t{}\t{}", entity, predicate, target)
    }

    /// 刷新缓冲区
    pub fn flush(&mut self) -> Result<(), std::io::Error> {
        self.writer.flush()
    }
}

/// 将三元组列表写入 TSV 文件
pub fn write_triples(path: &str, triples: &[(String, String, String)]) -> Result<(), std::io::Error> {
    let mut writer = TsvWriter::new(path)?;
    for (entity, predicate, target) in triples {
        writer.write_line(entity, predicate, target)?;
    }
    writer.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn test_write_triples() {
        let path = "test_output.tsv";
        let triples = vec![
            ("苹果".to_string(), "contains".to_string(), "苹".to_string()),
            ("苹果".to_string(), "contains".to_string(), "果".to_string()),
        ];
        write_triples(path, &triples).unwrap();

        let mut content = String::new();
        File::open(path).unwrap().read_to_string(&mut content).unwrap();
        assert!(content.contains("entity\tpredicate\ttarget"));
        assert!(content.contains("苹果\tcontains\t苹"));

        std::fs::remove_file(path).unwrap();
    }
}