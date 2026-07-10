use std::fs::File;
use std::io::{BufWriter, Write};

pub struct TsvWriter {
    writer: BufWriter<File>,
}

impl TsvWriter {
    pub fn new(path: &str) -> Result<Self, std::io::Error> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "entity\tpredicate\ttarget")?;
        Ok(TsvWriter { writer })
    }

    pub fn write_line(
        &mut self,
        entity: &str,
        predicate: &str,
        target: &str,
    ) -> Result<(), std::io::Error> {
        writeln!(self.writer, "{}\t{}\t{}", entity, predicate, target)
    }

    pub fn flush(&mut self) -> Result<(), std::io::Error> {
        self.writer.flush()
    }
}

pub fn write_triples(
    path: &str,
    triples: &[(String, String, String)],
) -> Result<(), std::io::Error> {
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
            ("Apple".to_string(), "contains".to_string(), "A".to_string()),
            ("Apple".to_string(), "contains".to_string(), "p".to_string()),
        ];
        write_triples(path, &triples).unwrap();

        let mut content = String::new();
        File::open(path).unwrap().read_to_string(&mut content).unwrap();
        assert!(content.contains("entity\tpredicate\ttarget"));
        assert!(content.contains("Apple\tcontains\tA"));

        std::fs::remove_file(path).unwrap();
    }
}
