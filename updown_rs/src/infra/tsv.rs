//! WikiLine TSV 输出（适配完整流水线）

use crate::emergence::cycle::CycleResult;
use crate::emergence::extractor::RelationNetwork;
use crate::emergence::relations::{DerivedRule, DerivedConstraint, SelfOrganizedRelations};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const TSV_HEADER: &str = "record_id\tentity\tpredicate\ttarget\tsource\tstate\tconflicts\tparent\ttimestamp\n";

pub struct TsvWriter {
    output_dir: String,
    record_counter: u64,
    seen: HashSet<(String, String, String)>,
}

impl TsvWriter {
    pub fn new(output_dir: &str) -> Self {
        Self { output_dir: output_dir.to_string(), record_counter: 0, seen: HashSet::new() }
    }

    /// 写入 containment 和 connection 边（Phase 1）
    pub fn write_relations(
        &mut self,
        psi: &RelationNetwork,
        relations: &SelfOrganizedRelations,
        cycle: &CycleResult,
        source: &str,
    ) -> String {
        self.seen.clear();
        let mut records: Vec<Vec<String>> = Vec::new();
        let wc = psi.word_count;
        let ts = now_ts();

        // containment: 字 → 词
        for ci in 0..psi.char_to_words.len() {
            let ent = &psi.node_texts[ci + wc];
            for &wi in &psi.char_to_words[ci] {
                self.push_rec(&mut records, ent, "containment", &psi.node_texts[wi], source, "-", &ts);
            }
        }

        // containment: 词 → 词（子串）
        for wi in 0..wc {
            for &sj in &psi.word_to_super_words[wi] {
                self.push_rec(&mut records, &psi.node_texts[wi], "containment", &psi.node_texts[sj], source, "-", &ts);
            }
        }

        // connection: 字—字 Jaccard ≥ T
        for (ci, cj, _j, shared) in &relations.char_relations {
            let ent = &psi.node_texts[*ci + wc];
            let tgt = &psi.node_texts[*cj + wc];
            let par = if !shared.is_empty() { &psi.node_texts[shared[0]] } else { "-" };
            self.push_rec(&mut records, ent, "connection", tgt, source, par, &ts);
            self.push_rec(&mut records, tgt, "connection", ent, source, par, &ts);
        }

        // connection: 词—词 Jaccard ≥ T
        for (wi, wj, _j, shared) in &relations.word_relations {
            let ent = &psi.node_texts[*wi];
            let tgt = &psi.node_texts[*wj];
            let par = if !shared.is_empty() { &psi.node_texts[shared[0] + wc] } else { "-" };
            self.push_rec(&mut records, ent, "connection", tgt, source, par, &ts);
            self.push_rec(&mut records, tgt, "connection", ent, source, par, &ts);
        }

        // 涌现统计（作为 meta 记录）
        let meta_ent = "pipeline";
        self.push_rec(&mut records, meta_ent, "meta", &format!("containment_count={}", cycle.emergence.containment_count), source, "-", &ts);
        self.push_rec(&mut records, meta_ent, "meta", &format!("char_cocc_pairs={}", cycle.emergence.char_coccurrence_pairs), source, "-", &ts);
        self.push_rec(&mut records, meta_ent, "meta", &format!("word_assoc_pairs={}", cycle.emergence.word_association_pairs), source, "-", &ts);
        self.push_rec(&mut records, meta_ent, "meta", &format!("total_rounds={}", cycle.termination.total_rounds), source, "-", &ts);

        self.write_file(source, "relations", &records)
    }

    /// 写入规则和约束（Phase 1 输出）
    pub fn write_rules(
        &mut self,
        rules: &[DerivedRule],
        constraints: &[DerivedConstraint],
        source: &str,
    ) -> String {
        self.seen.clear();
        let mut records: Vec<Vec<String>> = Vec::new();
        let ts = now_ts();

        for rule in rules {
            self.push_rec(&mut records, "rule", rule.rule_type.label(), &format!("conf={:.4}", rule.confidence), source, "-", &ts);
        }
        for c in constraints {
            self.push_rec(&mut records, "constraint", &c.invariant_equation, &format!("val={:.4}", c.value), source, "-", &ts);
        }

        self.write_file(source, "rules", &records)
    }

    fn push_rec(
        &mut self,
        records: &mut Vec<Vec<String>>,
        entity: &str,
        predicate: &str,
        target: &str,
        source: &str,
        parent: &str,
        ts: &str,
    ) {
        let key = (entity.to_string(), predicate.to_string(), target.to_string());
        if self.seen.contains(&key) { return; }
        self.seen.insert(key);
        self.record_counter += 1;
        let rid = format!("up_{:08x}", self.record_counter);
        records.push(vec![rid, entity.to_string(), predicate.to_string(), target.to_string(), source.to_string(), "unheld".to_string(), "-".to_string(), parent.to_string(), ts.to_string()]);
    }

    fn write_file(&self, source: &str, suffix: &str, records: &[Vec<String>]) -> String {
        std::fs::create_dir_all(&self.output_dir).ok();
        let fp = Path::new(&self.output_dir).join(format!("{}_{}.tsv", source, suffix));
        let fp_s = fp.to_string_lossy().to_string();
        let existed = fp.exists();
        let file = File::options().append(true).create(true).open(&fp).expect("无法打开 TSV");
        let mut w = BufWriter::new(file);
        if !existed { w.write_all(TSV_HEADER.as_bytes()).ok(); }
        for rec in records {
            w.write_all((rec.join("\t") + "\n").as_bytes()).ok();
        }
        w.flush().ok();
        println!("  │ TSV: {} → {} ({} 条)", suffix, fp.display(), records.len());
        fp_s
    }
}

fn now_ts() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| format!("{:?}", d))
        .unwrap_or_default()
}

// RuleType label helper
impl crate::emergence::relations::RuleType {
    pub fn label(&self) -> &str {
        match self {
            crate::emergence::relations::RuleType::CooccurrenceRule => "cooccurrence_rule",
            crate::emergence::relations::RuleType::StructuralRule => "structural_rule",
        }
    }
}
