use std::path::PathBuf;

use n_iter_rs::experiments;

fn main() {
    let output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().join("experiments").join("011-fca-n-iteration").join("output");
    let _ = std::fs::create_dir_all(&output_dir);

    println!("{}", "=".repeat(64));
    println!("  v3.57 ONLY");
    println!("{}", "=".repeat(64));

    experiments::run_d0_corrected_prediction();
}
