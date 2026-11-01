//! Time dataset construction + training on `<dir>/{x,y}.npy` with `<dir>/params.json`.
use std::collections::HashMap;
use std::time::Instant;

use lgbm_core::boosting::Gbdt;
use lgbm_core::config::Config;
use lgbm_core::io::dataset::Dataset;

fn load_npy(path: &std::path::Path) -> Result<(Vec<f64>, Vec<usize>), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + hlen])?;
    let shape: Vec<usize> = header
        .split("'shape': (")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .unwrap_or("")
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let vals = bytes[10 + hlen..]
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap_or([0; 8])))
        .collect();
    Ok((vals, shape))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).ok_or("usage: train_bench <dir>")?);
    let v: HashMap<String, serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("params.json"))?)?;
    let mut params: HashMap<String, String> = HashMap::new();
    if let Some(serde_json::Value::Object(p)) = v.get("params") {
        for (k, val) in p {
            let s = match val {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            params.insert(k.clone(), s);
        }
    }
    params.insert("num_iterations".into(), v["num_boost_round"].to_string());
    let cfg = Config::from_params(&params)?;
    let (x, shape) = load_npy(&dir.join("x.npy"))?;
    let (y, _) = load_npy(&dir.join("y.npy"))?;
    let t0 = Instant::now();
    let ds = Dataset::from_dense(&x, shape[0], shape[1], &y, &cfg, &[])?;
    let t1 = Instant::now();
    let model = Gbdt::train(&cfg, &ds)?;
    let t2 = Instant::now();
    println!(
        "dataset {:.3}s  train {:.3}s  trees {}",
        (t1 - t0).as_secs_f64(),
        (t2 - t1).as_secs_f64(),
        model.num_trees()
    );
    std::fs::write(
        dir.join("ours_model.txt"),
        model.save_model_to_string(0, -1, 0)?,
    )?;
    Ok(())
}
