//! Build a dataset from `x.npy`/`y.npy` + params (key=value args) and save it as `.bin`.
//! Usage: write_bin <case_dir> <out.bin> [categorical,indices] [key=value ...]
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
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::Path::new(&args[1]);
    let cats: Vec<i32> = args
        .get(3)
        .map(|s| s.split(',').filter_map(|c| c.parse().ok()).collect())
        .unwrap_or_default();
    let params = args.iter().skip(4).cloned().collect::<Vec<_>>().join(" ");
    let cfg = Config::from_str_params(&params)?;
    let (x, shape) = load_npy(&dir.join("x.npy"))?;
    let (y, _) = load_npy(&dir.join("y.npy"))?;
    let ds = Dataset::from_dense(&x, shape[0], shape[1], &y, &cfg, &cats)?;
    ds.save_binary(&args[2])?;
    Ok(())
}
