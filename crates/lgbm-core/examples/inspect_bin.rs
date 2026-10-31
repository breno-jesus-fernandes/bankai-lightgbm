//! Print the structure of an upstream binary dataset file.
use lgbm_core::io::binary::BinaryDataset;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for path in std::env::args().skip(1) {
        let ds = BinaryDataset::read_file(&path)?;
        let h = &ds.header;
        let multi = ds.groups.iter().filter(|g| g.is_multi_val).count();
        let dense_multi = ds.groups.iter().filter(|g| g.is_dense_multi_val).count();
        let sparse = ds.groups.iter().filter(|g| g.is_sparse).count();
        let max_feats = ds
            .groups
            .iter()
            .map(|g| g.bin_mappers.len())
            .max()
            .unwrap_or(0);
        println!(
            "{path}: features {}/{} groups {} (multi_val {multi}, dense_multi {dense_multi}, sparse {sparse}, max feats/group {max_feats})",
            h.num_features, h.num_total_features, h.num_groups
        );
    }
    Ok(())
}
