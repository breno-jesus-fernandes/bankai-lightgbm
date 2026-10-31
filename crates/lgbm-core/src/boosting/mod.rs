//! Boosting models. upstream: src/boosting/

mod gbdt;
mod model_text;
mod sample;
mod train;

pub use gbdt::Gbdt;
pub use train::GbdtTrainer;
