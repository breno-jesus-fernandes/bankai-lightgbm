//! Shared constants and small enums. upstream: include/LightGBM/meta.h

/// `kZeroThreshold = 1e-35f` upstream: a *float* literal widened to double.
pub const K_ZERO_THRESHOLD: f64 = 1e-35_f32 as f64;

/// `kEpsilon = 1e-15f` (score_t).
pub const K_EPSILON: f32 = 1e-15;

/// How missing values are routed at a numerical split (bits 2-3 of `decision_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingType {
    None,
    Zero,
    NaN,
}

impl MissingType {
    pub fn from_bits(bits: i8) -> Self {
        match bits {
            1 => MissingType::Zero,
            2 => MissingType::NaN,
            _ => MissingType::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_threshold_matches_float_literal_semantics() {
        assert_ne!(K_ZERO_THRESHOLD, 1e-35_f64);
        assert_eq!(K_ZERO_THRESHOLD as f32, 1e-35_f32);
    }
}
