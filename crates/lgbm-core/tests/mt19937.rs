//! Conformance oracle for the `std::mt19937` engine that upstream's quantized
//! stochastic rounding seeds (src/treelearner/gradient_discretizer.cpp, v4.7.0).
//!
//! The expected value is the C++ standard's own predefined test: the 10000th
//! consecutive invocation of a default-constructed `std::mt19937` is
//! `4123659995` (ISO/IEC 14882 [rand.predef]). This single vector uniquely pins
//! the seeding, the state recurrence (twist) and the tempering, so it validates
//! the engine for every seed upstream derives from `random_seed`.
#![allow(clippy::unwrap_used)]

use lgbm_core::treelearner::gradient_discretizer::mt19937::Mt19937;

#[test]
fn mt19937_matches_cxx_standard_predefined_vector() {
    let mut engine = Mt19937::default();
    let mut value = 0u32;
    for _ in 0..10_000 {
        value = engine.next_u32();
    }
    assert_eq!(value, 4_123_659_995);
}

#[test]
fn mt19937_default_equals_explicit_default_seed() {
    let mut default_engine = Mt19937::default();
    let mut seeded_engine = Mt19937::new(5489);
    for _ in 0..1000 {
        assert_eq!(default_engine.next_u32(), seeded_engine.next_u32());
    }
}
