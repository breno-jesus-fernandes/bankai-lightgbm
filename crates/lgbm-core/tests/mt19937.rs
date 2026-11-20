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

// The quantized stochastic-rounding path draws `std::uniform_real_distribution<double>(0,1)`
// and `std::uniform_int_distribution<int>(0,n)` over the engine. Those distributions are
// implementation-defined; bankai pins libc++ (the host/fixture stdlib, D-031). The expected
// values below are produced by clang/libc++ `<random>` on this platform.

#[test]
fn uniform_real_01_matches_libcxx() {
    // std::mt19937(42), std::uniform_real_distribution<double>(0,1), first five draws, as bits.
    let expected: [u64; 5] = [
        0x3fe97d47b66bfc3c,
        0x3fc77aca8779b102,
        0x3fe8f33a88f76c7f,
        0x3fe3196582932836,
        0x3fdc88861e49fc36,
    ];
    let mut engine = Mt19937::new(42);
    for (i, &bits) in expected.iter().enumerate() {
        let v = engine.uniform_real_01();
        assert_eq!(
            v.to_bits(),
            bits,
            "draw {i}: {v} != {}",
            f64::from_bits(bits)
        );
    }
}

#[test]
fn uniform_int_inclusive_matches_libcxx() {
    // std::mt19937(7), std::uniform_int_distribution<int>(0,1000), first eight draws.
    let mut e = Mt19937::new(7);
    for want in [175, 196, 537, 502, 579, 211, 919, 615] {
        assert_eq!(e.uniform_int_inclusive(1000), want);
    }
    // A power-of-two range (0,255) and a non-power-of-two (0,300), each from mt19937(123).
    let mut e255 = Mt19937::new(123);
    for want in [254, 109, 126, 66] {
        assert_eq!(e255.uniform_int_inclusive(255), want);
    }
    let mut e300 = Mt19937::new(123);
    for want in [98, 230, 17, 83] {
        assert_eq!(e300.uniform_int_inclusive(300), want);
    }
}
