//! Parsing and bit helpers. upstream: include/LightGBM/utils/common.h

use crate::error::{Error, Result};

/// Number parsing that accepts the spellings upstream's C++ writer can emit
/// (`inf`, `-inf`, `nan`, `-nan`).
pub trait ParseNum: Sized {
    fn parse_num(s: &str) -> Option<Self>;
}

macro_rules! impl_parse_int {
    ($($t:ty),*) => {$(
        impl ParseNum for $t {
            fn parse_num(s: &str) -> Option<Self> { s.parse().ok() }
        }
    )*};
}
impl_parse_int!(i8, i32, i64, u32, u64, usize);

macro_rules! impl_parse_float {
    ($($t:ty),*) => {$(
        impl ParseNum for $t {
            fn parse_num(s: &str) -> Option<Self> {
                s.parse().ok().or_else(|| match s.to_ascii_lowercase().as_str() {
                    "-nan" | "nan(ind)" | "-nan(ind)" => Some(<$t>::NAN),
                    "1.#inf" => Some(<$t>::INFINITY),
                    "-1.#inf" => Some(<$t>::NEG_INFINITY),
                    _ => None,
                })
            }
        }
    )*};
}
impl_parse_float!(f32, f64);

pub fn parse_scalar<T: ParseNum>(s: &str) -> Result<T> {
    let s = s.trim();
    T::parse_num(s).ok_or_else(|| Error::Parse(format!("cannot parse number from {s:?}")))
}

/// Parse exactly `n` whitespace-separated values (upstream `StringToArray(str, n)`).
pub fn parse_array<T: ParseNum>(s: &str, n: usize) -> Result<Vec<T>> {
    let out = parse_array_any(s)?;
    if out.len() != n {
        return Err(Error::Parse(format!(
            "array size mismatch: expected {n}, got {}",
            out.len()
        )));
    }
    Ok(out)
}

/// Parse all whitespace-separated values.
pub fn parse_array_any<T: ParseNum>(s: &str) -> Result<Vec<T>> {
    s.split_ascii_whitespace().map(parse_scalar).collect()
}

/// `a * b + c` with the same contraction the upstream C++ build applies.
///
/// Upstream wheels are compiled by clang/gcc with floating-point contraction enabled; on
/// targets with a hardware FMA in the baseline ISA (aarch64) expressions of the form
/// `x += a * b` become a fused multiply-add, on baseline x86-64 they do not. Using the
/// same rounding keeps predictions bit-identical to upstream on each platform.
#[inline(always)]
pub fn madd(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a * b + c
    }
}

/// upstream: `Common::FindInBitset`
#[inline]
pub fn find_in_bitset(bits: &[u32], pos: u32) -> bool {
    let i1 = (pos / 32) as usize;
    if i1 >= bits.len() {
        return false;
    }
    (bits[i1] >> (pos % 32)) & 1 == 1
}

/// upstream: `Common::Softmax(const double*, double*, int)`
#[inline]
pub fn softmax(io: &mut [f64]) {
    let mut wmax = io[0];
    for &v in &io[1..] {
        wmax = v.max(wmax);
    }
    let mut wsum = 0.0;
    for v in io.iter_mut() {
        *v = (*v - wmax).exp();
        wsum += *v;
    }
    for v in io.iter_mut() {
        *v /= wsum;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn parses_exact_count_and_special_floats() {
        let v: Vec<f64> = parse_array("1 -inf  nan -nan 2.5e-3", 5).unwrap();
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], f64::NEG_INFINITY);
        assert!(v[2].is_nan() && v[3].is_nan());
        assert_eq!(v[4], 2.5e-3);
        assert!(parse_array::<f64>("1 2", 3).is_err());
        assert!(parse_array::<i32>("1 x", 2).is_err());
    }

    #[test]
    fn float_parsing_is_correctly_rounded() {
        let s = "-0.10661721840970087";
        assert_eq!(
            parse_scalar::<f64>(s).unwrap().to_bits(),
            (-0.10661721840970087_f64).to_bits()
        );
    }

    #[test]
    fn bitset_lookup() {
        assert!(find_in_bitset(&[0b10, 0b10], 1));
        assert!(find_in_bitset(&[0b10, 0b10], 33));
        assert!(!find_in_bitset(&[0b10, 0b10], 0));
        assert!(!find_in_bitset(&[0b10, 0b10], 64));
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut v = [1.0, 2.0, 3.0];
        softmax(&mut v);
        assert!((v.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        assert!(v[2] > v[1] && v[1] > v[0]);
    }
}
