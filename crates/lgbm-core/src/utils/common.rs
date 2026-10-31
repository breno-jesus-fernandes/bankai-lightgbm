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

/// upstream: `Common::Pow` (exponentiation by squaring, *not* `powi`).
pub fn pow_legacy(base: f64, power: i32) -> f64 {
    if power < 0 {
        1.0 / pow_legacy(base, -power)
    } else if power == 0 {
        1.0
    } else if power % 2 == 0 {
        pow_legacy(base * base, power / 2)
    } else if power % 3 == 0 {
        pow_legacy(base * base * base, power / 3)
    } else {
        base * pow_legacy(base, power - 1)
    }
}

/// upstream: `Common::Atof`, the legacy parser used by `StringToArrayFast<double>` and
/// `CommonC::Atof`. It is **not** correctly rounded; we reproduce it bit-for-bit.
pub fn atof_legacy(s: &str) -> Option<f64> {
    let b = s.trim_matches(' ').as_bytes();
    let mut p = 0;
    let mut sign = 1.0;
    if p < b.len() && b[p] == b'-' {
        sign = -1.0;
        p += 1;
    } else if p < b.len() && b[p] == b'+' {
        p += 1;
    }
    let is_num = p < b.len() && (b[p].is_ascii_digit() || matches!(b[p], b'.' | b'e' | b'E'));
    if !is_num {
        return match String::from_utf8_lossy(&b[p..]).to_ascii_lowercase().as_str() {
            "na" | "nan" | "null" => Some(f64::NAN),
            "inf" | "infinity" => Some(sign * 1e308),
            _ => None,
        };
    }
    let mut value = 0.0;
    while p < b.len() && b[p].is_ascii_digit() {
        value = value * 10.0 + f64::from(b[p] - b'0');
        p += 1;
    }
    if p < b.len() && b[p] == b'.' {
        let mut right = 0.0;
        let mut nn = 0;
        p += 1;
        while p < b.len() && b[p].is_ascii_digit() {
            right = f64::from(b[p] - b'0') + right * 10.0;
            nn += 1;
            p += 1;
        }
        value += right / pow_legacy(10.0, nn);
    }
    let mut frac = false;
    let mut scale = 1.0;
    if p < b.len() && (b[p] == b'e' || b[p] == b'E') {
        p += 1;
        if p < b.len() && b[p] == b'-' {
            frac = true;
            p += 1;
        } else if p < b.len() && b[p] == b'+' {
            p += 1;
        }
        let mut expon: u32 = 0;
        while p < b.len() && b[p].is_ascii_digit() {
            expon = expon.wrapping_mul(10).wrapping_add(u32::from(b[p] - b'0'));
            p += 1;
        }
        expon = expon.min(308);
        while expon >= 50 {
            scale *= 1e50;
            expon -= 50;
        }
        while expon >= 8 {
            scale *= 1e8;
            expon -= 8;
        }
        while expon > 0 {
            scale *= 10.0;
            expon -= 1;
        }
    }
    if p != b.len() {
        return None;
    }
    Some(sign * if frac { value / scale } else { value * scale })
}

/// upstream `StringToArrayFast<T>` for floating types: legacy [`atof_legacy`] per value.
pub fn parse_array_fast_f64(s: &str, n: usize) -> Result<Vec<f64>> {
    let out: Vec<f64> = s
        .split_ascii_whitespace()
        .map(|t| atof_legacy(t).ok_or_else(|| Error::Parse(format!("cannot parse number from {t:?}"))))
        .collect::<Result<_>>()?;
    if out.len() != n {
        return Err(Error::Parse(format!("array size mismatch: expected {n}, got {}", out.len())));
    }
    Ok(out)
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
    fn legacy_atof_is_not_correctly_rounded_but_deterministic() {
        // A value where the legacy parser differs from correct rounding.
        let s = "0.10000000000000001";
        let legacy = atof_legacy(s).unwrap();
        let correct: f64 = s.parse().unwrap();
        assert!((legacy - correct).abs() <= 2.0 * f64::EPSILON * correct);
        assert_eq!(atof_legacy("1e-05").unwrap(), 1.0 / 100000.0);
        assert_eq!(atof_legacy("-2.5").unwrap(), -2.5);
        assert!(atof_legacy("nan").unwrap().is_nan());
        assert_eq!(atof_legacy("inf").unwrap(), 1e308);
        assert!(atof_legacy("x1").is_none());
    }

    #[test]
    fn legacy_pow_matches_upstream_recursion() {
        assert_eq!(pow_legacy(10.0, 3), 1000.0);
        assert_eq!(pow_legacy(10.0, -2), 1.0 / 100.0);
        assert_eq!(pow_legacy(10.0, 0), 1.0);
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
