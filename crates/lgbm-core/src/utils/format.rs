//! Number formatting identical to upstream's writers.
//!
//! upstream uses fmt (`{:g}`, `{:.17g}`) and `std::ostream <<` (default precision 6, i.e.
//! `%g`). Both follow C `printf` `%g` rules, which Rust's formatter has no equivalent for.

/// C `printf("%.{precision}g", v)`.
pub fn format_g(v: f64, precision: usize) -> String {
    if v.is_nan() {
        return if v.is_sign_negative() {
            "-nan".into()
        } else {
            "nan".into()
        };
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf".into() } else { "inf".into() };
    }
    let p = precision.max(1);
    if v == 0.0 {
        return if v.is_sign_negative() {
            "-0".into()
        } else {
            "0".into()
        };
    }
    // Correctly rounded scientific form with p significant digits gives the exponent X.
    let sci = format!("{:.*e}", p - 1, v);
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let x: i32 = exp.parse().unwrap_or(0);
    if x < -4 || x >= p as i32 {
        let m = strip_trailing_zeros(mantissa);
        let sign = if x < 0 { '-' } else { '+' };
        format!("{m}e{sign}{:02}", x.unsigned_abs())
    } else {
        let decimals = (p as i32 - 1 - x).max(0) as usize;
        strip_trailing_zeros(&format!("{v:.decimals$}")).to_string()
    }
}

/// `%g` for a `float` value (formatted from its exact binary value, as fmt/printf do).
pub fn format_g_f32(v: f32, precision: usize) -> String {
    format_g(f64::from(v), precision)
}

fn strip_trailing_zeros(s: &str) -> &str {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.')
    } else {
        s
    }
}

/// Upstream `ArrayToString<false>` for doubles: `{:g}` joined by spaces.
pub fn join_g(vals: &[f64]) -> String {
    join(vals.iter().map(|v| format_g(*v, 6)))
}

/// Upstream `ArrayToString<true>` for doubles: `{:.17g}` joined by spaces.
pub fn join_g17(vals: &[f64]) -> String {
    join(vals.iter().map(|v| format_g(*v, 17)))
}

pub fn join_display<T: std::fmt::Display>(vals: &[T]) -> String {
    join(vals.iter().map(ToString::to_string))
}

fn join(it: impl Iterator<Item = String>) -> String {
    let mut out = String::new();
    for (i, s) in it.enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(&s);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_printf_g() {
        // Expected strings from C printf("%g") / Python "%g".
        let cases: &[(f64, &str)] = &[
            (0.168055, "0.168055"),
            (1600.0, "1600"),
            (4307.79123, "4307.79"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (123456.0, "123456"),
            (1234567.0, "1.23457e+06"),
            (-0.0059702, "-0.0059702"),
            (1e100, "1e+100"),
            (0.5, "0.5"),
            (100000.0, "100000"),
            (999999.5, "1e+06"),
            (1.0, "1"),
        ];
        for (v, want) in cases {
            assert_eq!(format_g(*v, 6), *want, "{v}");
        }
    }

    #[test]
    fn matches_printf_g17() {
        let cases: &[(f64, &str)] = &[
            (-0.10661721840970087, "-0.10661721840970087"),
            (0.1, "0.10000000000000001"),
            (1.0000000180025095e-35, "1.0000000180025095e-35"),
            (2.0, "2"),
            (0.25, "0.25"),
        ];
        for (v, want) in cases {
            assert_eq!(format_g(*v, 17), *want, "{v}");
        }
    }

    #[test]
    fn special_values() {
        assert_eq!(format_g(f64::INFINITY, 6), "inf");
        assert_eq!(format_g(f64::NEG_INFINITY, 17), "-inf");
        assert_eq!(format_g(f64::NAN, 6), "nan");
        assert_eq!(format_g(0.0, 6), "0");
        assert_eq!(format_g(-0.0, 6), "-0");
    }

    #[test]
    fn joins_with_single_spaces() {
        assert_eq!(join_display(&[1, -2, 3]), "1 -2 3");
        assert_eq!(join_g(&[]), "");
    }
}
