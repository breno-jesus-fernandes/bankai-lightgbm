//! Number formatting identical to upstream's writers.
//!
//! upstream uses fmt (`{:g}`, `{:.17g}`) and `std::ostream <<` (default precision 6, i.e.
//! `%g`). Both follow C `printf` `%g` rules, which Rust's formatter has no equivalent for.

/// C `printf("%.{precision}g", v)`.
pub fn format_g(v: f64, precision: usize) -> String {
    let mut out = String::with_capacity(24);
    write_g(&mut out, v, precision);
    out
}

/// Append C `printf("%.{precision}g", v)` to `out` (single formatting pass).
pub fn write_g(out: &mut String, v: f64, precision: usize) {
    use std::fmt::Write;
    if v.is_nan() {
        out.push_str(if v.is_sign_negative() { "-nan" } else { "nan" });
        return;
    }
    if v.is_infinite() {
        out.push_str(if v < 0.0 { "-inf" } else { "inf" });
        return;
    }
    if v == 0.0 {
        out.push_str(if v.is_sign_negative() { "-0" } else { "0" });
        return;
    }
    let p = precision.max(1);
    // Correctly rounded p significant digits: "-d.ddddde-X". Both %g styles use exactly
    // these digits, so the fixed form is built from them without reformatting.
    let mut sci = [0u8; 40];
    let len = {
        let mut cur = std::io::Cursor::new(&mut sci[..]);
        let _ = std::io::Write::write_fmt(&mut cur, format_args!("{:.*e}", p - 1, v));
        cur.position() as usize
    };
    let sci = &sci[..len];
    let neg = sci[0] == b'-';
    let body = if neg { &sci[1..] } else { sci };
    let epos = body.iter().position(|&c| c == b'e').unwrap_or(body.len());
    let mut x: i32 = 0;
    let mut eneg = false;
    for &c in &body[(epos + 1).min(body.len())..] {
        if c == b'-' {
            eneg = true;
        } else {
            x = x * 10 + i32::from(c - b'0');
        }
    }
    if eneg {
        x = -x;
    }
    let mut digits = [0u8; 40];
    let mut nd = 0;
    for &c in &body[..epos] {
        if c != b'.' {
            digits[nd] = c;
            nd += 1;
        }
    }
    // %g drops trailing zeros.
    while nd > 1 && digits[nd - 1] == b'0' {
        nd -= 1;
    }
    let digits = &digits[..nd];
    if neg {
        out.push('-');
    }
    let push = |out: &mut String, d: &[u8]| {
        for &c in d {
            out.push(c as char);
        }
    };
    if x < -4 || x >= p as i32 {
        push(out, &digits[..1]);
        if nd > 1 {
            out.push('.');
            push(out, &digits[1..]);
        }
        let _ = write!(
            out,
            "e{}{:02}",
            if x < 0 { '-' } else { '+' },
            x.unsigned_abs()
        );
    } else if x < 0 {
        out.push_str("0.");
        for _ in 0..(-x - 1) {
            out.push('0');
        }
        push(out, digits);
    } else {
        let int_len = (x + 1) as usize;
        if nd <= int_len {
            push(out, digits);
            for _ in nd..int_len {
                out.push('0');
            }
        } else {
            push(out, &digits[..int_len]);
            out.push('.');
            push(out, &digits[int_len..]);
        }
    }
}

/// `%g` for a `float` value (formatted from its exact binary value, as fmt/printf do).
pub fn format_g_f32(v: f32, precision: usize) -> String {
    format_g(f64::from(v), precision)
}

/// Upstream `ArrayToString<false>` for doubles: `{:g}` joined by spaces.
pub fn join_g(vals: &[f64]) -> String {
    join_with(vals, 6)
}

/// Upstream `ArrayToString<true>` for doubles: `{:.17g}` joined by spaces.
pub fn join_g17(vals: &[f64]) -> String {
    join_with(vals, 17)
}

fn join_with(vals: &[f64], precision: usize) -> String {
    let mut out = String::with_capacity(vals.len() * (precision + 6));
    for (i, v) in vals.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        write_g(&mut out, *v, precision);
    }
    out
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
    fn fixed_and_scientific_boundaries() {
        assert_eq!(format_g(1e-4, 6), "0.0001");
        assert_eq!(format_g(1.5e-4, 17), "0.00014999999999999999");
        assert_eq!(format_g(123456.7, 6), "123457");
        assert_eq!(format_g(-12.5, 6), "-12.5");
        assert_eq!(format_g(1e16, 17), "10000000000000000");
        assert_eq!(
            format_g(1e17, 17),
            "1.0000000000000000e+17"
                .replace(".0000000000000000", "")
                .as_str()
        );
        assert_eq!(format_g(3.0e-310, 6), "3e-310");
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
