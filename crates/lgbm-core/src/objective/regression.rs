//! Regression objectives. upstream: src/objective/regression_objective.hpp

use super::train::{Objective, sign};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::io::dataset::Metadata;
use crate::utils::common::madd;

/// Which upstream class (`RegressionL2loss` and its subclasses).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    L2,
    L1,
    Quantile,
    Huber,
    Fair,
    Poisson,
    Mape,
    Gamma,
    Tweedie,
}

/// upstream: `RegressionL2loss` and subclasses (unweighted, except plain L2).
pub struct Regression {
    kind: Kind,
    sqrt: bool,
    label: Vec<f32>,
    weights: Vec<f32>,
    /// huber `alpha` (double)
    alpha: f64,
    /// quantile `alpha_` (score_t)
    alpha_f32: f32,
    fair_c: f64,
    max_delta_step: f64,
    rho: f64,
    /// MAPE `label_weight_`
    label_weight: Vec<f32>,
}

impl Regression {
    pub fn new(name: &str, cfg: &Config, md: &Metadata) -> Result<Self> {
        let kind = match name {
            "regression" => Kind::L2,
            "regression_l1" => Kind::L1,
            "quantile" => Kind::Quantile,
            "huber" => Kind::Huber,
            "fair" => Kind::Fair,
            "poisson" => Kind::Poisson,
            "mape" => Kind::Mape,
            "gamma" => Kind::Gamma,
            "tweedie" => Kind::Tweedie,
            other => return Err(Error::Unsupported(format!("objective={other}"))),
        };
        // Huber and the Poisson family disable the sqrt transform.
        let sqrt = cfg.reg_sqrt
            && !matches!(
                kind,
                Kind::Huber | Kind::Poisson | Kind::Gamma | Kind::Tweedie
            );
        if cfg.reg_sqrt && !sqrt {
            log::warn!("Cannot use sqrt transform in {name} Regression, will auto disable it");
        }
        let alpha_f32 = cfg.alpha as f32;
        if kind == Kind::Quantile && !(alpha_f32 > 0.0 && alpha_f32 < 1.0) {
            return Err(Error::InvalidParameter(
                "quantile alpha must be in (0, 1)".into(),
            ));
        }
        // upstream: RegressionL2loss::Init
        let label: Vec<f32> = if sqrt {
            md.label
                .iter()
                .map(|&l| sign(f64::from(l)) as f32 * l.abs().sqrt())
                .collect()
        } else {
            md.label.clone()
        };
        if matches!(kind, Kind::Poisson | Kind::Gamma | Kind::Tweedie) {
            // upstream: RegressionPoissonLoss::Init label checks
            let miny = label.iter().copied().fold(f32::INFINITY, f32::min);
            let sumy: f64 = label.iter().map(|&l| f64::from(l)).sum();
            if miny < 0.0 {
                return Err(Error::InvalidParameter(format!(
                    "[{name}]: at least one target label is negative"
                )));
            }
            if sumy == 0.0 {
                return Err(Error::InvalidParameter(format!(
                    "[{name}]: sum of labels is zero"
                )));
            }
        }
        let mut label_weight = Vec::new();
        if kind == Kind::Mape {
            if label.iter().any(|l| l.abs() < 1.0) {
                log::warn!(
                    "Some label values are < 1 in absolute value. MAPE is unstable with such \
                     values, so LightGBM rounds them to 1.0 when calculating MAPE."
                );
            }
            label_weight = if md.weights.is_empty() {
                label.iter().map(|l| 1.0f32 / l.abs().max(1.0)).collect()
            } else {
                label
                    .iter()
                    .zip(&md.weights)
                    .map(|(l, &w)| 1.0f32 / l.abs().max(1.0) * w)
                    .collect()
            };
        }
        Ok(Regression {
            kind,
            sqrt,
            label,
            weights: md.weights.clone(),
            alpha: cfg.alpha,
            alpha_f32,
            fair_c: cfg.fair_c,
            max_delta_step: cfg.poisson_max_delta_step,
            rho: cfg.tweedie_variance_power,
            label_weight,
        })
    }

    /// upstream: `RegressionL2loss::BoostFromScore`
    fn l2_boost_from_score(&self) -> f64 {
        if self.weights.is_empty() {
            let mut suml = 0.0f64;
            for &l in &self.label {
                suml += f64::from(l);
            }
            suml / self.label.len() as f64
        } else {
            let (mut suml, mut sumw) = (0.0f64, 0.0f64);
            for (&l, &wt) in self.label.iter().zip(&self.weights) {
                suml = madd(f64::from(l), f64::from(wt), suml);
                sumw += f64::from(wt);
            }
            suml / sumw
        }
    }
}

impl Objective for Regression {
    fn name(&self) -> &'static str {
        match self.kind {
            Kind::L2 => "regression",
            Kind::L1 => "regression_l1",
            Kind::Quantile => "quantile",
            Kind::Huber => "huber",
            Kind::Fair => "fair",
            Kind::Poisson => "poisson",
            Kind::Mape => "mape",
            Kind::Gamma => "gamma",
            Kind::Tweedie => "tweedie",
        }
    }

    fn to_model_string(&self) -> String {
        if self.sqrt {
            format!("{} sqrt", self.name())
        } else {
            self.name().to_string()
        }
    }

    fn get_gradients(&self, score: &[f64], g: &mut [f32], h: &mut [f32]) {
        if self.weights.is_empty() {
            self.gradients_unweighted(score, g, h);
        } else {
            self.gradients_weighted(score, g, h);
        }
    }

    fn boost_from_score(&self, _class_id: usize) -> f64 {
        let n = self.label.len();
        let label = |i: usize| self.label[i];
        let weight = |i: usize| self.weights[i];
        let weighted = !self.weights.is_empty();
        match self.kind {
            Kind::L2 | Kind::Huber | Kind::Fair => self.l2_boost_from_score(),
            Kind::Poisson | Kind::Gamma | Kind::Tweedie => safe_log(self.l2_boost_from_score()),
            Kind::L1 | Kind::Quantile => {
                let alpha = if self.kind == Kind::L1 {
                    0.5
                } else {
                    f64::from(self.alpha_f32)
                };
                f64::from(if weighted {
                    weighted_percentile(n, label, weight, alpha)
                } else {
                    percentile(n, label, alpha)
                })
            }
            Kind::Mape => f64::from(weighted_percentile(n, label, |i| self.label_weight[i], 0.5)),
        }
    }

    fn is_constant_hessian(&self) -> bool {
        match self.kind {
            Kind::Fair | Kind::Poisson | Kind::Gamma | Kind::Tweedie => false,
            // upstream returns true even with weights (hessians are then the weights).
            Kind::Mape => true,
            _ => self.weights.is_empty(),
        }
    }

    fn is_renew_tree_output(&self) -> bool {
        matches!(self.kind, Kind::L1 | Kind::Quantile | Kind::Mape)
    }

    fn renew_tree_output(&self, output: f64, score: &[f64], indices: &[i32]) -> f64 {
        let residual = |i: usize| {
            let r = indices[i] as usize;
            f64::from(self.label[r]) - score[r]
        };
        let n = indices.len();
        let alpha = match self.kind {
            Kind::L1 | Kind::Mape => 0.5,
            Kind::Quantile => f64::from(self.alpha_f32),
            _ => return output,
        };
        if self.kind == Kind::Mape {
            weighted_percentile(
                n,
                residual,
                |i| self.label_weight[indices[i] as usize],
                alpha,
            )
        } else if self.weights.is_empty() {
            percentile(n, residual, alpha)
        } else {
            weighted_percentile(n, residual, |i| self.weights[indices[i] as usize], alpha)
        }
    }
}

impl Regression {
    fn gradients_unweighted(&self, score: &[f64], g: &mut [f32], h: &mut [f32]) {
        let n = self.label.len();
        let label = &self.label;
        match self.kind {
            Kind::L2 => {
                for i in 0..n {
                    g[i] = (score[i] - f64::from(label[i])) as f32;
                    h[i] = 1.0;
                }
            }
            Kind::L1 => {
                for i in 0..n {
                    g[i] = sign(score[i] - f64::from(label[i])) as f32;
                    h[i] = 1.0;
                }
            }
            Kind::Quantile => {
                let a = self.alpha_f32;
                for i in 0..n {
                    let delta = (score[i] - f64::from(label[i])) as f32;
                    g[i] = if delta >= 0.0 { 1.0f32 - a } else { -a };
                    h[i] = 1.0;
                }
            }
            Kind::Huber => {
                for i in 0..n {
                    let diff = score[i] - f64::from(label[i]);
                    g[i] = if diff.abs() <= self.alpha {
                        diff as f32
                    } else {
                        (f64::from(sign(diff)) * self.alpha) as f32
                    };
                    h[i] = 1.0;
                }
            }
            Kind::Fair => {
                let c = self.fair_c;
                for i in 0..n {
                    let x = score[i] - f64::from(label[i]);
                    g[i] = (c * x / (x.abs() + c)) as f32;
                    h[i] = (c * c / ((x.abs() + c) * (x.abs() + c))) as f32;
                }
            }
            Kind::Poisson => {
                let emds = self.max_delta_step.exp();
                for i in 0..n {
                    let e = score[i].exp();
                    g[i] = (e - f64::from(label[i])) as f32;
                    h[i] = (e * emds) as f32;
                }
            }
            Kind::Mape => {
                for i in 0..n {
                    let diff = score[i] - f64::from(label[i]);
                    g[i] = sign(diff) as f32 * self.label_weight[i];
                    h[i] = 1.0;
                }
            }
            Kind::Gamma => {
                for i in 0..n {
                    let e = (-score[i]).exp();
                    let l = f64::from(label[i]);
                    // clang contracts `1.0 - l * e` into a fused multiply-subtract.
                    g[i] = madd(-l, e, 1.0) as f32;
                    h[i] = (l * e) as f32;
                }
            }
            Kind::Tweedie => {
                let rho = self.rho;
                for i in 0..n {
                    let e1 = ((1.0 - rho) * score[i]).exp();
                    let e2 = ((2.0 - rho) * score[i]).exp();
                    let nl = f64::from(-label[i]);
                    g[i] = madd(nl, e1, e2) as f32;
                    h[i] = madd(nl * (1.0 - rho), e1, (2.0 - rho) * e2) as f32;
                }
            }
        }
    }

    /// The `weights_ != nullptr` branches of upstream's `GetGradients`.
    fn gradients_weighted(&self, score: &[f64], g: &mut [f32], h: &mut [f32]) {
        let n = self.label.len();
        let (label, w) = (&self.label, &self.weights);
        match self.kind {
            Kind::L2 => {
                for i in 0..n {
                    g[i] = ((score[i] - f64::from(label[i])) as f32) * w[i];
                    h[i] = w[i];
                }
            }
            Kind::L1 => {
                for i in 0..n {
                    g[i] = sign(score[i] - f64::from(label[i])) as f32 * w[i];
                    h[i] = w[i];
                }
            }
            Kind::Quantile => {
                let a = self.alpha_f32;
                for i in 0..n {
                    let delta = (score[i] - f64::from(label[i])) as f32;
                    g[i] = if delta >= 0.0 {
                        (1.0f32 - a) * w[i]
                    } else {
                        -a * w[i]
                    };
                    h[i] = w[i];
                }
            }
            Kind::Huber => {
                for i in 0..n {
                    let diff = score[i] - f64::from(label[i]);
                    g[i] = if diff.abs() <= self.alpha {
                        (diff * f64::from(w[i])) as f32
                    } else {
                        (f64::from(sign(diff) as f32 * w[i]) * self.alpha) as f32
                    };
                    h[i] = w[i];
                }
            }
            Kind::Fair => {
                let c = self.fair_c;
                for i in 0..n {
                    let x = score[i] - f64::from(label[i]);
                    let wi = f64::from(w[i]);
                    g[i] = (c * x / (x.abs() + c) * wi) as f32;
                    h[i] = (c * c / ((x.abs() + c) * (x.abs() + c)) * wi) as f32;
                }
            }
            Kind::Poisson => {
                let emds = self.max_delta_step.exp();
                for i in 0..n {
                    let e = score[i].exp();
                    let wi = f64::from(w[i]);
                    g[i] = ((e - f64::from(label[i])) * wi) as f32;
                    h[i] = (e * emds * wi) as f32;
                }
            }
            Kind::Mape => {
                for i in 0..n {
                    let diff = score[i] - f64::from(label[i]);
                    g[i] = sign(diff) as f32 * self.label_weight[i];
                    h[i] = w[i];
                }
            }
            Kind::Gamma => {
                for i in 0..n {
                    let e = (-score[i]).exp();
                    let l = f64::from(label[i]);
                    let wi = f64::from(w[i]);
                    g[i] = (madd(-l, e, 1.0) * wi) as f32;
                    h[i] = (l * e * wi) as f32;
                }
            }
            Kind::Tweedie => {
                let rho = self.rho;
                for i in 0..n {
                    let e1 = ((1.0 - rho) * score[i]).exp();
                    let e2 = ((2.0 - rho) * score[i]).exp();
                    let nl = f64::from(-label[i]);
                    let wi = f64::from(w[i]);
                    g[i] = (madd(nl, e1, e2) * wi) as f32;
                    h[i] = (madd(nl * (1.0 - rho), e1, (2.0 - rho) * e2) * wi) as f32;
                }
            }
        }
    }
}

/// upstream: `Common::SafeLog`
fn safe_log(x: f64) -> f64 {
    if x > 0.0 { x.ln() } else { f64::NEG_INFINITY }
}

/// Element type of upstream's percentile macros (`label_t` or `double`); arithmetic on
/// two values happens in that type before widening, as in C++.
trait PVal: Copy + PartialOrd {
    fn widen(self) -> f64;
    fn diff(self, other: Self) -> f64;
    fn narrow(v: f64) -> Self;
}

impl PVal for f32 {
    fn widen(self) -> f64 {
        f64::from(self)
    }
    fn diff(self, other: Self) -> f64 {
        f64::from(self - other)
    }
    fn narrow(v: f64) -> Self {
        v as f32
    }
}

impl PVal for f64 {
    fn widen(self) -> f64 {
        self
    }
    fn diff(self, other: Self) -> f64 {
        self - other
    }
    fn narrow(v: f64) -> Self {
        v
    }
}

/// upstream: `ArrayArgs::ArgMax` (first maximum)
fn arg_max<T: PVal>(a: &[T]) -> usize {
    let mut best = 0;
    for i in 1..a.len() {
        if a[i] > a[best] {
            best = i;
        }
    }
    best
}

/// upstream: `ArrayArgs::ArgMin` (first minimum)
fn arg_min<T: PVal>(a: &[T]) -> usize {
    let mut best = 0;
    for i in 1..a.len() {
        if a[i] < a[best] {
            best = i;
        }
    }
    best
}

/// upstream: `ArrayArgs::Partition` (3-way partition, descending). Returns `(l, r)`.
fn partition<T: PVal>(a: &mut [T], start: isize, end: isize) -> (isize, isize) {
    let mut i = start - 1;
    let mut j = end - 1;
    let mut p = i;
    let mut q = j;
    if start >= end - 1 {
        return (start - 1, end);
    }
    let v = a[(end - 1) as usize];
    loop {
        loop {
            i += 1;
            if a[i as usize] > v {
                continue;
            }
            break;
        }
        loop {
            j -= 1;
            if v > a[j as usize] && j != start {
                continue;
            }
            break;
        }
        if i >= j {
            break;
        }
        a.swap(i as usize, j as usize);
        if a[i as usize] == v {
            p += 1;
            a.swap(p as usize, i as usize);
        }
        if v == a[j as usize] {
            q -= 1;
            a.swap(j as usize, q as usize);
        }
    }
    a.swap(i as usize, (end - 1) as usize);
    j = i - 1;
    i += 1;
    let mut k = start;
    while k <= p {
        a.swap(k as usize, j as usize);
        k += 1;
        j -= 1;
    }
    k = end - 2;
    while k >= q {
        a.swap(i as usize, k as usize);
        k -= 1;
        i += 1;
    }
    (j, i)
}

/// upstream: `ArrayArgs::ArgMaxAtK` (quickselect; `k` is an index, descending order)
fn arg_max_at_k<T: PVal>(a: &mut [T], start: isize, end: isize, k: isize) -> isize {
    if start >= end - 1 {
        return start;
    }
    let (l, r) = partition(a, start, end);
    if (k > l && k < r) || (l == start - 1 && r == end - 1) {
        k
    } else if k <= l {
        arg_max_at_k(a, start, l + 1, k)
    } else {
        arg_max_at_k(a, r, end, k)
    }
}

/// upstream: `PercentileFun` macro
fn percentile<T: PVal>(n: usize, read: impl Fn(usize) -> T, alpha: f64) -> T {
    if n <= 1 {
        return read(0);
    }
    let mut r: Vec<T> = (0..n).map(read).collect();
    let float_pos = (n - 1) as f64 * (1.0 - alpha);
    let pos = float_pos as i32 + 1;
    if pos < 1 {
        return r[arg_max(&r)];
    }
    let pos = pos as usize;
    if pos >= n {
        return r[arg_min(&r)];
    }
    let bias = float_pos - (pos - 1) as f64;
    let (v1, v2) = if pos > n / 2 {
        arg_max_at_k(&mut r, 0, n as isize, pos as isize - 1);
        (r[pos - 1], r[pos + arg_max(&r[pos..])])
    } else {
        arg_max_at_k(&mut r, 0, n as isize, pos as isize);
        (r[arg_min(&r[..pos])], r[pos])
    };
    // `v1 - (v1 - v2) * bias`, contracted by clang into a fused multiply-subtract.
    T::narrow(madd(-v1.diff(v2), bias, v1.widen()))
}

/// upstream: `WeightedPercentileFun` macro
fn weighted_percentile<T: PVal>(
    n: usize,
    read: impl Fn(usize) -> T,
    weight: impl Fn(usize) -> f32,
    alpha: f64,
) -> T {
    if n <= 1 {
        return read(0);
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        let (x, y) = (read(a), read(b));
        if x < y {
            std::cmp::Ordering::Less
        } else if y < x {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    let mut cdf = Vec::with_capacity(n);
    let mut acc = f64::from(weight(idx[0]));
    cdf.push(acc);
    for &i in &idx[1..] {
        acc += f64::from(weight(i));
        cdf.push(acc);
    }
    let threshold = cdf[n - 1] * alpha;
    let pos = cdf.partition_point(|&c| c <= threshold).min(n - 1);
    if pos == 0 || pos == n - 1 {
        return read(idx[pos]);
    }
    let v1 = read(idx[pos - 1]);
    let v2 = read(idx[pos]);
    if cdf[pos] - cdf[pos - 1] >= 1.0 {
        let t = (threshold - cdf[pos - 1]) / (cdf[pos] - cdf[pos - 1]);
        T::narrow(madd(t, v2.diff(v1), v1.widen()))
    } else {
        v1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_matches_order_statistics() {
        let data = [5.0f64, 1.0, 4.0, 4.0, 2.0, 9.0, 7.0, 3.0];
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| b.total_cmp(a));
        for &alpha in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let got = percentile(data.len(), |i| data[i], alpha);
            let fp = (data.len() - 1) as f64 * (1.0 - alpha);
            let pos = fp as usize + 1;
            let bias = fp - (pos - 1) as f64;
            let (v1, v2) = (sorted[pos - 1], sorted[pos]);
            assert!(
                (got - (v1 - (v1 - v2) * bias)).abs() < 1e-12,
                "alpha {alpha}"
            );
        }
    }

    #[test]
    fn arg_max_at_k_selects_kth_largest() {
        let base = [3.0f64, 3.0, 1.0, 8.0, 5.0, 5.0, 0.0, 7.0, 3.0];
        let mut sorted = base.to_vec();
        sorted.sort_by(|a, b| b.total_cmp(a));
        for k in 0..base.len() {
            let mut a = base.to_vec();
            arg_max_at_k(&mut a, 0, base.len() as isize, k as isize);
            assert_eq!(a[k], sorted[k], "k={k}");
        }
    }
}
