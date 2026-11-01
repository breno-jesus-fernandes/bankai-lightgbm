//! upstream: include/LightGBM/utils/random.h — a small LCG used for all of LightGBM's
//! sampling. Reproduced exactly so seeds give identical results.

use std::collections::BTreeSet;

#[derive(Debug, Clone)]
pub struct Random {
    x: u32,
}

impl Default for Random {
    fn default() -> Self {
        Random { x: 123_456_789 }
    }
}

impl Random {
    pub fn new(seed: i32) -> Self {
        Random { x: seed as u32 }
    }

    #[inline]
    fn rand_int16(&mut self) -> i32 {
        self.x = self.x.wrapping_mul(214_013).wrapping_add(2_531_011);
        ((self.x >> 16) & 0x7FFF) as i32
    }

    #[inline]
    fn rand_int32(&mut self) -> i32 {
        self.x = self.x.wrapping_mul(214_013).wrapping_add(2_531_011);
        (self.x & 0x7FFF_FFFF) as i32
    }

    /// Random integer in `[lower, upper)` from 15 random bits.
    #[inline]
    pub fn next_short(&mut self, lower: i32, upper: i32) -> i32 {
        self.rand_int16() % (upper - lower) + lower
    }

    /// Random integer in `[lower, upper)` from 31 random bits.
    #[inline]
    pub fn next_int(&mut self, lower: i32, upper: i32) -> i32 {
        self.rand_int32() % (upper - lower) + lower
    }

    /// Random float in `[0, 1)`.
    #[inline]
    pub fn next_float(&mut self) -> f32 {
        self.rand_int16() as f32 / 32768.0
    }

    /// upstream: `Random::Sample` — `k` sorted distinct indices from `0..n`.
    pub fn sample(&mut self, n: i32, k: i32) -> Vec<i32> {
        let mut ret = Vec::with_capacity(k.max(0) as usize);
        if k > n || k <= 0 {
            return ret;
        }
        if k == n {
            return (0..n).collect();
        }
        if k > 1 && f64::from(k) > f64::from(n) / f64::from(k).log2() {
            for i in 0..n {
                let prob = (k as usize - ret.len()) as f64 / f64::from(n - i);
                if f64::from(self.next_float()) < prob {
                    ret.push(i);
                }
            }
        } else {
            let mut set = BTreeSet::new();
            for r in (n - k)..n {
                let v = self.next_int(0, r + 1);
                if !set.insert(v) {
                    set.insert(r);
                }
            }
            ret.extend(set);
        }
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_msvc_style_lcg() {
        // x1 = 214013 * 42 + 2531011 = 11519557; (x1 >> 16) & 0x7fff = 175
        let mut r = Random::new(42);
        assert_eq!(r.next_short(0, 32767), 175);
    }

    #[test]
    fn sample_is_sorted_distinct_and_sized() {
        let mut r = Random::new(7);
        for (n, k) in [(100, 10), (100, 90), (10, 10), (5, 0), (3, 5)] {
            let s = r.sample(n, k);
            if k > n || k <= 0 {
                assert!(s.is_empty());
                continue;
            }
            assert!(s.windows(2).all(|w| w[0] < w[1]));
            assert!(s.iter().all(|&v| (0..n).contains(&v)));
            if k <= n / 2 {
                assert_eq!(s.len(), k as usize);
            }
        }
    }
}
