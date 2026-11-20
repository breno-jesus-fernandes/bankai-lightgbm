//! A port of C++'s `std::mt19937`, the 32-bit Mersenne Twister engine.
//!
//! upstream: src/treelearner/gradient_discretizer.cpp (v4.7.0) seeds
//! `std::mt19937` to draw the per-row noise for quantized stochastic rounding.
//! Unlike the rest of LightGBM, that path does not use the small LCG
//! (`utils::random`, SPEC-NUM-9); it relies on the standard library engine.
//!
//! The engine is defined by the C++ standard with fixed parameters
//! (`mersenne_twister_engine<uint32_t, 32, 624, 397, 31, 0x9908b0df, 11,
//! 0xffffffff, 7, 0x9d2c5680, 15, 0xefc60000, 18, 1812433253>`), so its raw
//! 32-bit output sequence is identical on every platform and standard library.
//! Only the *distributions* layered on top (`uniform_real_distribution`,
//! `uniform_int_distribution`) are implementation defined; those are handled
//! separately by the quantized stochastic-rounding work (T2.7).

/// Degree of recurrence (state size, in words).
const N: usize = 624;
/// Middle word (recurrence offset).
const M: usize = 397;
/// Most significant `w - r` bits (`r = 31`).
const UPPER_MASK: u32 = 0x8000_0000;
/// Least significant `r` bits.
const LOWER_MASK: u32 = 0x7fff_ffff;
/// Twist coefficient `a`.
const MATRIX_A: u32 = 0x9908_b0df;
/// Seeding multiplier `f`.
const INIT_MULT: u32 = 1_812_433_253;
/// The standard's default seed.
const DEFAULT_SEED: u32 = 5489;

/// The 32-bit Mersenne Twister, matching `std::mt19937`.
pub struct Mt19937 {
    state: [u32; N],
    index: usize,
}

impl Default for Mt19937 {
    fn default() -> Self {
        Self::new(DEFAULT_SEED)
    }
}

impl Mt19937 {
    /// Seed the engine as `std::mt19937(seed)` does.
    ///
    /// `state[0] = seed`, then
    /// `state[i] = f * (state[i-1] ^ (state[i-1] >> 30)) + i`, all modulo 2^32.
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; N];
        state[0] = seed;
        for i in 1..N {
            let prev = state[i - 1];
            state[i] = INIT_MULT
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        // Force a full twist on the first draw, like the standard engine.
        Self { state, index: N }
    }

    /// Regenerate the whole state vector (the Mersenne Twister "twist").
    fn twist(&mut self) {
        for i in 0..N {
            let y = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % N] & LOWER_MASK);
            let mut next = self.state[(i + M) % N] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= MATRIX_A;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }

    /// Return the next 32-bit output, matching `std::mt19937::operator()`.
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        // Tempering (u=11, d=0xffffffff, s=7, b=0x9d2c5680, t=15, c=0xefc60000, l=18).
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// libc++ `std::uniform_real_distribution<double>(0, 1)` over this engine.
    ///
    /// libc++ returns `(b - a) * generate_canonical<double, 53>(g) + a`, which for `(0, 1)` is
    /// the canonical value itself. With the engine's 32-bit range, `generate_canonical<double,
    /// 53>` draws twice: `sp = g1 + g2 * 2^32` (first draw low), returned as `sp / 2^64`, with
    /// the additions and division in this order so the double rounding matches libc++.
    pub fn uniform_real_01(&mut self) -> f64 {
        let g1 = f64::from(self.next_u32());
        let g2 = f64::from(self.next_u32());
        let rp = 4_294_967_296.0_f64; // 2^32, the engine's range
        let sp = g1 + g2 * rp;
        sp / (rp * rp)
    }

    /// libc++ `std::uniform_int_distribution<int>(0, n)` (inclusive) over this engine.
    ///
    /// libc++ takes the low `w` bits of one engine draw (`w` the width that covers `n + 1`,
    /// bumped by one when `n + 1` is not a clean power of two within that width) and rejects
    /// values `>= n + 1`. `n + 1` fits in `u32` here (`n` is a row count), so a single draw per
    /// attempt suffices — no multi-word `__independent_bits_engine` assembly is needed.
    pub fn uniform_int_inclusive(&mut self, n: u32) -> u32 {
        let rp = n.wrapping_add(1);
        if rp == 1 {
            return 0; // n == 0: the only value is 0
        }
        if rp == 0 {
            return self.next_u32(); // n == u32::MAX: the whole 32-bit range
        }
        let mut w = 32 - rp.leading_zeros() - 1;
        if rp & (u32::MAX >> (32 - w)) != 0 {
            w += 1;
        }
        let mask = if w >= 32 { u32::MAX } else { (1u32 << w) - 1 };
        loop {
            let u = self.next_u32() & mask;
            if u < rp {
                return u;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_predefined_vector() {
        // ISO/IEC 14882 [rand.predef]: the 10000th invocation is 4123659995.
        let mut engine = Mt19937::default();
        let mut value = 0;
        for _ in 0..10_000 {
            value = engine.next_u32();
        }
        assert_eq!(value, 4_123_659_995);
    }
}
