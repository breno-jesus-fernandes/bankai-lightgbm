//! Independent-lane split arithmetic, preserving scalar accumulation and tie order.
//! upstream: src/treelearner/feature_histogram.hpp (v4.7.0), GetSplitGains / GetLeafGain.

/// Scalar fallback for the unconstrained, unsmoothed split gain without L1 regularization.
#[inline(always)]
pub fn split_gain_scalar(gl: f64, hl: f64, gr: f64, hr: f64, l2: f64) -> f64 {
    gl * gl / (hl + l2) + gr * gr / (hr + l2)
}

/// Evaluate left and right gains in independent SIMD lanes, then add in scalar order.
/// No prefix sums, reciprocal approximations, or fused operations are used.
#[inline(always)]
pub fn split_gain(gl: f64, hl: f64, gr: f64, hr: f64, l2: f64) -> f64 {
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    if std::arch::is_aarch64_feature_detected!("neon") {
        use std::arch::aarch64::*;
        // SAFETY: NEON was detected; all operations use register values, with no pointers.
        unsafe {
            let g = vsetq_lane_f64::<1>(gr, vdupq_n_f64(gl));
            let h = vsetq_lane_f64::<1>(hr, vdupq_n_f64(hl));
            let gain = vdivq_f64(vmulq_f64(g, g), vaddq_f64(h, vdupq_n_f64(l2)));
            return vgetq_lane_f64::<0>(gain) + vgetq_lane_f64::<1>(gain);
        }
    }
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if std::arch::is_x86_feature_detected!("sse2") {
        use std::arch::x86_64::*;
        // SAFETY: SSE2 was detected; all operations use register values, with no pointers.
        unsafe {
            let g = _mm_set_pd(gr, gl);
            let h = _mm_set_pd(hr, hl);
            let gain = _mm_div_pd(_mm_mul_pd(g, g), _mm_add_pd(h, _mm_set1_pd(l2)));
            return _mm_cvtsd_f64(gain) + _mm_cvtsd_f64(_mm_unpackhi_pd(gain, gain));
        }
    }
    split_gain_scalar(gl, hl, gr, hr, l2)
}
