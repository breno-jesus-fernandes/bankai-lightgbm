//! Quickselect helpers. upstream: include/LightGBM/utils/array_args.h
//!
//! Upstream's selection order matters for ties and for the element left at position `k`,
//! so these are straight ports rather than `select_nth_unstable`.

/// Element type of upstream's percentile macros (`label_t` or `double`); arithmetic on
/// two values happens in that type before widening, as in C++.
pub(crate) trait PVal: Copy + PartialOrd {
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
pub(crate) fn arg_max<T: PVal>(a: &[T]) -> usize {
    let mut best = 0;
    for i in 1..a.len() {
        if a[i] > a[best] {
            best = i;
        }
    }
    best
}

/// upstream: `ArrayArgs::ArgMin` (first minimum)
pub(crate) fn arg_min<T: PVal>(a: &[T]) -> usize {
    let mut best = 0;
    for i in 1..a.len() {
        if a[i] < a[best] {
            best = i;
        }
    }
    best
}

/// upstream: `ArrayArgs::Partition` (3-way partition, descending). Returns `(l, r)`.
pub(crate) fn partition<T: PVal>(a: &mut [T], start: isize, end: isize) -> (isize, isize) {
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
pub(crate) fn arg_max_at_k<T: PVal>(a: &mut [T], start: isize, end: isize, k: isize) -> isize {
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
