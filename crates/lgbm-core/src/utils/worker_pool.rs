//! A per-Booster `rayon` thread pool for the short parallel regions of one tree.
//!
//! SPEC-CONC-1: parallel loops use `rayon`, each Booster owns one scoped `rayon::ThreadPool`
//! sized from `num_threads`, and the global pool is never used for training. Every region runs
//! through `install`, so the work stays on this pool's threads.
//!
//! The parallel primitives here replace `utils::spin_pool`. Each task writes a disjoint piece of
//! the output and the callers merge partials in index order (SPEC-CONC-2), so results do not
//! depend on how rayon schedules the tasks and `thread_invariance` holds across thread counts.

use rayon::prelude::*;

use crate::error::{Error, Result};

/// A fixed-size `rayon` pool. The learner only builds one when `num_threads > 1`.
pub struct WorkerPool {
    pool: std::sync::Arc<rayon::ThreadPool>,
    threads: usize,
}

impl WorkerPool {
    /// Builds a dedicated pool of `threads` workers (never the global rayon pool).
    pub fn new(threads: usize) -> Result<Self> {
        let threads = threads.max(1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|k| format!("bankai-worker-{k}"))
            .build()
            .map_err(|e| Error::Io(std::io::Error::other(e.to_string())))?;
        Ok(Self {
            pool: std::sync::Arc::new(pool),
            threads,
        })
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// A handle to the underlying pool, so the caller can run a whole tree inside one
    /// `install` (the per-region primitives below then nest cheaply on the same pool).
    pub fn handle(&self) -> std::sync::Arc<rayon::ThreadPool> {
        std::sync::Arc::clone(&self.pool)
    }

    /// Runs `f(i)` for every `i < n` on this pool and returns when all have finished. A panic in
    /// any task is re-raised on the calling thread.
    pub fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        if self.threads == 1 || n <= 1 {
            for i in 0..n {
                f(i);
            }
            return;
        }
        self.pool.install(|| {
            (0..n).into_par_iter().for_each(f);
        });
    }

    /// Calls `f(i, &mut items[i])` for every item, in parallel over disjoint elements.
    pub fn for_each_mut<T: Send>(&self, items: &mut [T], f: impl Fn(usize, &mut T) + Sync) {
        if self.threads == 1 || items.len() <= 1 {
            for (i, x) in items.iter_mut().enumerate() {
                f(i, x);
            }
            return;
        }
        self.pool.install(|| {
            items.par_iter_mut().enumerate().for_each(|(i, x)| f(i, x));
        });
    }

    /// Calls `f(c, chunk)` for each `chunk`-sized piece of `data` (the last may be shorter).
    pub fn for_each_chunk_mut<T: Send>(
        &self,
        data: &mut [T],
        chunk: usize,
        f: impl Fn(usize, &mut [T]) + Sync,
    ) {
        let chunk = chunk.max(1);
        if self.threads == 1 {
            for (c, piece) in data.chunks_mut(chunk).enumerate() {
                f(c, piece);
            }
            return;
        }
        self.pool.install(|| {
            data.par_chunks_mut(chunk)
                .enumerate()
                .for_each(|(c, piece)| f(c, piece));
        });
    }

    /// Calls `f(j, &mut data[lo..hi])` for each `(lo, hi)` in `ranges`, one task per range.
    /// `ranges` must be ascending and disjoint.
    pub fn for_each_range_mut<T: Send>(
        &self,
        data: &mut [T],
        ranges: &[(usize, usize)],
        f: impl Fn(usize, &mut [T]) + Sync,
    ) -> Result<()> {
        let total = data.len();
        let mut pieces: Vec<&mut [T]> = Vec::with_capacity(ranges.len());
        let mut rest = data;
        let mut pos = 0usize;
        for &(lo, hi) in ranges {
            if lo < pos || hi < lo || hi > total {
                return Err(Error::InvalidParameter(
                    "histogram ranges must be ascending and disjoint".into(),
                ));
            }
            let (_skip, after_skip) = rest.split_at_mut(lo - pos);
            let (piece, tail) = after_skip.split_at_mut(hi - lo);
            pieces.push(piece);
            rest = tail;
            pos = hi;
        }
        self.for_each_mut(&mut pieces, |j, piece| f(j, piece));
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

    #[test]
    fn run_executes_every_index_exactly_once() {
        let pool = WorkerPool::new(3).unwrap();
        let hits: Vec<AtomicU32> = (0..11).map(|_| AtomicU32::new(0)).collect();
        pool.run(11, &|i| {
            hits[i].fetch_add(1, Ordering::Relaxed);
        });
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
    }

    #[test]
    fn for_each_mut_updates_every_item() {
        let pool = WorkerPool::new(2).unwrap();
        let mut v: Vec<u64> = (0..9).collect();
        pool.for_each_mut(&mut v, |i, x| *x = *x * 10 + i as u64);
        assert_eq!(v, (0..9).map(|i| i * 11).collect::<Vec<u64>>());
    }

    #[test]
    fn for_each_range_mut_hands_out_disjoint_slices() {
        let pool = WorkerPool::new(2).unwrap();
        let mut data = vec![0u32; 10];
        pool.for_each_range_mut(&mut data, &[(0, 3), (5, 9)], |j, s| {
            for x in s.iter_mut() {
                *x = j as u32 + 1;
            }
        })
        .unwrap();
        assert_eq!(data, [1, 1, 1, 0, 0, 2, 2, 2, 2, 0]);
    }

    #[test]
    fn for_each_range_mut_rejects_overlapping_ranges() {
        let pool = WorkerPool::new(2).unwrap();
        let mut data = vec![0u32; 10];
        let err = pool.for_each_range_mut(&mut data, &[(0, 5), (3, 8)], |_, _| {});
        assert!(err.is_err());
    }

    #[test]
    fn chunks_cover_the_slice_in_order() {
        let pool = WorkerPool::new(3).unwrap();
        let mut data = vec![0usize; 7];
        pool.for_each_chunk_mut(&mut data, 3, |c, s| {
            for x in s.iter_mut() {
                *x = c;
            }
        });
        assert_eq!(data, [0, 0, 0, 1, 1, 1, 2]);
    }

    #[test]
    fn worker_panic_is_reraised_on_the_caller() {
        let pool = WorkerPool::new(2).unwrap();
        let result = catch_unwind(AssertUnwindSafe(|| {
            pool.run(4, &|i| {
                if i == 3 {
                    panic!("task failure");
                }
            });
        }));
        assert!(result.is_err());
        // The pool stays usable afterwards.
        let hits = AtomicUsize::new(0);
        pool.run(4, &|_| {
            hits.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(hits.load(Ordering::Relaxed), 4);
    }
}
