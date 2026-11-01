//! Persistent worker threads for the short parallel regions of one tree.
//!
//! Upstream's OpenMP threads spin between regions, so a region costs about a microsecond to
//! start. Rayon's workers sleep after a short idle period, and waking them costs 15-30 us on
//! the 2-vCPU benchmark machine, which is more than many training regions take.
//!
//! This is the only module where `unsafe` is used for parallelism; the learner reaches it
//! through the safe helpers below.

#![allow(unsafe_code)]

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;

use crate::error::{Error, Result};

/// Spins before an idle worker sleeps on the condition variable.
const SPIN_LIMIT: u32 = 200_000;

/// Keeps hot atomics on separate cache lines, so spinning workers and the caller do not
/// invalidate each other's lines.
#[repr(align(128))]
struct Padded<T>(T);

struct Shared {
    /// Bumped once per region; a worker runs a region when it sees a new value.
    generation: Padded<AtomicUsize>,
    /// Workers that finished the current region.
    done: Padded<AtomicUsize>,
    tasks: Padded<AtomicUsize>,
    /// The current region's task function, as the two words of a `&dyn Fn(usize)`.
    job: Mutex<Option<JobPtr>>,
    /// Set when a worker task panicked during the current region.
    panicked: AtomicBool,
    sleepers: AtomicUsize,
    shutdown: Padded<AtomicBool>,
    sleep_lock: Mutex<()>,
    wake: Condvar,
}

/// A fixed set of threads that run `n` tasks per region. Task `i` runs on thread
/// `i % threads`; the calling thread is thread 0. Regions must not be nested.
pub struct SpinPool {
    shared: Arc<Shared>,
    threads: usize,
    handles: Vec<JoinHandle<()>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl SpinPool {
    /// Starts `threads - 1` workers; the caller is the remaining thread.
    pub fn new(threads: usize) -> Result<Self> {
        let threads = threads.max(1);
        let shared = Arc::new(Shared {
            generation: Padded(AtomicUsize::new(0)),
            done: Padded(AtomicUsize::new(0)),
            tasks: Padded(AtomicUsize::new(0)),
            job: Mutex::new(None),
            panicked: AtomicBool::new(false),
            sleepers: AtomicUsize::new(0),
            shutdown: Padded(AtomicBool::new(false)),
            sleep_lock: Mutex::new(()),
            wake: Condvar::new(),
        });
        let mut handles = Vec::with_capacity(threads - 1);
        for k in 1..threads {
            let s = Arc::clone(&shared);
            let spawned = std::thread::Builder::new()
                .name(format!("bankai-worker-{k}"))
                .spawn(move || worker(&s, k, threads));
            match spawned {
                Ok(h) => handles.push(h),
                Err(e) => {
                    // Stop the workers already started before reporting the failure.
                    let partial = SpinPool {
                        shared,
                        threads,
                        handles,
                    };
                    drop(partial);
                    return Err(Error::Io(e));
                }
            }
        }
        Ok(SpinPool {
            shared,
            threads,
            handles,
        })
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Runs `f(i)` for every `i < n` and returns when all of them have finished. A panic in
    /// any task is re-raised on the calling thread after the region has drained.
    pub fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        if self.threads == 1 || n <= 1 {
            for i in 0..n {
                f(i);
            }
            return;
        }
        let s = &*self.shared;
        // SAFETY: a worker reads this pointer only after it sees the generation bump below,
        // and this function does not return before every worker has incremented `done`, so
        // `f` is alive for every use. The transmute only erases the lifetime of the reference;
        // a raw fat pointer keeps the data and vtable provenance (storing it as `[usize; 2]`
        // would strip the vtable's provenance, which Miri reports as undefined behaviour).
        let raw = JobPtr(unsafe {
            std::mem::transmute::<&(dyn Fn(usize) + Sync), *const (dyn Fn(usize) + Sync)>(f)
        });
        *lock(&s.job) = Some(raw);
        s.tasks.0.store(n, Ordering::Relaxed);
        s.done.0.store(0, Ordering::Relaxed);
        s.panicked.store(false, Ordering::Relaxed);
        s.generation.0.fetch_add(1, Ordering::SeqCst);
        if s.sleepers.load(Ordering::SeqCst) > 0 {
            let _g = lock(&s.sleep_lock);
            s.wake.notify_all();
        }
        let own = catch_unwind(AssertUnwindSafe(|| {
            let mut i = 0;
            while i < n {
                f(i);
                i += self.threads;
            }
        }));
        while s.done.0.load(Ordering::Acquire) < self.threads - 1 {
            std::hint::spin_loop();
        }
        if let Err(payload) = own {
            resume_unwind(payload);
        }
        if s.panicked.load(Ordering::Acquire) {
            resume_unwind(Box::new("a spin_pool worker task panicked"));
        }
    }

    /// Calls `f(i, &mut items[i])` for every item, one task per item.
    pub fn for_each_mut<T: Send>(&self, items: &mut [T], f: impl Fn(usize, &mut T) + Sync) {
        let cells: Vec<Mutex<&mut T>> = items.iter_mut().map(Mutex::new).collect();
        self.run(cells.len(), &|i| {
            let mut cell = lock(&cells[i]);
            f(i, &mut **cell);
        });
    }

    /// Calls `f(c, chunk)` for each `chunk`-sized piece of `data` (the last may be shorter).
    pub fn for_each_chunk_mut<T: Send>(
        &self,
        data: &mut [T],
        chunk: usize,
        f: impl Fn(usize, &mut [T]) + Sync,
    ) {
        let mut chunks: Vec<&mut [T]> = data.chunks_mut(chunk.max(1)).collect();
        self.for_each_mut(&mut chunks, |c, piece| f(c, piece));
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

/// The published task: a raw pointer to the `run` caller's closure. It is valid for every
/// worker until `run` returns, which waits for all workers to finish.
#[derive(Clone, Copy)]
struct JobPtr(*const (dyn Fn(usize) + Sync));

// SAFETY: the pointee is `Sync`, so it may be called from any thread, and `run` keeps it alive
// until every worker is done with it.
unsafe impl Send for JobPtr {}

fn worker(s: &Shared, k: usize, threads: usize) {
    let mut seen = 0usize;
    loop {
        let mut spins = 0u32;
        loop {
            let g = s.generation.0.load(Ordering::SeqCst);
            if g != seen {
                seen = g;
                break;
            }
            if s.shutdown.0.load(Ordering::SeqCst) {
                return;
            }
            if spins < SPIN_LIMIT {
                spins += 1;
                std::hint::spin_loop();
            } else {
                let guard = lock(&s.sleep_lock);
                s.sleepers.fetch_add(1, Ordering::SeqCst);
                let guard = s
                    .wake
                    .wait_while(guard, |_| {
                        s.generation.0.load(Ordering::SeqCst) == seen
                            && !s.shutdown.0.load(Ordering::SeqCst)
                    })
                    .unwrap_or_else(|e| e.into_inner());
                s.sleepers.fetch_sub(1, Ordering::SeqCst);
                drop(guard);
                spins = 0;
            }
        }
        let raw = *lock(&s.job);
        if let Some(raw) = raw {
            let n = s.tasks.0.load(Ordering::Relaxed);
            // SAFETY: `raw` was made from a live `&dyn Fn(usize) + Sync` by `run`, which is
            // still blocked until this worker increments `done` below.
            let f: &(dyn Fn(usize) + Sync) = unsafe { &*raw.0 };
            let result = catch_unwind(AssertUnwindSafe(|| {
                let mut i = k;
                while i < n {
                    f(i);
                    i += threads;
                }
            }));
            if result.is_err() {
                s.panicked.store(true, Ordering::Release);
            }
        }
        s.done.0.fetch_add(1, Ordering::Release);
    }
}

impl Drop for SpinPool {
    fn drop(&mut self) {
        self.shared.shutdown.0.store(true, Ordering::SeqCst);
        {
            let _g = lock(&self.shared.sleep_lock);
            self.shared.wake.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn run_executes_every_index_exactly_once() {
        let pool = SpinPool::new(3).unwrap();
        let hits: Vec<AtomicU32> = (0..11).map(|_| AtomicU32::new(0)).collect();
        pool.run(11, &|i| {
            hits[i].fetch_add(1, Ordering::Relaxed);
        });
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
    }

    #[test]
    fn workers_are_reused_across_many_regions() {
        let pool = SpinPool::new(2).unwrap();
        let total = AtomicUsize::new(0);
        for round in 0..500 {
            pool.run(4, &|i| {
                total.fetch_add(i + round, Ordering::Relaxed);
            });
        }
        let expected: usize = (0..500).map(|r| 4 * r + 6).sum();
        assert_eq!(total.load(Ordering::Relaxed), expected);
    }

    #[test]
    fn for_each_mut_updates_every_item() {
        let pool = SpinPool::new(2).unwrap();
        let mut v: Vec<u64> = (0..9).collect();
        pool.for_each_mut(&mut v, |i, x| *x = *x * 10 + i as u64);
        assert_eq!(v, (0..9).map(|i| i * 11).collect::<Vec<u64>>());
    }

    #[test]
    fn for_each_range_mut_hands_out_disjoint_slices() {
        let pool = SpinPool::new(2).unwrap();
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
        let pool = SpinPool::new(2).unwrap();
        let mut data = vec![0u32; 10];
        let err = pool.for_each_range_mut(&mut data, &[(0, 5), (3, 8)], |_, _| {});
        assert!(err.is_err());
    }

    #[test]
    fn chunks_cover_the_slice_in_order() {
        let pool = SpinPool::new(3).unwrap();
        let mut data = vec![0usize; 7];
        pool.for_each_chunk_mut(&mut data, 3, |c, s| {
            for x in s.iter_mut() {
                *x = c;
            }
        });
        assert_eq!(data, [0, 0, 0, 1, 1, 1, 2]);
    }

    #[test]
    fn worker_panic_is_reraised_on_the_caller_without_deadlock() {
        let pool = SpinPool::new(2).unwrap();
        // Task 1 runs on the worker thread.
        let result = catch_unwind(AssertUnwindSafe(|| {
            pool.run(2, &|i| {
                if i == 1 {
                    panic!("task failure");
                }
            });
        }));
        assert!(result.is_err());
        // The pool stays usable afterwards.
        let hits = AtomicUsize::new(0);
        pool.run(2, &|_| {
            hits.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(hits.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn dropping_the_pool_joins_its_workers() {
        let pool = SpinPool::new(4).unwrap();
        pool.run(4, &|_| {});
        drop(pool);
    }
}
