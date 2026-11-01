//! EXPERIMENT (not for merge until the unsafe policy is decided): a persistent pool whose
//! workers spin between parallel regions, the way upstream's OpenMP threads do. Rayon's
//! workers sleep after a short idle period, and waking them costs 15-30 us on the
//! 2-vCPU benchmark machine, more than many training regions take.
#![allow(unsafe_code)]

use std::sync::atomic::{AtomicBool, AtomicUsize, AtomicU64, Ordering};

pub static CALLER_TASK_NS: AtomicU64 = AtomicU64::new(0);
pub static CALLER_WAIT_NS: AtomicU64 = AtomicU64::new(0);
pub static WORKER_TASK_NS: AtomicU64 = AtomicU64::new(0);
pub static REGIONS: AtomicU64 = AtomicU64::new(0);
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// Spins before a worker sleeps on the condition variable.
const SPIN_LIMIT: u32 = 2_000_000_000;

struct Shared {
    /// Bumped once per region; workers wait for it to change.
    generation: AtomicUsize,
    /// The current region's task function, as the two words of a `&dyn Fn(usize)`.
    job: Mutex<Option<[usize; 2]>>,
    tasks: AtomicUsize,
    /// Workers that finished the current region.
    done: AtomicUsize,
    sleepers: AtomicUsize,
    shutdown: AtomicBool,
    sleep_lock: Mutex<()>,
    wake: Condvar,
}

/// A fixed set of threads that run `n` tasks per region. Task `i` runs on thread
/// `i % threads`; the caller is thread 0. Regions must not nest.
pub struct SpinPool {
    shared: Arc<Shared>,
    threads: usize,
    handles: Vec<JoinHandle<()>>,
}

impl SpinPool {
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let shared = Arc::new(Shared {
            generation: AtomicUsize::new(0),
            job: Mutex::new(None),
            tasks: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            sleep_lock: Mutex::new(()),
            wake: Condvar::new(),
        });
        let handles = (1..threads)
            .map(|k| {
                let s = Arc::clone(&shared);
                std::thread::spawn(move || worker(&s, k, threads))
            })
            .collect();
        SpinPool {
            shared,
            threads,
            handles,
        }
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Runs `f(i)` for every `i < n` and returns when all have finished.
    pub fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        if self.threads == 1 || n <= 1 {
            for i in 0..n {
                f(i);
            }
            return;
        }
        let s = &*self.shared;
        // SAFETY: the pointer is only read by workers between the generation bump below and
        // their `done` increment, and this function does not return before every worker has
        // incremented `done`. So `f` outlives every use. Pointer layout: a fat `&dyn` is two
        // words; `transmute` only changes the lifetime, which is the sole part erased here.
        let raw: [usize; 2] =
            unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), [usize; 2]>(f) };
        *s.job.lock().unwrap_or_else(|e| e.into_inner()) = Some(raw);
        s.tasks.store(n, Ordering::Relaxed);
        s.done.store(0, Ordering::Relaxed);
        s.generation.fetch_add(1, Ordering::SeqCst);
        if s.sleepers.load(Ordering::SeqCst) > 0 {
            let _g = s.sleep_lock.lock().unwrap_or_else(|e| e.into_inner());
            s.wake.notify_all();
        }
        REGIONS.fetch_add(1, Ordering::Relaxed);
        let t0 = std::time::Instant::now();
        let mut i = 0;
        while i < n {
            f(i);
            i += self.threads;
        }
        let t1 = std::time::Instant::now();
        while s.done.load(Ordering::Acquire) < self.threads - 1 {
            std::hint::spin_loop();
        }
        CALLER_TASK_NS.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
        CALLER_WAIT_NS.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

fn worker(s: &Shared, k: usize, threads: usize) {
    let mut seen = 0usize;
    loop {
        let mut spins = 0u32;
        loop {
            let g = s.generation.load(Ordering::SeqCst);
            if g != seen {
                seen = g;
                break;
            }
            if s.shutdown.load(Ordering::SeqCst) {
                return;
            }
            if spins < SPIN_LIMIT {
                spins += 1;
                std::hint::spin_loop();
            } else {
                let guard = s.sleep_lock.lock().unwrap_or_else(|e| e.into_inner());
                s.sleepers.fetch_add(1, Ordering::SeqCst);
                let guard = s
                    .wake
                    .wait_while(guard, |_| {
                        s.generation.load(Ordering::SeqCst) == seen
                            && !s.shutdown.load(Ordering::SeqCst)
                    })
                    .unwrap_or_else(|e| e.into_inner());
                s.sleepers.fetch_sub(1, Ordering::SeqCst);
                drop(guard);
                spins = 0;
            }
        }
        let raw = match *s.job.lock().unwrap_or_else(|e| e.into_inner()) {
            Some(r) => r,
            None => continue,
        };
        let n = s.tasks.load(Ordering::Relaxed);
        // SAFETY: `raw` was made from a live `&dyn Fn(usize) + Sync` by `run`, which is still
        // blocked until this worker increments `done`; the reference is shared (`Sync`).
        let f: &(dyn Fn(usize) + Sync) =
            unsafe { std::mem::transmute::<[usize; 2], &(dyn Fn(usize) + Sync)>(raw) };
        let t0 = std::time::Instant::now();
        let mut i = k;
        while i < n {
            f(i);
            i += threads;
        }
        WORKER_TASK_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        s.done.fetch_add(1, Ordering::Release);
    }
}

impl Drop for SpinPool {
    fn drop(&mut self) {
        if std::env::var("BK_POOL_STATS").is_ok() { eprintln!("regions {} caller_tasks {:.1}ms caller_wait {:.1}ms worker_tasks {:.1}ms", REGIONS.swap(0,Ordering::Relaxed), CALLER_TASK_NS.swap(0,Ordering::Relaxed) as f64/1e6, CALLER_WAIT_NS.swap(0,Ordering::Relaxed) as f64/1e6, WORKER_TASK_NS.swap(0,Ordering::Relaxed) as f64/1e6); }
        self.shared.shutdown.store(true, Ordering::SeqCst);
        {
            let _g = self.shared.sleep_lock.lock().unwrap_or_else(|e| e.into_inner());
            self.shared.wake.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

/// A buffer shared by the tasks of one region. Each task must use only ranges no other
/// task of the same region touches.
pub struct SharedMut<T>(*mut T, usize);

// SAFETY: the pointer is only dereferenced through `range`, whose contract keeps concurrent
// tasks on disjoint elements; `T: Send` lets those elements move between threads.
unsafe impl<T: Send> Sync for SharedMut<T> {}

impl<T> SharedMut<T> {
    pub fn new(s: &mut [T]) -> Self {
        SharedMut(s.as_mut_ptr(), s.len())
    }

    /// `&mut` to `lo..hi`.
    ///
    /// # Safety
    /// No other live reference to any element in `lo..hi` may exist while the result is used.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn range(&self, lo: usize, hi: usize) -> &mut [T] {
        assert!(lo <= hi && hi <= self.1, "SharedMut range out of bounds");
        // SAFETY: bounds were checked above; disjointness is the caller's contract.
        unsafe { std::slice::from_raw_parts_mut(self.0.add(lo), hi - lo) }
    }
}
