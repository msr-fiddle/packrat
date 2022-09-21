use log::{info, warn};
use nix::sched::{sched_setaffinity, CpuSet};
use nix::unistd::Pid;
use std::time::Instant;
use x86::cpuid::CpuId;

#[allow(dead_code)]
extern "C" {
    fn kernel_x86_avx512f_fp32(a: i64);
    fn kernel_x86_avx512f_fp64(a: i64);
}

fn run_bench(tid: usize) {
    let mut cpu_set = CpuSet::new();
    cpu_set.set(tid).unwrap();
    sched_setaffinity(Pid::from_raw(0), &cpu_set).unwrap();

    let iterations = 0x200000000;
    let instructions_per_loop = 10 * 16 * 2;

    let start = Instant::now();
    unsafe {
        kernel_x86_avx512f_fp32(iterations as i64);
    }
    let time_used = start.elapsed().as_nanos();
    let perf = (iterations * instructions_per_loop) / time_used;
    info!("{} GFlops", perf);
}

fn main() {
    env_logger::init();
    let cpuid = CpuId::new();
    let features = cpuid.get_extended_feature_info().unwrap();
    let has_avx512f = features.has_avx512f();

    match has_avx512f {
        true => {
            unsafe {
                let start = Instant::now();
                kernel_x86_avx512f_fp32(100);
                let time_used = start.elapsed().as_nanos();
                let divisor = (100 * 10) as f64 / 2.7;
                info!(
                    "Cycles to execute one FMA {} cycles",
                    time_used / divisor as u128
                );
            }
            let mut threads = Vec::with_capacity(6);
            for tid in 0..6 {
                threads.push(std::thread::spawn(move || {
                    run_bench(tid);
                }));
            }

            for t in threads {
                t.join().unwrap();
            }
        }
        false => warn!("Does not have AVX512f"),
    }
}
