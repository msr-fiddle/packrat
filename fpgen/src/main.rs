use clap::{value_t, App, Arg};
use log::{info, warn};
use nix::sched::{sched_setaffinity, CpuSet};
use nix::unistd::Pid;
use std::time::Instant;
use x86::cpuid::CpuId;

const OPS_PER_LOOP: usize = 10;
const FMA_UNITS: usize = 2;
const SP_FLOPS_PER_FMA: usize = 16;
const DP_FLOPS_PER_FMA: usize = 8;

static SP_FLOPS_PER_ITERATION: usize = FMA_UNITS * SP_FLOPS_PER_FMA * OPS_PER_LOOP;
static DP_FLOPS_PER_ITERATION: usize = FMA_UNITS * DP_FLOPS_PER_FMA * OPS_PER_LOOP;

extern "C" {
    fn kernel_x86_avx512f_fp32(a: i64);
    fn kernel_x86_avx512f_fp64(a: i64);
}

fn run_kernel(run_sp: bool, iterations: i64) -> u128 {
    match run_sp {
        true => unsafe {
            let start = Instant::now();
            kernel_x86_avx512f_fp32(iterations);
            start.elapsed().as_nanos()
        },
        false => unsafe {
            let start = Instant::now();
            kernel_x86_avx512f_fp64(iterations);
            start.elapsed().as_nanos()
        },
    }
}

fn run_bench(run_sp: bool, core_id: usize, iterations: i64) {
    let mut cpu_set = CpuSet::new();
    cpu_set.set(core_id).unwrap();
    sched_setaffinity(Pid::from_raw(0), &cpu_set).unwrap();

    let time_used = run_kernel(run_sp, iterations);

    let perf = match run_sp {
        true => (iterations * SP_FLOPS_PER_ITERATION as i64) as u128 / time_used,
        false => (iterations * DP_FLOPS_PER_ITERATION as i64) as u128 / time_used,
    };

    info!("{} GFlops", perf);
}

fn warmup(run_sp: bool) {
    let warmup_iterations = 100;
    let cpu_freq = cpu_freq::get()[0].max.unwrap_or(2600.0) as f64 / 1e3;

    let total_time_ns = run_kernel(run_sp, warmup_iterations);
    let total_fmas = (warmup_iterations * OPS_PER_LOOP as i64) as f64;
    let cycles_per_fma = (total_time_ns as f64 * cpu_freq) / total_fmas;
    info!("Cycles to execute one FMA {} cycles", cycles_per_fma);
}

fn setup(cores: usize, precision: String, iterations: i64) {
    let cpuid = CpuId::new();
    let features = cpuid.get_extended_feature_info().unwrap();
    let has_avx512f = features.has_avx512f();
    let run_sp = match precision.as_str() {
        "single" => true,
        "double" => false,
        _ => panic!("Invalid precision"),
    };

    match has_avx512f {
        true => {
            warmup(run_sp);
            let mut threads = Vec::with_capacity(cores);
            for core_id in 0..cores {
                threads.push(std::thread::spawn(move || {
                    run_bench(run_sp, core_id, iterations);
                }));
            }

            for t in threads {
                t.join().unwrap();
            }
        }
        false => warn!("Does not have AVX512f"),
    }
}

fn main() {
    env_logger::init();
    let args = std::env::args();
    let matches = App::new("FPGEN")
        .about("Run the FPGEN benchmark on the given cores")
        .arg(
            Arg::new("cores")
                .short('c')
                .long("cores")
                .takes_value(true)
                .default_value("1")
                .help("Set the number of threads to use for benchmark!"),
        )
        .arg(
            Arg::new("precision")
                .short('p')
                .long("precision")
                .takes_value(true)
                .possible_values(&vec!["single", "double"])
                .default_value("single")
                .help("Set the name of the component"),
        )
        .arg(
            Arg::new("iterations")
                .short('i')
                .long("iterations")
                .takes_value(true)
                .default_value("1000000000")
                .help(
                    "Set the number of iterations to run the benchmark for. Defaults to 1000000000",
                ),
        )
        .get_matches_from(args);

    let cores = value_t!(matches, "cores", usize).unwrap_or_else(|e| e.exit());
    let precision = value_t!(matches, "precision", String).unwrap_or_else(|e| e.exit());
    let iterations = value_t!(matches, "iterations", i64).unwrap_or_else(|e| e.exit());

    setup(cores, precision, iterations);
}
