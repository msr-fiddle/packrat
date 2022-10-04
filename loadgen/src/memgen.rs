#![feature(stdsimd)]

use clap::{value_t, App, Arg};
use log::{info, warn};
use nix::sched::{sched_setaffinity, CpuSet};
use nix::unistd::Pid;
use std::{
    alloc::{alloc, Layout},
    arch::x86_64::*,
};
use x86::cpuid::CpuId;

const ALLOCATE_PER_THREAD: usize = 16 * 1024 * 1024;
const ALIGNED: usize = 64;

fn bench(core_id: usize) {
    let mut cpu_set = CpuSet::new();
    cpu_set.set(core_id).unwrap();
    sched_setaffinity(Pid::from_raw(0), &cpu_set).unwrap();

    let layout =
        Layout::from_size_align(ALLOCATE_PER_THREAD, ALIGNED).expect("unable to create layout");
    let array = unsafe { alloc(layout) };
    let increment = 512;
    loop {
        let mut ptr = array as *mut u8;
        unsafe {
            let _zero = _mm512_setzero_si512();
            while ptr.add(increment) < array.add(ALLOCATE_PER_THREAD) {
                _mm_prefetch(ptr.add(increment) as *mut i8, _MM_HINT_T2);
                _mm512_store_si512(ptr as *mut i32, _zero);
                _mm512_store_si512(ptr.add(64) as *mut i32, _zero);
                _mm512_store_si512(ptr.add(128) as *mut i32, _zero);
                _mm512_store_si512(ptr.add(192) as *mut i32, _zero);
                _mm512_store_si512(ptr.add(256) as *mut i32, _zero);
                _mm512_store_si512(ptr.add(320) as *mut i32, _zero);
                _mm512_store_si512(ptr.add(384) as *mut i32, _zero);
                _mm512_store_si512(ptr.add(448) as *mut i32, _zero);

                ptr = ptr.add(increment);
            }
        }
    }
}

fn setup(cores: usize, skip: usize) {
    let cpuid = CpuId::new();
    let features = cpuid.get_extended_feature_info().unwrap();
    let has_avx512f = features.has_avx512f();

    let core_ids = allocate_cores(cores, skip);

    match has_avx512f {
        true => {
            let mut threads = Vec::with_capacity(cores);
            for core_id in core_ids {
                threads.push(std::thread::spawn(move || {
                    bench(core_id);
                }));
            }

            for t in threads {
                t.join().unwrap();
            }
        }
        false => warn!("Does not have AVX512f"),
    }
}

fn allocate_cores(how_many: usize, skip: usize) -> Vec<usize> {
    let topology = corealloc::topology::MachineTopology::new();
    let cpuinfo = topology.cpus_on_socket(0);
    let all_cpus = cpuinfo
        .iter()
        .map(|c| c.cpu as usize)
        .collect::<Vec<usize>>();
    let cores = all_cpus[skip..how_many + skip].to_vec();
    info!("Started benchmark on cores: {:?}", cores);
    cores
}

fn main() {
    env_logger::init();
    let args = std::env::args();
    let matches = App::new("MEMGen")
        .about("Run the MEMGen benchmark on the given cores")
        .arg(
            Arg::new("cores")
                .short('c')
                .long("cores")
                .takes_value(true)
                .default_value("3")
                .help("Set the number of threads to use for benchmark!"),
        )
        .arg(
            Arg::new("skip")
                .short('s')
                .long("skip")
                .takes_value(true)
                .default_value("1")
                .help("Set the number of threads to skip!"),
        )
        .get_matches_from(args);

    let cores = value_t!(matches, "cores", usize).unwrap_or_else(|e| e.exit());
    let skip = value_t!(matches, "skip", usize).unwrap_or_else(|e| e.exit());

    setup(cores, skip);
}
