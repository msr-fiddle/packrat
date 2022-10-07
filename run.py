"""
Script to run the benchmark for variable number of threads.
The script not only decides the number of threads, but also
pins the threads to the cores to avoid the performance impact
of thread migration.
"""
#!/usr/bin/python3

from pathlib import Path
import time
import csv
import logging
import math
from multiprocessing import Process, set_start_method
import os
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
import psutil

from utils.topology import CPUInfo, read_cpu_freq
from utils.optimizer import Optimizer
from benchmarks.config import Benchmark, MemoryAllocator, ModelSource, Optimizations, RunType, Config, ThreadMapping, ThreadPinning
from benchmarks.cache import store
from benchmarks.resnet import ResnetBench
from benchmarks.inception import InceptionBench
from benchmarks.gpt2 import GptBench
from benchmarks.bert import BertBench
from benchmarks.bench import Bench


def parse_args():
    args = ArgumentParser(description="Run the benchmark",
                          formatter_class=ArgumentDefaultsHelpFormatter)
    args.add_argument("--benchmark", type=str, default=Benchmark.resnet.name,
                      choices=[bench.name for bench in Benchmark],
                      help="Pick a benchmark to run.")
    args.add_argument("--run-type", type=str, default=RunType.manual.name,
                      choices=[run.name for run in RunType],
                      help="Pick a run type.")
    args.add_argument("--optimization", type=str, default=Optimizations.script.name,
                      choices=[opt.name for opt in Optimizations],
                      help="Pick an optimization to use.")
    args.add_argument("--mapping", type=str, default=ThreadMapping.sequential.name,
                      choices=[mapping.name for mapping in ThreadMapping],
                      help="Pick a thread mapping.")
    args.add_argument("--batch-size", type=int, default=1,
                      help="The batch size")
    args.add_argument("--iterations", type=int, default=100,
                      help="The number of iterations")
    args.add_argument("--interop-threads", type=int, default=1,
                      help="The number of interop threads")
    args.add_argument("--intraop-threads", type=int, default=1,
                      help="The number of intraop threads")
    args.add_argument("--core-list", type=int, nargs='+', default=[],
                      help="The list of cores to pin to")
    args.add_argument("--flops", type=int, default=0,
                      help="The number of flops")
    args.add_argument("--instance_id", type=int, default=1,
                      help="The number of instances")
    args.add_argument("--pinning", type=str, default=ThreadPinning.numactl.name,
                      choices=[pin.name for pin in ThreadPinning],
                      help="Pick a thread pinning scheme.")
    args.add_argument("--source", type=str, default=ModelSource.torch.name,
                      choices=[source.name for source in ModelSource],
                      help="Pick a model source.")
    args.add_argument("--storename", type=str, default=None,
                      help="The name of the store (handled internally)")
    args.add_argument("--log", type=str, default="INFO",
                      choices=["debug", "info",
                               "warning", "error", "critical"],
                      help="Log level")
    args.add_argument("--allocator", type=str, default=MemoryAllocator.default.name,
                      choices=[alloc.name for alloc in MemoryAllocator],
                      help="Pick a memory allocator.")
    return args.parse_args()


def static_checks(args: Namespace):
    if args.source == ModelSource.cache.name:
        if args.storename is None:
            raise Exception("Cache store name must be provided")
        if args.optimization == Optimizations.script.name:
            raise Exception(
                "Cache store does not support torchscript optimization yet!")


def set_env(env, pinning: ThreadPinning):
    """
    Set the environment variable
    """
    env["KMP_BLOCKTIME"] = "1"

    if pinning == ThreadPinning.omp:
        env["OMP_SCHEDULE"] = "STATIC"
        env["OMP_PROC_BIND"] = "CLOSE"
        env["KMP_AFFINITY"] = "granularity=fine"
    else:
        env.pop("OMP_SCHEDULE", None)
        env.pop("OMP_PROC_BIND", None)
        env.pop("KMP_AFFINITY", None)
    return env


def set_memory_allocator(env, allocator: MemoryAllocator):
    """
    Set the memory allocator
    """
    tcmalloc_lib = Path("/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4")
    jemalloc_lib = Path("/usr/lib/x86_64-linux-gnu/libjemalloc.so.2")

    if allocator == MemoryAllocator.default:
        env.pop('LD_PRELOAD', None)
        env.pop('MALLOC_CONF', None)
    elif allocator == MemoryAllocator.tcmalloc and tcmalloc_lib.exists():
        env["LD_PRELOAD"] = tcmalloc_lib.as_posix()
    elif allocator == MemoryAllocator.jemalloc and jemalloc_lib.exists():
        env["MALLOC_CONF"]="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
        env["LD_PRELOAD"] = jemalloc_lib.as_posix()
    else:
        raise Exception("Unable to set memory allocator")
    return env


def update_config(config: Config, instance_id: int, thread_mapping: str, batch_size: int, intraop_num: int, core_list: list) -> Config:
    """
    Update the configuration based on the arguments
    """
    config.set_instance_id(instance_id)
    config.set_mapping(ThreadMapping[thread_mapping])
    config.set_batch_size(batch_size)
    config.set_interop_threads(1)
    config.set_intraop_threads(intraop_num)
    config.set_core_list(core_list)
    return config


def run_single_instance(config: Config):
    """
    Run the benchmark with the given parameters
    """

    logging.debug("Running the benchmark with %s", config)
    if config.run_type == RunType.manual and config.pinnning == ThreadPinning.numactl:
        os.sched_setaffinity(0, config.core_list)

    bench: Bench = None
    if config.benchmark == Benchmark.resnet:
        bench = ResnetBench()
    elif config.benchmark == Benchmark.inception:
        bench = InceptionBench()
    elif config.benchmark == Benchmark.gpt2:
        bench = GptBench()
    elif config.benchmark == Benchmark.bert:
        bench = BertBench()
    else:
        raise Exception("Unknown benchmark")

    bench.latencies = [None] * (config.iterations)
    bench.run(config)
    bench.report(config)


def freq_measurement(processes):
    """
    Measure the frequency of the cores
    """
    os.sched_setaffinity(0, {1})
    file = open(f"freqs_{processes}.csv", "a+")
    writer = csv.writer(file, delimiter=",")
    writer.writerow(["Instances", "Iterations", "Freq"])

    core_list = [2*p for p in range(processes)]
    logging.debug("Starting freq measurement")
    count = 0
    while True:
        core_freqs = []
        for core in core_list:
            core_freqs.append(read_cpu_freq(core))
        writer.writerow([processes, count, sum(core_freqs)/len(core_freqs)])
        file.flush()
        count += 1
        time.sleep(1)


def lower_power_of_two(cores: int) -> int:
    return 2**(math.floor(math.log(cores, 2)))


def set_log_level(level: str):
    """
    Set the log level
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.getLogger().setLevel(numeric_level)


if __name__ == '__main__':
    set_start_method('spawn', force=True)
    arguments = parse_args()
    set_log_level(arguments.log)

    if arguments.source == ModelSource.cache.name:
        cache = store.Cache()
        arguments.storename = cache.storename

    # static_checks should be called after setting the storename
    static_checks(arguments)

    topology = CPUInfo()
    core_count = int(psutil.cpu_count(logical=False) /
                     len(topology.get_sockets()))
    proclist = topology.allocate_cores(
        "socket", core_count, arguments.mapping)

    set_env(os.environ, arguments.pinning)
    set_memory_allocator(os.environ, MemoryAllocator[arguments.allocator])

    if arguments.instance_id == 1:
        logging.debug("Running single instance")
        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            for intraop in range(1, core_count + 1):
                config = Config(arguments)
                config = update_config(config, arguments.instance_id, arguments.mapping,
                                       batch_size, intraop, proclist[0:intraop])

                process = Process(
                    target=run_single_instance, args=(config,))
                process.start()
                process.join()
    else:
        logging.debug("Running multiple instances")
        core_count = lower_power_of_two(core_count)
        optimizer = Optimizer()

        for batch_size in [8, 16, 32, 64, 128, 256, 512, 1024]:
            # ====================== 1 instance ======================
            # Run single instance baseline <core_count, batch_size>
            # ========================================================
            config = Config(arguments)
            config = update_config(config, 0, arguments.mapping,
                                   batch_size, core_count, proclist[0:core_count])
            process = Process(
                target=run_single_instance, args=(config,))
            process.start()
            process.join()

            # ================ Multiple instances ====================
            # Run the multi-instance optimal configuration
            # ========================================================
            optimal_instances = []
            optimizer.solution(core_count, batch_size,
                               arguments.benchmark, optimal_instances)
            total_instances = len(optimal_instances)
            instances, cmd, config = [], [], []
            starting_core = 0

            for i in range(total_instances):
                cores_per_instance = optimal_instances[i][0]
                batch_per_instance = optimal_instances[i][1]

                instance_config = Config(arguments)
                instance_config = update_config(instance_config, i + 1, arguments.mapping, batch_per_instance,
                                                cores_per_instance, proclist[starting_core:starting_core + cores_per_instance])
                config.append(instance_config)
                starting_core += cores_per_instance

                p = Process(
                    target=run_single_instance, args=(config[i],))
                instances.append(p)

            for instance in instances:
                instance.start()
            for instance in instances:
                instance.join()
