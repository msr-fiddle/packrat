"""
Script to run the benchmark for variable number of threads.
The script not only decides the number of threads, but also
pins the threads to the cores to avoid the performance impact
of thread migration.
"""
#!/usr/bin/python3

import multiprocessing
import logging
import os
import subprocess
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
import psutil
from benchmarks.resnet import ResnetBench
from utils.topology import CPUInfo
from utils.optimizer import Optimizer

from benchmarks.config import Benchmark, ModelSource, Optimizations, RunType, Config, ThreadMapping, ThreadPinning
from benchmarks.cache import store


def parse_args():
    args = ArgumentParser(description="Run the benchmark",
                          formatter_class=ArgumentDefaultsHelpFormatter)
    args.add_argument("--benchmark", type=str, default=Benchmark.resnet.name,
                      help=f"Pick a benchmark to run {[bench.name for bench in Benchmark]}")
    args.add_argument("--run-type", type=str, default=RunType.manual.name,
                      help=f"Pick a run type {[run.name for run in RunType]}")
    args.add_argument("--optimization", type=str, default=Optimizations.script.name,
                      help=f"Pick an optimization to use {[opt.name for opt in Optimizations]}")
    args.add_argument("--mapping", type=str, default=ThreadMapping.sequential.name,
                      help=f"Pick a thread mapping {[map.name for map in ThreadMapping]}")
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
                      help=f"Pick a thread pinning scheme {[pin.name for pin in ThreadPinning]}")
    args.add_argument("--source", type=str, default=ModelSource.torch.name,
                      help=f"Pick a model source {[src.name for src in ModelSource]}")
    args.add_argument("--storename", type=str, default=None,
                      help="The name of the store (handled internally)")
    return args.parse_args()


def static_checks(args: Namespace):
    if args.source == ModelSource.cache.name:
        if args.storename is None:
            raise Exception("Cache store name must be provided")
        if args.optimization == Optimizations.script.name:
            raise Exception(
                "Cache store does not support torchscript optimization yet!")


def set_env(env, env_name: str, env_value: str):
    """
    Set the environment variable
    """
    env[env_name] = env_value


def run_with_parameters(config: Config):
    """
    Run the benchmark with the given parameters
    """
    myenv = os.environ.copy()
    set_env(myenv, "KMP_BLOCKTIME", "1")

    cmd = []
    if config.run_type == RunType.manual:
        proclist = config.core_list
        cmd = []

        if config.pinnning == ThreadPinning.numactl:
            cmd = ["numactl"]
            cmd.append("-C {}".format(",".join(str(x)
                                               for x in proclist)))
        elif config.pinnning == ThreadPinning.omp:
            # Static division of work among threads
            set_env(myenv, "OMP_SCHEDULE", "STATIC")
            # Schedule the thread near to the parent thread
            set_env(myenv, "OMP_PROC_BIND", "CLOSE")
            set_env(myenv, "KMP_AFFINITY",
                    f"granularity=fine,proclist={proclist},explicit")
        else:
            cmd = []

    cmd.append("python3")
    cmd.append(f"./benchmarks/{config.benchmark.name}.py")
    cmd.append(repr(config))
    logging.info("Running: %s", " ".join(cmd))
    ret = subprocess.check_output(cmd, env=myenv)
    if config.intraop_threads == 1:
        flops = int(ret.decode("utf-8").split("\n")[-2])
        config.set_flops(flops)


def run(args: Namespace):
    """
    Main function
    """
    config = Config(args)

    topology = CPUInfo()
    core_count = int(psutil.cpu_count(logical=False) /
                     len(topology.get_sockets()))
    for batch_size in [1, 8, 16, 32]:
        for interop in [1]:
            for intraop in range(1, core_count + 1):
                proclist = topology.allocate_cores(
                    "socket", intraop, config.mapping.name)

                # Set the configuration
                config.set_mapping(config.mapping)
                config.set_core_list(proclist)
                config.set_interop_threads(interop)
                config.set_intraop_threads(intraop)
                config.set_batch_size(batch_size)

                # Run the benchmark
                run_with_parameters(config)


def start_bench(config):
    os.sched_setaffinity(0, config.core_list)
    config.benchmark = Benchmark.resnet
    bench = ResnetBench()
    bench.latencies = [None] * config.iterations
    bench.run(config)
    bench.report(config)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    args = parse_args()

    if args.source == ModelSource.cache.name:
        cache = store.Cache()
        args.storename = cache.storename

    # static_checks should be called after setting the storename
    static_checks(args)

    assert args.instance_id != 1, "Instance id must be greater than 1"
    set_env(os.environ, "KMP_BLOCKTIME", "1")
    set_env(os.environ, "OMP_SCHEDULE", "STATIC")
    set_env(os.environ, "OMP_PROC_BIND", "CLOSE")
    set_env(os.environ, "KMP_AFFINITY",
            f"granularity=fine,explicit")

    def lower_power_of_two(x):
        import math
        return 2**(math.floor(math.log(x, 2)))

    topology = CPUInfo()
    core_count = lower_power_of_two(int(psutil.cpu_count(logical=False) /
                                        len(topology.get_sockets())))
    corelist = topology.allocate_cores(
        "socket", core_count, args.mapping)

    for i in [16]:
        optimal_instances = [(1, 16) for ins in range(i)]

        total_instances = len(optimal_instances)
        instances, cmd, config = [], [], []
        starting_core = 0
        for i in range(total_instances):
            cores_per_instance = optimal_instances[i][0]
            batch_per_instance = optimal_instances[i][1]

            config.append(Config(args))
            config[i].set_instance_id(i + 1)
            config[i].set_batch_size(batch_per_instance)
            config[i].set_intraop_threads(cores_per_instance)
            config[i].set_core_list(
                corelist[starting_core:starting_core + cores_per_instance])
            starting_core += cores_per_instance

            instances.append(multiprocessing.Process(
                target=start_bench, args=(config[i],)))

        for instance in instances:
            instance.start()
        for instance in instances:
            instance.join()
