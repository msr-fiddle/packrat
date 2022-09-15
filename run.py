"""
Script to run the benchmark for variable number of threads.
The script not only decides the number of threads, but also
pins the threads to the cores to avoid the performance impact
of thread migration.
"""
#!/usr/bin/python3

import logging
import os
import subprocess
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
import psutil
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


def update_config(config: Config, instance_id: int, thread_mapping: ThreadMapping, batch_size: int, intraop_num: int, core_list: list) -> Config:
    """
    Update the configuration based on the arguments
    """
    config.set_instance_id(instance_id)
    config.set_mapping(thread_mapping)
    config.set_batch_size(batch_size)
    config.set_interop_threads(1)
    config.set_intraop_threads(intraop_num)
    config.set_core_list(core_list)
    return config


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
        for intraop in range(1, core_count + 1):
            proclist = topology.allocate_cores(
                "socket", intraop, config.mapping.name)

            # Update the configuration
            config = update_config(config, args.instance_id, ThreadMapping[args.mapping],
                                   batch_size, intraop, proclist)

            # Run the benchmark
            run_with_parameters(config)


def run_multi_instances(args: Namespace):
    assert args.instance_id != 1, "Instance id must be greater than 1"

    def lower_power_of_two(x):
        import math
        return 2**(math.floor(math.log(x, 2)))

    topology = CPUInfo()
    core_count = lower_power_of_two(int(psutil.cpu_count(logical=False) /
                                        len(topology.get_sockets())))
    corelist = topology.allocate_cores(
        "socket", core_count, args.mapping)

    optimizer = Optimizer()
    for batch_size in [8, 16, 32, 64, 128, 256, 512, 1024]:
        single_instance_config = Config(args)
        single_instance_config = update_config(single_instance_config, 1, ThreadMapping[args.mapping],
                                               batch_size, core_count, corelist)
        run_with_parameters(single_instance_config)

        optimal_instances = []
        optimizer.solution(core_count, batch_size,
                           args.benchmark, optimal_instances)

        total_instances = len(optimal_instances)
        instances, cmd, config = [], [], []
        starting_core = 0
        for i in range(total_instances):
            cores_per_instance = optimal_instances[i][0]
            batch_per_instance = optimal_instances[i][1]

            instance_config = Config(args)
            instance_config = update_config(instance_config, i + 1, ThreadMapping[args.mapping], batch_per_instance,
                                            cores_per_instance, corelist[starting_core:starting_core + cores_per_instance])
            config.append(instance_config)
            starting_core += cores_per_instance

            cmd.append([
                "numactl", "-C {}".format(",".join(str(x)
                                                   for x in config[i].core_list)),
                "python3"
            ])
            cmd[i].append(f"./benchmarks/{config[i].benchmark.name}.py")
            cmd[i].append(repr(config[i]))

            instances.append(subprocess.Popen(
                cmd[i], env=os.environ.copy(), stdout=subprocess.DEVNULL))

        for instance in instances:
            instance.wait()


if __name__ == '__main__':
    arguments = parse_args()

    if arguments.source == ModelSource.cache.name:
        cache = store.Cache()
        arguments.storename = cache.storename

    # static_checks should be called after setting the storename
    static_checks(arguments)

    if arguments.instance_id == 1:
        run(arguments)
    else:
        run_multi_instances(arguments)
