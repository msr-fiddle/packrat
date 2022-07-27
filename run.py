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
from argparse import ArgumentParser, Namespace
import psutil
from utils.topology import CPUInfo

from benchmarks.config import Benchmark, Optimizations, RunType, Config, ThreadMapping


def parse_args():
    args = ArgumentParser(description="Run the benchmark")
    args.add_argument("--benchmark", type=str, default=Benchmark.resnet.name,
                      help="The benchmark to run")
    args.add_argument("--run-type", type=str, default=RunType.manual.name,
                      help="The run type")
    args.add_argument("--optimization", type=str, default=Optimizations.none.name,
                      help="The optimization to use")
    args.add_argument("--mapping", type=str, default=ThreadMapping.sequential.name,
                      help="The thread mapping")
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
    return args.parse_args()


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
        # Static division of work among threads
        set_env(myenv, "OMP_SCHEDULE", "STATIC")
        # Schedule the thread near to the parent thread
        set_env(myenv, "OMP_PROC_BIND", "CLOSE")
        set_env(myenv, "KMP_AFFINITY",
                f"granularity=fine,proclist={proclist},explicit")

        cmd = [
            "numactl", "-C {}".format(",".join(str(x)
                                               for x in proclist)),
            "python3"
        ]

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


if __name__ == '__main__':
    arguments = parse_args()
    run(arguments)
