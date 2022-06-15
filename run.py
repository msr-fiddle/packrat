"""
Script to run the benchmark for variable number of threads.
The script not only decides the number of threads, but also
pins the threads to the cores to avoid the performance impact
of thread migration.
"""
#!/usr/bin/python3

from enum import Enum
import os
import subprocess
import sys
import psutil
import utils.topology as topology


class Benchmark(Enum):
    """
    Enum for the benchmark
    """
    resnet = 1


class RunType(Enum):
    """
    Enum for the run
    """
    default = 1
    manual = 2


def set_env(env, env_name: str, env_value: str):
    """
    Set the environment variable
    """
    env[env_name] = env_value


def run(bench: Benchmark, r_type: RunType, batch_size: int):
    """
    Main function
    """

    core_count = psutil.cpu_count(logical=False)
    for cores in range(0, core_count + 1, 4):
        threads = max(cores, 1)
        proclist = topology.allocate_cores("socket", max(cores, 1))
        myenv = os.environ.copy()

        set_env(myenv, "KMP_BLOCKTIME", "1")
        set_env(myenv, "OMP_NUM_THREADS", str(threads))
        set_env(myenv, "MKL_NUM_THREADS", str(threads))

        if r_type == RunType.manual:
            # Static division of work among threads
            set_env(myenv, "OMP_SCHEDULE", "STATIC")
            # Schedule the thread near to the parent thread
            set_env(myenv, "OMP_PROC_BIND", "CLOSE")
            set_env(myenv, "KMP_AFFINITY",
                    "granularity=fine,proclist={},explicit".format(proclist))

            cmd = [
                "numactl", "-C {}".format(",".join(str(x) for x in proclist)),
                "python", "./benchmarks/{}.py".format(bench.name),
                "{}".format(r_type.name),
                "{}".format(batch_size)
            ]
        elif r_type == RunType.default:
            cmd = [
                "python", "./benchmarks/{}.py".format(bench.name),
                "{}".format(r_type.name),
                "{}".format(batch_size)
            ]
        ret = subprocess.check_call(cmd, env=myenv)
        assert ret == 0


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Error: Too few arguments")
        print(
            "Usage: {0} [benchmark] [default/manual] [batch_size]".format(sys.argv[0]))
        sys.exit(1)

    BENCH = Benchmark[sys.argv[1]]
    TYPE = RunType[sys.argv[2]]
    batch_size = sys.argv[3]

    run(BENCH, TYPE, batch_size)
