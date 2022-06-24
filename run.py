"""
Script to run the benchmark for variable number of threads.
The script not only decides the number of threads, but also
pins the threads to the cores to avoid the performance impact
of thread migration.
"""
#!/usr/bin/python3

import os
import subprocess
import psutil
import utils.topology as topology

from benchmarks.config import RunType, Config


def set_env(env, env_name: str, env_value: str):
    """
    Set the environment variable
    """
    env[env_name] = env_value


def run(config: Config):
    """
    Main function
    """

    core_count = psutil.cpu_count(logical=False)
    batch_size = 1
    while batch_size < core_count:
        for cores in range(0, core_count + 1, 4):
            threads = max(cores, 1)
            proclist = topology.allocate_cores("socket", max(cores, 1))
            config.set_core_list(proclist)
            config.set_batch_size(batch_size)
            myenv = os.environ.copy()

            set_env(myenv, "KMP_BLOCKTIME", "1")
            set_env(myenv, "OMP_NUM_THREADS", str(threads))
            set_env(myenv, "MKL_NUM_THREADS", str(threads))

            if config.run_type == RunType.manual:
                # Static division of work among threads
                set_env(myenv, "OMP_SCHEDULE", "STATIC")
                # Schedule the thread near to the parent thread
                set_env(myenv, "OMP_PROC_BIND", "CLOSE")
                set_env(myenv, "KMP_AFFINITY",
                        "granularity=fine,proclist={},explicit".format(proclist))

                cmd = [
                    "numactl", "-C {}".format(",".join(str(x)
                                              for x in proclist)),
                    "python", "./benchmarks/{}.py".format(
                        config.benchmark.name)
                ]
            elif config.run_type == RunType.default:
                cmd = [
                    "python", "./benchmarks/{}.py".format(
                        config.benchmark.name)
                ]
            config.update()
            ret = subprocess.check_call(cmd, env=myenv)
            assert ret == 0
        batch_size *= 2


if __name__ == '__main__':
    config = Config()
    config.reinitialize()
    run(config)
