"""
Script to run the benchmark for variable number of threads.
The script not only decides the number of threads, but also
pins the threads to the cores to avoid the performance impact
of thread migration.
"""
#!/usr/bin/python3

import os
from multiprocessing import Process
import subprocess
import psutil
import utils.topology as topology

import benchmarks.config as config
from benchmarks.config import RunType, Config


def set_env(env, env_name: str, env_value: str):
    """
    Set the environment variable
    """
    env[env_name] = env_value


def run():
    """
    Main function
    """

    core_count = psutil.cpu_count(logical=False)
    for cores in range(0, core_count + 1, 4):
        threads = max(cores, 1)
        proclist = topology.allocate_cores("socket", max(cores, 1))
        object.__setattr__(config.cur_config, 'core_list', proclist)
        myenv = os.environ.copy()

        set_env(myenv, "KMP_BLOCKTIME", "1")
        set_env(myenv, "OMP_NUM_THREADS", str(threads))
        set_env(myenv, "MKL_NUM_THREADS", str(threads))

        if config.cur_config.run_type == RunType.manual:
            # Static division of work among threads
            set_env(myenv, "OMP_SCHEDULE", "STATIC")
            # Schedule the thread near to the parent thread
            set_env(myenv, "OMP_PROC_BIND", "CLOSE")
            set_env(myenv, "KMP_AFFINITY",
                    "granularity=fine,proclist={},explicit".format(proclist))

            cmd = [
                "numactl", "-C {}".format(",".join(str(x) for x in proclist)),
                "python", "./benchmarks/{}.py".format(
                    config.cur_config.benchmark.name)
            ]
        elif config.cur_config.run_type == RunType.default:
            cmd = [
                "python", "./benchmarks/{}.py".format(
                    config.cur_config.benchmark.name)
            ]
        ret = subprocess.check_call(cmd, env=myenv)
        assert ret == 0


if __name__ == '__main__':
    run()
