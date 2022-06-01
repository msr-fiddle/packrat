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


def set_env(env, env_name: str, env_value: str):
    """
    Set the environment variable
    """
    env[env_name] = env_value
    #os.system('export {0}="{1}"'.format(env_name, env_value))


def run():
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
        set_env(myenv, "KMP_AFFINITY",
                "granularity=fine,proclist={},explicit".format(proclist))

        cmd = [
            "numactl", "-C {}".format(",".join(str(x) for x in proclist)),
            "python", "./benchmarks/resnet.py"
        ]
        ret = subprocess.check_call(cmd, env=myenv)
        assert ret == 0


if __name__ == '__main__':
    run()
