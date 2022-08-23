from argparse import ArgumentParser
from enum import Enum
import sys
from timeit import default_timer as timer


def subprocess_bench(instances: int):
    import subprocess
    processes = []
    cmd = "python -c 'import torch; print(torch.__version__)'"
    start = timer()
    for _ in range(instances):
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        processes.append(p)
    for p in processes:
        p.wait()
    end = timer()
    return end - start


def process_bench(instances: int):
    from multiprocessing import Process, Barrier

    def nop(barrier: Barrier):
        import torch
        barrier.wait()
        return torch.__version__

    processes = []
    barrier = Barrier(instances)
    start = timer()
    for _ in range(instances):
        p = Process(target=nop, args=(barrier,))
        processes.append(p)
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    return timer() - start


def torch_process_bench(instances: int):
    from torch.multiprocessing import Process, Barrier

    def nop(barrier: Barrier):
        import torch
        barrier.wait()
        return torch.__version__

    processes = []
    barrier = Barrier(instances)
    start = timer()
    for _ in range(instances):
        p = Process(target=nop, args=(barrier,))
        processes.append(p)
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    return timer() - start


def fork_process_bench(instances: int):
    import torch
    import os
    start = timer()
    children = []
    for _ in range(instances):
        child_pid = os.fork()
        if child_pid == 0:
            version = torch.__version__
            sys.exit(0)
        else:
            children.append(child_pid)
    for child in children:
        os.waitpid(child, 0)
    return timer() - start


def shmem_process_bench(instances: int):
    pass


class RunType(Enum):
    """
    Enum for the run
    """
    subprocess = 1
    process = 2
    fork = 3
    shmem = 4
    torch = 5


if __name__ == "__main__":
    args = ArgumentParser(description="Run the plot script")
    args.add_argument("-b", "--bench", type=str, default=RunType.subprocess.name,
                      help="Script supports the following plot types {}".format([type.name for type in RunType]))

    if len(sys.argv) < 2:
        args.print_help()
        sys.exit(1)
    input = args.parse_args()

    instances = [2 ** i for i in range(0, 11)]
    if input.bench == RunType.subprocess.name:
        for i in instances:
            print(f"{i} subprocesses: {subprocess_bench(i) * 1000} ms")
    elif input.bench == RunType.process.name:
        for i in instances:
            print(f"{i} processes: {process_bench(i) * 1000} ms")
    elif input.bench == RunType.fork.name:
        for i in instances:
            print(f"{i} shmem processes: {fork_process_bench(i) * 1000} ms")
    elif input.bench == RunType.shmem.name:
        for i in instances:
            print(f"{i} fork processes: {shmem_process_bench(i) * 1000} ms")
    elif input.bench == RunType.torch.name:
        for i in instances:
            print(f"{i} torch processes: {torch_process_bench(i) * 1000} ms")
    else:
        print("Error: Unrecognized bench type '{0}'".format(input.bench))
        sys.exit(1)
