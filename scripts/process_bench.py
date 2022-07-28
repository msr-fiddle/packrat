from concurrent.futures import process
import subprocess
from timeit import default_timer as timer


def process_bench(instances: int):
    processes = []
    start = timer()
    for _ in range(instances):
        p = subprocess.Popen(["python3", "--version"],
                             stdout=subprocess.DEVNULL)
        processes.append(p)
    for p in processes:
        p.wait()
    end = timer()
    return end - start


if __name__ == "__main__":
    for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        print(f"{i} processes: {process_bench(i) * 1000} ms")
