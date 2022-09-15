import os
import sys
from time import sleep
import csv


def read_cpu_freq(core_id: int):
    basedir = "/sys/devices/system/cpu"
    fname = f"cpu{core_id}/cpufreq/scaling_cur_freq"
    fpath = os.path.join(basedir, fname)
    with open(fpath, "rb") as f:
        data = f.read().decode("utf-8")
        return int(data)/1e6


def read_cpu_freqs(core_counts: int):
    freqs = []
    for i in [2*c for c in range(core_counts)]:
        freqs.append(read_cpu_freq(i))
    avg = sum(freqs)/len(freqs)
    return avg


def write_cpu_freqs(core_counts: int):
    writer = csv.writer(open("freqs.csv", "a+"), delimiter=",")
    for i in range(100):
        freq = read_cpu_freqs(core_counts)
        print(freq)
        writer.writerow([i, freq])
        sleep(1)


if __name__ == "__main__":
    core_counts = int(sys.argv[1])
    os.sched_setaffinity(0, [1])
    write_cpu_freqs(core_counts)
