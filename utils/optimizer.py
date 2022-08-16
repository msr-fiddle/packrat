"""
The problem we have is the 2D analogue of this knapsack problem.

opt[t, b] is the total latency of processing b inputs with t threads.
Then you can compute opt[t, b] just looking at opt[t', b'] where t' < t and b' < b.

opt[t, b] can be computed as follows:
opt[t, b] = min_{t', b'} (max(opt(t-t', b-b'), latency[t', b'])

latency[t',b'] = latency of processing a batch with b' inputs with t' threads (from profiles)
"""


from io import BytesIO
import urllib.request
import pickle
import pandas as pd


def get_data(model: str, benchmark: str):
    """
    Get the data from the benchmark.
    :param model: model name
    :param benchmark: benchmark type latency or throughput
    :return: dataframe
    """
    base = "https://msr-fiddle.github.io/naf"
    url = f"{base}/{model}/skylake2x/large-batches/{model}_{benchmark}.csv"
    print(url)
    print(f" > Fetching {url} ...", end='')
    try:
        with urllib.request.urlopen(url) as response:
            print(f"OK ({response.getcode()}).")
            data = pd.read_csv(BytesIO(response.read()))
            return data
    except urllib.error.HTTPError as error:
        print(f"FAILED ({error.code}).")
        return None


def format_data(data):
    """
    Format the data into a matrix.
    :param data: dataframe
    :return: matrix
    """
    data = data.loc[:, ["topology", "intraop_threads",
                        "batch_size", "latency(avg)"]]

    latencies = {}
    for thread in data.intraop_threads.unique():
        batch_latencies = {}
        for batch in data.batch_size.unique():
            latency = data.loc[(data["intraop_threads"] == thread) &
                               (data["batch_size"] == batch) &
                               (data["topology"] == "sequential")]["latency(avg)"].values[0]
            batch_latencies[batch] = latency

        latencies[thread] = batch_latencies

    return latencies


class Optimizer:
    """
    Configuration optimizer for the benchmark.
    """

    def __init__(self):
        """Initialize the optimizer."""
        self.opt = {}
        self.latencies = {}
        self.workloads = ["resnet", "inception", "gpt2", "bert"]
        for workload in self.workloads:
            self.latencies[workload] = format_data(
                get_data(workload, "latency"))
            try:
                with open(f'utils/{workload}.pickle', 'rb') as file:
                    self.opt[workload] = pickle.load(file)
            except FileNotFoundError:
                self.opt[workload] = self.min_latency(workload)
                with open(f'{workload}.pickle', 'wb') as file:
                    pickle.dump(self.opt[workload], file)

    def min_latency(self, workload):
        """
        Given a set of threads and batches, find the minimum latency.
        :param max_threads: maximum number of threads
        :param max_batch: maximum number of batches
        :param latency: matrix of latencies
        :return: optimial latency matrix
        """

        latency = self.latencies[workload]
        max_threads = max(latency.keys())
        max_batch = max(latency[max_threads].keys())

        opt = [[float('inf') for _ in range(max_batch + 1)]
               for _ in range(max_threads + 1)]

        for thread in range(1, max_threads + 1):
            for batch in range(1, max_batch + 1):
                for t_i in range(1, thread + 1):
                    for b_i in range(1, batch + 1):
                        if t_i in latency and b_i in latency[t_i]:
                            opt[thread][batch] = min(opt[thread][batch],
                                                     max(opt[thread - t_i][batch - b_i],
                                                         latency[t_i][b_i])
                                                     )
                        if thread in latency and batch in latency[thread]:
                            opt[thread][batch] = min(
                                opt[thread][batch], latency[thread][batch])

        return opt

    def solution(self, threads, batch, workload, result):
        if threads <= 0 or batch <= 0:
            return 0

        latency = self.latencies[workload]
        opt = self.opt[workload]
        max_threads, max_batch = 0, 0

        for t_i in range(threads, 0, -1):
            for b_i in range(batch, 0, -1):
                if t_i in latency and b_i in latency[t_i]:
                    if opt[threads][batch] == opt[t_i][b_i] == latency[t_i][b_i]:
                        max_threads, max_batch = threads - t_i, batch - b_i
                        result.append((t_i, b_i))
                        break

        return self.solution(max_threads, max_batch, workload, result)


if __name__ == "__main__":
    optimizer = Optimizer()
    instances = []
    optimizer.solution(18, 1024, "resnet", instances)
    print(instances)
