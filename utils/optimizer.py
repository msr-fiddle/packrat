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
import numpy as np
import pandas as pd


class Optimizer:
    """
    Configuration optimizer for the benchmark.
    """

    def get_data(self, model: str, benchmark: str):
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

    def format_data(self, data):
        """
        Format the data into a matrix.
        :param data: dataframe
        :return: matrix
        """
        data = data.loc[:, ["intraop_threads", "batch_size", "latency(avg)"]]

        latencies = {}
        for thread in data.intraop_threads.unique():
            batch_latencies = {}
            for batch in data.batch_size.unique():
                latency = data.loc[(data["intraop_threads"] == thread) &
                                   (data["batch_size"] == batch)]["latency(avg)"].values[0]
                batch_latencies[batch] = latency

            latencies[thread] = batch_latencies

        return latencies

    def __init__(self):
        """Initialize the optimizer."""
        self.latencies = self.format_data(self.get_data("resnet", "latency"))

    def min_latency(self, max_threads, max_batch, latency):
        """
        Given a set of threads and batches, find the minimum latency.
        :param max_threads: maximum number of threads
        :param max_batch: maximum number of batches
        :param latency: matrix of latencies
        :return: optimial latency matrix
        """

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

        # print(f"latency { latency}")
        # np.set_printoptions(linewidth=np.inf)
        # print(f"opt { np.array(opt, dtype=np.float16)}")
        return opt

    def solution(self, threads, batch, latency, opt, result):
        if threads <= 0 or batch <= 0:
            return 0

        max_threads, max_batch = 0, 0
        for t in range(threads, 0, -1):
            for b in range(batch, 0, -1):
                if t in latency and b in latency[t]:
                    if opt[threads][batch] == opt[t][b] == latency[t][b]:
                        max_threads, max_batch = threads - t, batch - b
                        result.append((t, b))
                        break

        return self.solution(max_threads, max_batch, latency, opt, result)


if __name__ == "__main__":
    optimizer = Optimizer()
    assert optimizer.min_latency(
        3, 3, np.array([[0, 0, 0, 0], [0, 7, 8, 9], [0, 4, 5, 6], [0, 1, 2, 3]]))[3][3] == 3

    max_threads = max(optimizer.latencies.keys())
    max_batch = max(optimizer.latencies[max_threads].keys())
    opt = optimizer.min_latency(
        max_threads, max_batch, optimizer.latencies)

    instances = []
    optimizer.solution(max_threads, max_batch,
                       optimizer.latencies, opt, instances)
    print(instances)
