"""
The problem we have is the 2D analogue of this knapsack problem.

opt[t, b] is the total latency of processing b inputs with t threads.
Then you can compute opt[t, b] just looking at opt[t', b'] where t' < t and b' < b.

opt[t, b] can be computed as follows:
opt[t, b] = min_{t', b'} (max(opt(t-t', b-b'), latency[t', b'])

latency[t',b'] = latency of processing a batch with b' inputs with t' threads (from profiles)
"""


import numpy as np


class Optimizer:
    """
    Configuration optimizer for the benchmark.
    """

    def __init__(self):
        pass

    def min_latency(self, max_threads, max_batch, latency):
        """
        Given a set of threads and batches, find the minimum latency.
        :param max_threads: maximum number of threads
        :param max_batch: maximum number of batches
        :param latency: matrix of latencies
        :return: optimial latency matrix
        """

        opt = [[0 for _ in range(max_batch + 1)]
               for _ in range(max_threads + 1)]
        assert latency.shape == (max_threads + 1, max_batch + 1)

        for batch in range(1, max_batch + 1):
            opt[0][batch] = float('inf')

        for thread in range(1, max_threads + 1):
            for batch in range(1, max_batch + 1):
                opt[thread][batch] = float('inf')
                for t_i in range(1, thread + 1):
                    for b_i in range(1, batch + 1):
                        opt[thread][batch] = min(opt[thread][batch], opt[thread - t_i]
                                                 [batch - b_i] + latency[t_i][b_i])

        # print(f"latency { latency}")
        # print(f"opt { np.array(opt)}")
        return opt[max_threads][max_batch]


if __name__ == "__main__":
    optimizer = Optimizer()
    assert optimizer.min_latency(
        3, 3, np.array([[0, 0, 0, 0], [0, 7, 8, 9], [0, 4, 5, 6], [0, 1, 2, 3]])) == 3
