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
    def __init__(self):
        pass

    def minLatency(self, maxThreads, maxBatch, latency):
        """
        Given a set of threads and batches, find the minimum latency of processing a batch with n inputs.
        :param maxThreads: maximum number of threads
        :param maxBatch: maximum number of batches
        :param latency: matrix of latencies
        :return: optimial latency matrix
        """

        opt = [[0 for _ in range(maxBatch + 1)] for _ in range(maxThreads + 1)]
        assert latency.shape == (maxThreads + 1, maxBatch + 1)

        for t in range(1, maxThreads + 1):
            for b in range(1, maxBatch + 1):
                opt[t][b] = float('inf')
                for t_i in range(1, t + 1):
                    for b_i in range(1, b + 1):
                        opt[t][b] = min(opt[t][b], opt[t - t_i]
                                        [b - b_i] + latency[t_i][b_i])

        print(f"latency { latency}")
        print(f"opt { np.array(opt)}")
        return opt[maxThreads][maxBatch]


if __name__ == "__main__":
    optimizer = Optimizer()
    assert optimizer.minLatency(
        3, 3, np.array([[0, 0, 0, 0], [0, 7, 8, 9], [0, 4, 5, 6], [0, 1, 2, 3]])) == 3
