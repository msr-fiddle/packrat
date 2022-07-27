import numpy
import torch
from pypapi import events, papi_high as high
from timeit import default_timer as timer


def test():
    print("size, expected, measured, GFLOPS/s")
    for n in [10, 30, 100, 300, 1000, 10000, 20000, 30000, 50000, 100000]:
        tensor1 = torch.randn(n, n, dtype=torch.float64)
        tensor2 = torch.randn(n, n, dtype=torch.float64)
        start_time = timer()
        high.start_counters([events.PAPI_DP_OPS, events.PAPI_SP_OPS])
        x = high.read_counters()
        assert x, [0, 0]

        # Matmul, pytorch
        torch.matmul(tensor1, tensor2)

        x = high.stop_counters()
        end_time = timer()
        print(n, n*n*(2*n-1), sum(x), (sum(x) / (end_time-start_time)) / 1e9)


test()
