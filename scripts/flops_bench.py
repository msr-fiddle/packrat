import time
import numpy
from pypapi import events, papi_high as high


def test():
    for n in [10, 30, 100, 300, 1000, 10000, 20000, 30000, 50000, 100000]:
        aa = numpy.mgrid[0:n:1, 0:n:1][0]
        high.start_counters([events.PAPI_DP_OPS])
        start_time = time.time()
        a = numpy.fft.fft(aa)
        end_time = time.time()
        x = high.stop_counters()
        print(n, x[0] / (end_time-start_time))


test()
