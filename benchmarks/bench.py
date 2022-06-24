from interface import Interface
import interface
import torch
import os
import csv
import logging

from config import Config


class Bench(Interface):
    Interface.latencies = []

    def run(self, config: Config) -> None:
        pass

    def run_inference(self, model: torch.nn.Module, data: torch.Tensor):
        pass

    def get_model(self) -> torch.nn.Module:
        pass

    def get_test_data(self, batch_size: int) -> torch.Tensor:
        pass

    @interface.default
    def report(self, config: Config) -> None:
        num_threads = len(config.core_list)

        def report_latency():
            if len(self.latencies) > 0:
                self.latencies.sort()
                filename = config.benchmark.name + "_latency.csv"
                exists = os.path.isfile(filename)
                writer = csv.writer(open(filename, "a+"), delimiter=",")
                if not exists:
                    writer.writerow(
                        ["threads", "batch_size", "latency(min), latency(avg), latency(max)"])
                min, max, avg = self.latencies[0] * 1000,  (sum(self.latencies) / len(
                    self.latencies)) * 1000, self.latencies[-1] * 1000
                writer.writerow([num_threads, config.batch_size,
                                self.latencies[0], min, max, avg])

                logging.debug("Threads: {}, BatchSize {}, Latencyies: Min {:.2f}, Average  {:.2f}, Max {:.2f} ".format(
                    num_threads, config.batch_size, min, max, avg))

        def report_throughput():
            if len(self.latencies) > 0:
                filename = config.benchmark.name + "_throughput.csv"
                exists = os.path.isfile(filename)
                writer = csv.writer(open(filename, "a+"), delimiter=",")
                if not exists:
                    writer.writerow(
                        ["threads", "batch_size", "throughput"])
                throughput = len(self.latencies) / sum(self.latencies)
                writer.writerow(
                    [num_threads, config.batch_size, throughput])
                logging.debug("Threads: {}, BatchSize {}, Throughput: {} ".format(
                              num_threads, config.batch_size, throughput))

        report_latency()
        report_throughput()
        return
