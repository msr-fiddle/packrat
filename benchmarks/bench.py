import sys
from interface import Interface
import interface
import torch
import os
import csv
import logging

from config import Config, Optimizations


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
    def optimize_memory_layout(self, optimization: Optimizations, model: torch.nn.Module, data: torch.Tensor):
        if optimization == Optimizations.none:
            return model, data
        elif optimization == Optimizations.script:
            model = torch.jit.script(model)
            model = torch.jit.optimize_for_inference(model)
            return model, data
        elif optimization == Optimizations.channels_last:
            return model.to(memory_format=torch.channels_last), data.to(memory_format=torch.channels_last)
        elif optimization == Optimizations.mkldnn:
            from torch.utils import mkldnn
            return mkldnn.to_mkldnn(model), data.to_mkldnn()
        elif optimization == Optimizations.ipex:
            import intel_extension_for_pytorch as ipex
            model = model.to(memory_format=torch.channels_last)
            model = ipex.optimize(model)
            data = data.to(memory_format=torch.channels_last)
            return model, data

    @interface.default
    def report(self, config: Config) -> None:
        benchmark = config.benchmark.name + '_' + \
            config.run_type.name + '_' + config.optimization.name

        def report_latency():
            if len(self.latencies) > 0:
                self.latencies.sort()
                filename = config.benchmark.name + "_latency.csv"
                exists = os.path.isfile(filename)
                writer = csv.writer(open(filename, "a+"), delimiter=",")
                if not exists:
                    writer.writerow(
                        ["benchmark", "topology", "interop_threads", "intraop_threads", "batch_size", "latency(min)", "latency(avg)", "latency(max)", "flops"])
                min, max, avg = self.latencies[0] * 1000,  (sum(self.latencies) / len(
                    self.latencies)) * 1000, self.latencies[-1] * 1000
                lat = sum(self.latencies)
                flops = (config.flops / lat)/1e9
                writer.writerow(
                    [benchmark, config.mapping.name, config.interop_threads, config.intraop_threads, config.batch_size, min, max, avg, flops])

                logging.debug("Benchmark: {}, Threads: {}, BatchSize: {}, Latencyies: Min {: .2f}, Average  {: .2f}, Max {: .2f} ".format(
                    benchmark, config.interop_threads, config.intraop_threads, config.batch_size, min, max, avg))

        def report_throughput():
            if len(self.latencies) > 0:
                filename = config.benchmark.name + "_throughput.csv"
                exists = os.path.isfile(filename)
                writer = csv.writer(open(filename, "a+"), delimiter=",")
                if not exists:
                    writer.writerow(
                        ["benchmark", "topology", "interop_threads", "intraop_threads", "batch_size", "throughput"])
                throughput = (config.batch_size *
                              config.iterations) / sum(self.latencies)
                writer.writerow(
                    [benchmark, config.mapping.name, config.interop_threads, config.intraop_threads, config.batch_size, throughput])
                logging.debug("Benchmark: {}, Threads: {}, BatchSize: {}, Throughput: {} ".format(
                              benchmark, config.interop_threads, config.intraop_threads, config.batch_size, throughput))

        report_latency()
        report_throughput()
        sys.stdout.write(str(config.flops) + '\n')
