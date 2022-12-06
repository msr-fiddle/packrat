import sys
from interface import Interface
import interface
import torch
import os
import csv
import logging
import fcntl

from .cache import store
from .config import Benchmark, Config, Optimizations, ModelSource

logging.basicConfig(level=logging.INFO)


class Bench(Interface):
    Interface.latencies = []

    def run(self, config: Config) -> None:
        pass

    def run_inference(self, model: torch.nn.Module, data: torch.Tensor):
        pass

    @interface.default
    def get_model(self, config: Config) -> torch.nn.Module:
        """
        Cache store can return the model from the torch hub or from the cache.
        However, torch hub does not support torchscript for default language-based models.
        """
        from transformers import GPT2LMHeadModel, BertModel

        if config.optimization == Optimizations.script:
            if config.benchmark == Benchmark.gpt2:
                model = GPT2LMHeadModel.from_pretrained(
                    'gpt2', torchscript=True)
                model.eval()
                return model
            if config.benchmark == Benchmark.bert:
                model = BertModel.from_pretrained(
                    'bert-base-uncased', torchscript=True)
                model.eval()
                return model

        if config.source == ModelSource.torch:
            return store.get_model_from_torch(config.benchmark.name)

        if config.source == ModelSource.cache:
            assert config.storename is not None, "Store is not set"
            return store.get_model_from_plasma(config.storename, config.benchmark.name)

        raise Exception("Invalid source")

    def get_test_data(self, config: Config) -> torch.Tensor:
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

        row_header = ["benchmark", "topology", "pinning", "source", "allocator", "instance_id",
                      "interop_threads", "intraop_threads", "batch_size"]
        row = [benchmark, config.mapping.name, config.pinnning.name, config.source.name, config.allocator.name,
               config.instance_id, config.interop_threads, config.intraop_threads, config.batch_size]

        def report_latency():
            if len(self.latencies) > 0:
                self.latencies.sort()
                filename = config.benchmark.name + "_latency.csv"
                exists = os.path.isfile(filename)
                l_file = open(filename, "a+")
                writer = csv.writer(l_file, delimiter=",")

                if not exists:
                    writer.writerow(
                        row_header + ["latency(min)", "latency(avg)", "latency(max)", "flops"])
                min, max, avg = self.latencies[0] * 1000,  (sum(self.latencies) / len(
                    self.latencies)) * 1000, self.latencies[-1] * 1000
                lat = sum(self.latencies)
                flops = (config.flops / lat)/1e9
                fcntl.flock(l_file, fcntl.LOCK_EX)
                writer.writerow(
                    row + [min, max, avg, flops])
                fcntl.flock(l_file, fcntl.LOCK_UN)

        def report_throughput():
            if len(self.latencies) > 0:
                filename = config.benchmark.name + "_throughput.csv"
                exists = os.path.isfile(filename)
                t_file = open(filename, "a+")
                writer = csv.writer(t_file, delimiter=",")
                if not exists:
                    writer.writerow(
                        row_header + ["throughput"])
                throughput = (config.batch_size *
                              config.iterations) / sum(self.latencies)
                fcntl.flock(t_file, fcntl.LOCK_EX)
                writer.writerow(
                    row + [throughput])
                fcntl.flock(t_file, fcntl.LOCK_UN)

        report_latency()
        report_throughput()
        sys.stdout.write(str(config.flops) + '\n')
