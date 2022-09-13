"""
This benchmark is an extension to the benchmark explained at
https://pytorch.org/hub/pytorch_vision_resnet/
"""

import os
from sys import argv
import timeit
import urllib
from PIL import Image
import torch
import torchvision.models as models
from torchvision import transforms
from interface import implements

from bench import Bench
from config import Benchmark, Config, RunType


class ResnetBench(implements(Bench)):
    def run(self, config: Config) -> None:
        model = self.get_model(config)
        data = self.get_test_data(config.batch_size)

        model, data = self.optimize_memory_layout(
            config.optimization, model, data)

        if config.run_type == RunType.default:
            self.inference_benchmark(config, model, data)
        elif config.run_type == RunType.manual:
            self.inference_manual(config, model, data)

        # Measure FLOPS
        if config.intraop_threads == 1 and config.interop_threads == 1:
            self.measure_flops(config, model, data)

    def warmup(self, model, data):
        with torch.no_grad():
            timeit.Timer(lambda: self.run_inference(
                model, data)).timeit(number=10)

    def get_test_data(self, batch_size: int) -> torch.Tensor:
        url, filename = (
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog-{}.jpg".format(config.instance_id))
        try:
            urllib.request.urlretrieve(url, filename)
        except urllib.error.HTTPError:
            urllib.request.urlretrieve(url, filename)
        input_image = Image.open(filename)

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = []
        for i in range(0, batch_size, 1):
            input_batch.append(input_tensor)

        data = torch.stack(input_batch)
        return data

    def run_inference(self, model: torch.nn.Module, data: torch.Tensor):
        return model(data)

    def measure_flops(self, config, model, data):
        from pypapi import events, papi_high as high

        try:
            high.start_counters([events.PAPI_DP_OPS, events.PAPI_SP_OPS])

            for _ in range(config.iterations):
                self.run_inference(model, data)

            config.set_flops(sum(high.stop_counters()))
        except:
            print("Unable to use the performance counters!")

    def inference_manual(self, config: Config, model: torch.nn.Module, data: torch.Tensor):
        torch.set_num_threads(config.intraop_threads)
        torch.set_num_interop_threads(config.interop_threads)

        assert torch.get_num_threads, config.intraop_threads
        assert torch.get_num_interop_threads, config.interop_threads

        # print(torch.__config__.parallel_info())
        self.warmup(model, data)

        for index in range(config.iterations):
            self.latencies[index] = timeit.Timer(
                lambda: self.run_inference(model, data)).timeit(number=1)

    def inference_benchmark(self, model, data):
        import torch.utils.benchmark as benchmark
        num_threads = int(os.environ.get("OMP_NUM_THREADS"))
        timer = benchmark.Timer(stmt="run_inference(model, data)",
                                setup="from __main__ import run_inference",
                                globals={
                                    "model": model,
                                    "data": data,
                                },
                                num_threads=num_threads,
                                label="Latency Measurement",
                                sub_label="torch.utils.benchmark.").blocked_autorange(min_run_time=5)

        print(
            f"Thread {num_threads}, Mean: {timer.mean * 1e3:6.2f} ms, Median: {timer.median * 1e3:6.2f} ms")


if __name__ == "__main__":
    if len(argv) == 2:
        config = Config(None).from_string(argv[1])
    else:
        config = Config(None)
        config.benchmark = Benchmark.resnet
    bench = ResnetBench()
    bench.latencies = [None] * (config.iterations)
    bench.run(config)
    bench.report(config)
