# https://huggingface.co/transformers/v1.0.0/quickstart.html

import os
from sys import argv
import timeit
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from interface import implements

from .bench import Bench
from .config import Benchmark, Config, RunType


class GptBench(implements(Bench)):
    def run(self, config: Config) -> None:
        model = self.get_model(config)
        data = self.get_test_data(config)

        if config.optimization == "script":
            model = torch.jit.trace(model, data)
            model = model.optimize_for_inference()

        if config.run_type == RunType.default:
            self.inference_benchmark(config, model, data)
        elif config.run_type == RunType.manual:
            self.inference_manual(config, model, data)

    def warmup(self, model, data):
        with torch.no_grad():
            timeit.Timer(lambda: self.run_inference(
                model, data)).timeit(number=10)

    def get_test_data(self, config:  Config) -> torch.Tensor:
        text = "Who was Jim Henson ? Jim Henson was a"
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        indexed_tokens = tokenizer.encode(text)
        input_tensor = torch.tensor([indexed_tokens])

        input_batch = []
        for i in range(0, config.batch_size, 1):
            input_batch.append(input_tensor)

        data = torch.stack(input_batch)
        return data

    def run_inference(self, model: torch.nn.Module, data: torch.Tensor):
        return model(data)

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
        config.benchmark = Benchmark.gpt2
    bench = GptBench()
    bench.latencies = [None] * (config.iterations)
    bench.run(config)
    bench.report(config)
