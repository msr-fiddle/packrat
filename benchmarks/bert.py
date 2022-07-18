# https://huggingface.co/transformers/v1.0.0/quickstart.html

import os
from sys import argv
import timeit
import torch
from transformers import BertTokenizer, BertModel 
from interface import implements

from bench import Bench
from config import Benchmark, Config, RunType


class BertBench(implements(Bench)):
    def run(self, config: Config) -> None:
        model = self.get_model()
        data = self.get_test_data(config.batch_size)

        model, data = self.optimize_memory_layout(
            config.optimization, model, data)

        if config.run_type == RunType.default:
            self.inference_benchmark(config, model, data)
        elif config.run_type == RunType.manual:
            self.inference_manual(config, model, data)

    def get_model(self) -> torch.nn.Module:
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()
        return model

    def warmup(self, model, data):
        with torch.no_grad():
            timeit.Timer(lambda: self.run_inference(
                model, data)).timeit(number=10)

    def get_test_data(self, batch_size: int) -> torch.Tensor:
        text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenized_text = tokenizer.tokenize(text)

        masked_index = 8
        tokenized_text[masked_index] = '[MASK]'

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

        input_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        input_batch = []
        for i in range(0, batch_size, 1):
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

        self.warmup(model, data)

        for index in range(config.iterations):
            self.latencies[index] = timeit.Timer(
                lambda: self.run_inference(model, data)).timeit(number=1)

    def inference_benchmark(self, model, data):
        pass 


if __name__ == "__main__":
    if len(argv) == 2:
        config = Config(None).from_string(argv[1])
    else:
        config = Config(None)
        config.benchmark = Benchmark.inception
    bench = BertBench()
    bench.latencies = [None] * (config.iterations)
    bench.run(config)
    bench.report(config)
