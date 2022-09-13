# https://huggingface.co/transformers/v1.0.0/quickstart.html

from sys import argv
import timeit
import torch
from transformers import BertTokenizer, BertModel, logging
from interface import implements

from bench import Bench
from config import Benchmark, Config, RunType

logging.set_verbosity_error()


class BertBench(implements(Bench)):
    def run(self, config: Config) -> None:
        model = self.get_model(config)
        data, segments = self.get_test_data(config.batch_size)

        if config.optimization == "script":
            model = torch.jit.trace(model, data)
            model = model.optimize_for_inference()

        if config.run_type == RunType.default:
            self.inference_benchmark(config, model, data)
        elif config.run_type == RunType.manual:
            self.inference_manual(config, model, data, segments)

    def warmup(self, model, data, segments):
        with torch.no_grad():
            timeit.Timer(lambda: self.run_inference_custom(
                model, data, segments)).timeit(number=10)

    def get_test_data(self, batch_size: int) -> torch.Tensor:
        text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        input = []
        for _ in range(batch_size):
            input.append(text)
        encoded = tokenizer.batch_encode_plus(
            input, add_special_tokens=True, padding=True, return_tensors='pt', return_token_type_ids=True)

        return encoded['input_ids'], encoded['token_type_ids']

    def run_inference(self, model: torch.nn.Module, data: torch.Tensor):
        pass

    def run_inference_custom(self, model: torch.nn.Module, data: torch.Tensor, segments: torch.Tensor):
        return model(data, token_type_ids=segments)

    def inference_manual(self, config: Config, model: torch.nn.Module, data: torch.Tensor, segments):
        torch.set_num_threads(config.intraop_threads)
        torch.set_num_interop_threads(config.interop_threads)

        assert torch.get_num_threads, config.intraop_threads
        assert torch.get_num_interop_threads, config.interop_threads

        self.warmup(model, data, segments)

        for index in range(config.iterations):
            self.latencies[index] = timeit.Timer(
                lambda: self.run_inference_custom(model, data, segments)).timeit(number=1)

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
