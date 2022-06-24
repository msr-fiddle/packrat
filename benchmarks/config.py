from enum import Enum
import json


class Benchmark(Enum):
    """
    Enum for the benchmark
    """
    resnet = 1


class RunType(Enum):
    """
    Enum for the run
    """
    default = 1
    manual = 2


class Optimizations(Enum):
    """
    Enum for the optimizations
    """
    none = 1
    memory_layout = 2
    ipex_extensions = 3


class Config:
    """
    Common configurations
    """

    def __init__(self):
        with open('config.json', 'r') as json_file:
            self.data = json.load(json_file)
            self.benchmark = Benchmark[self.data['benchmark']]
            self.run_type = RunType[self.data['run_type']]
            self.batch_size = int(self.data['batch_size'])
            self.iterations = int(self.data['iterations'])
            self.core_list = self.data['core_list']
            json_file.close()

    def set_core_list(self, value: list) -> None:
        self.data['core_list'] = value

    def set_batch_size(self, value: int) -> None:
        self.data['batch_size'] = value

    def update(self) -> None:
        with open('config.json', 'w') as outfile:
            json.dump(self.data, outfile, indent=4)
