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
    channels_last = 2
    mkldnn = 2
    ipex = 3


class Config:
    """
    Common configurations
    """

    def __init__(self):
        self.read()

    def read(self):
        with open('config.json', 'r') as json_file:
            self.data = json.load(json_file)
            self.benchmark = Benchmark[self.data['benchmark']]
            self.run_type = RunType[self.data['run_type']]
            self.optimization = Optimizations[self.data['optimization']]
            self.batch_size = int(self.data['batch_size'])
            self.iterations = int(self.data['iterations'])
            self.interop_threads = int(self.data['interop_threads'])
            self.intraop_threads = int(self.data['intraop_threads'])
            self.core_list = self.data['core_list']
            json_file.close()

    def reinitialize(self) -> None:
        self.data = {
            'benchmark': self.benchmark.resnet.name,
            'run_type': self.run_type.manual.name,
            'optimization': Optimizations.none.name,
            'batch_size': 1,
            'iterations': 100,
            'interop_threads': 1,
            'intraop_threads': 1,
            'core_list': []
        }
        self.update()

    def set_optimization(self, value: Optimizations) -> None:
        self.data['optimization'] = value.name

    def set_core_list(self, value: list) -> None:
        self.data['core_list'] = value

    def set_batch_size(self, value: int) -> None:
        self.data['batch_size'] = value

    def set_interop_threads(self, value: int) -> None:
        self.data['interop_threads'] = value

    def set_intraop_threads(self, value: int) -> None:
        self.data['intraop_threads'] = value

    def update(self) -> None:
        with open('config.json', 'w') as outfile:
            outfile.write(json.dumps(self.data, indent=4))
            outfile.close()
        self.read()
