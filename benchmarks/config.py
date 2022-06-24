from enum import Enum


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
    benchmark: Benchmark
    run_type: RunType
    batch_size: int
    core_list: list

    def __init__(self):
        """
        Constructor
        """
        self.benchmark = Benchmark.resnet
        self.run_type = RunType.manual
        self.batch_size = 1
        self.core_list = []

    def __setattr__(self, name, value):
        super(Config, self).__setattr__(name, value)


global cur_config
cur_config = Config()
