from argparse import Namespace
from enum import Enum


class Benchmark(Enum):
    """
    Enum for the benchmark
    """
    resnet = 1
    inception = 2
    gpt2 = 3
    bert = 4


class RunType(Enum):
    """
    Enum for the run
    """
    default = 1
    manual = 2


class ThreadMapping(Enum):
    """
    Enum for the thread mapping
    """
    sequential = 1
    interleave = 2


class ThreadPinning(Enum):
    """
    Enum for the thread pinning
    """
    numactl = 1
    omp = 2
    none = 3


class Optimizations(Enum):
    """
    Enum for the optimizations
    """
    none = 1
    script = 2
    channels_last = 3
    mkldnn = 4
    ipex = 5


class ModelSource(Enum):
    """
    Enum for the model source
    """
    torch = 1
    cache = 2
    none = 3


class Config:
    """
    Common configurations
    """

    def __init__(self, args: None):
        if args is not (None):
            self.benchmark = Benchmark[args.benchmark]
            self.run_type = RunType[args.run_type]
            self.optimization = Optimizations[args.optimization]
            self.mapping = ThreadMapping[args.mapping]
            self.batch_size = int(args.batch_size)
            self.iterations = int(args.iterations)
            self.interop_threads = int(args.interop_threads)
            self.intraop_threads = int(args.intraop_threads)
            self.flops = int(args.flops)
            self.instance_id = int(args.instance_id)
            self.pinnning = ThreadPinning[args.pinning]
            self.core_list = args.core_list
        else:
            self.benchmark = Benchmark.resnet
            self.run_type = RunType.manual
            self.optimization = Optimizations.none
            self.mapping = ThreadMapping.sequential
            self.batch_size = 1
            self.iterations = 100
            self.interop_threads = 1
            self.intraop_threads = 1
            self.flops = 0
            self.instance_id = 1
            self.pinnning = ThreadPinning.numactl
            self.core_list = [0]

    def __repr__(self):
        return 'benchmark={}, run_type={}, optimization={}, mapping={}, batch_size={}, iterations={}, interop_threads={}, intraop_threads={}, flops={}, instance_id={}, enable_numactl={}, core_list={}'.format(self.benchmark.name, self.run_type.name, self.optimization.name, self.mapping.name, self.batch_size, self.iterations, self.interop_threads, self.intraop_threads, self.flops, self.instance_id, self.pinnning.name, self.core_list)

    @classmethod
    def from_string(self, string: str):
        """
        Convert a string to a Config
        """
        import re
        args = re.split(', |=', string)
        return self(Namespace(benchmark=args[1], run_type=args[3], optimization=args[5], mapping=args[7], batch_size=args[9], iterations=args[11], interop_threads=args[13], intraop_threads=args[15], flops=int(args[17]), instance_id=int(args[19]), pinning=args[21], core_list=args[23]))

    def set_optimization(self, value: Optimizations) -> None:
        self.optimization = value

    def set_mapping(self, value: ThreadMapping) -> None:
        self.mapping = value

    def set_core_list(self, value: list) -> None:
        self.core_list = value

    def set_batch_size(self, value: int) -> None:
        self.batch_size = value

    def set_interop_threads(self, value: int) -> None:
        self.interop_threads = value

    def set_intraop_threads(self, value: int) -> None:
        self.intraop_threads = value

    def set_flops(self, value: int) -> None:
        self.flops = value

    def set_instance_id(self, value: int) -> None:
        self.instance_id = value
