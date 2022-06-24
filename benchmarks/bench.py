from interface import Interface
import torch


class Bench(Interface):
    def run_inference(self, model: torch.nn.Module, data: torch.Tensor):
        pass

    def get_model(self) -> torch.nn.Module:
        pass

    def get_test_data(self, batch_size: int) -> torch.Tensor:
        pass
