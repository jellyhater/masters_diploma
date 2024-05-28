from abc import ABC, abstractmethod
import torch


class InpaintingPipeline(ABC):
    def __init__(
        self,
    ):
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

    @abstractmethod
    def load_pipe(self):
        pass



