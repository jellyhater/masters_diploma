from abc import ABC, abstractmethod
import torch


class InpaintingPipeline(ABC):
    def __init__(
        self,
        *args
    ):
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        self.pipe = self.load_pipe(*args)
    
    @abstractmethod
    def load_pipe(self, *args):
        pass




