import torch
from omegaconf import OmegaConf
from diffusers import *
from inpainting_pipe import InpaintingPipeline


class HuggingfacePipeline(InpaintingPipeline):

    def __init__(self, model_name):
        super().__init__(model_name)
        self.pipe.to(self.device)

    def load_pipe(self, model_name):
        conf = OmegaConf.load("config.yaml")
        conf = conf[model_name]

        dtype_map = {
            "fp16": torch.float16,
            "fp32": torch.float32
        }

        torch_dtype_str = conf["torch_dtype"]
        torch_dtype = dtype_map[torch_dtype_str]

        pipe_class_str = conf["class"]
        pipe_class = eval(pipe_class_str)

        pipe = pipe_class.from_pretrained(
            conf["checkpoint"],
            torch_dtype=torch_dtype,
            use_safetensors=conf["use_safetensors"]
        )
        return pipe


if __name__ == "__main__":
    pipe = HuggingfacePipeline("sdxl")

