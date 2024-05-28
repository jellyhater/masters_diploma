import torch
from omegaconf import OmegaConf
from diffusers import StableDiffusionXLInpaintPipeline
# from kandinsky3 import get_T2I_unet, get_T5encoder, get_movq, get_inpainting_unet
# from kandinsky3 import Kandinsky3T2IPipeline, Kandinsky3InpaintingPipeline


class InpaintingPipeline:

    def __init__(self, model_name):
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        self.conf = OmegaConf.load("config.yaml")
        
        self.pipe = self.load_pipe(model_name)
        self.pipe.to(self.device)

    def load_pipe(self, model_name):
        conf = self.conf[model_name]

        classes_map = {
            "sdxl": StableDiffusionXLInpaintPipeline,
            # "kandi3": Kandinsky3InpaintingPipeline
        }

        dtype_map = {
            "fp16": torch.float16,
            "fp32": torch.float32
        }

        torch_dtype_str = conf["torch_dtype"]
        torch_dtype = dtype_map[torch_dtype_str]

        pipe_class = classes_map[model_name]
        pipe = pipe_class.from_pretrained(
            conf["checkpoint"],
            torch_dtype=torch_dtype,
            use_safetensors=conf["use_safetensors"]
        )
        return pipe


if __name__ == "__main__":
    pipe = InpaintingPipeline("sdxl")

