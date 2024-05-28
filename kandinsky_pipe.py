from inpainting_pipe import InpaintingPipeline
from kandinsky3 import get_inpainting_pipeline

class KandinskyInpaintingPipeline(InpaintingPipeline):

    def load_pipe(self):
        pipe = get_inpainting_pipeline(
            device_map=self.device,
        )
        return pipe
    
if __name__ == "__main__":
    pipe = KandinskyInpaintingPipeline()