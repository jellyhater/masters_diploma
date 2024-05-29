import torch
import colorsys
import numpy as np
import gradio as gr
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from diffusers import StableDiffusionXLImg2ImgPipeline

from kandinsky_pipe import KandinskyInpaintingPipeline
from utils import poisson_blend

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)

pipe = KandinskyInpaintingPipeline().pipe
pipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", 
    torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    cache_dir='weights/sdxl_refiner/'
).to(device)


with gr.Blocks() as demo:
    gr.Markdown("# Master diploma: masked object generation")
    gr.Markdown(
        """
    To try the demo, upload an image and select object(s) you want to inpaint.
    Write a prompt to control the inpainting.
    Click on the "Submit" button to inpaint the selected object(s).
    Check "Background" to inpaint the background instead of the selected object(s).
    """
    )
    selected_pixels = gr.State([])
    with gr.Row():
        input_img = gr.Image(label="Input")
        mask_img = gr.Image(label="Mask", interactive=False)
        output_img = gr.Image(label="Output", interactive=False)
        refined_img = gr.Image(label="Refined", interactive=False)

    with gr.Row():
        prompt_text = gr.Textbox(lines=1, label="Prompt")
        is_background = gr.Checkbox(label="Background")
        smooth_results = gr.Checkbox(label="Possion blending")

    with gr.Row():
        submit = gr.Button("Submit")
        refine = gr.Button("Refine")
        clear = gr.Button("Clear")

    def generate_mask(image, bg, sel_pix, evt: gr.SelectData):
        sel_pix.append(evt.index)
        predictor.set_image(image)
        input_point = np.array(sel_pix)
        input_label = np.ones(input_point.shape[0])
        mask, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        # clear torch cache

        torch.cuda.empty_cache()
        if bg:
            mask = np.logical_not(mask)

        mask_img = Image.fromarray(mask[0, :, :])

        torch.cuda.empty_cache()

        return mask_img

    def inpaint(image, mask, prompt):
        image = Image.fromarray(image)
        mask = np.array(Image.fromarray(mask).convert("1"))
        mask = (mask > 0).astype(int)
        # inpaint
        output = pipe(
            text=prompt,
            image=image,
            mask=mask,
        )
        return output[0]
    
    def refine_output(output_img, prompt, mask, smooth_results):
        results_raw = [Image.fromarray(output_img)]
        mask = np.array(Image.fromarray(mask).convert("1"))
        mask = (mask > 0).astype(int)
        # refine output image
        results = []
        refiner_bs = 4
        k, m = len(results_raw) // refiner_bs, len(results_raw) % refiner_bs
        i = 0
        for minibatch in [refiner_bs] * k + [m]:
            if minibatch == 0:
                continue
            #torch.manual_seed(seed)
            results.extend(pipe_refiner(
                prompt, image=results_raw[i:i+minibatch], num_images_per_prompt=minibatch
            ).images)
            i += minibatch
        # poisson blend
        if smooth_results:
            results = [poisson_blend(output_img, result, mask) for result in results]

        torch.cuda.empty_cache()
        return results[0]

    def _clear(img, mask, out, prompt, neg_prompt, bg, smooth_results, refined_img):
        img = None
        mask = None
        out = None
        prompt = ""
        neg_prompt = ""
        bg = False,
        smooth_results = False,
        refined_img = None
        return img, mask, out, prompt, neg_prompt, bg, smooth_results, refined_img

    input_img.select(
        generate_mask,
        [input_img, is_background, selected_pixels],
        [mask_img],
    )
    submit.click(
        inpaint,
        inputs=[input_img, mask_img, prompt_text],
        outputs=[output_img],
    )
    refine.click(
        refine_output,
        inputs=[output_img, prompt_text, mask_img, smooth_results],
        outputs=[refined_img],
    )
    clear.click(
        _clear,
        inputs=[
            selected_pixels,
            input_img,
            mask_img,
            output_img,
            prompt_text,
            is_background,
            smooth_results,
            refined_img
        ],
        outputs=[
            input_img,
            mask_img,
            output_img,
            prompt_text,
            is_background,
            smooth_results,
            refined_img
        ],
    )

if __name__ == "__main__":
    demo.launch(share=True)