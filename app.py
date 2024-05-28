import torch
import colorsys
import numpy as np
import gradio as gr
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

from huggingface_pipe import HuggingfacePipeline

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)

pipe = HuggingfacePipeline(model_name="sdxl").pipe


with gr.Blocks() as demo:
    gr.Markdown("# MPCard project: background generation")
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

    with gr.Row():
        prompt_text = gr.Textbox(lines=1, label="Prompt")
        is_background = gr.Checkbox(label="Background")

    with gr.Row():
        submit = gr.Button("Submit")
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
        mask = Image.fromarray(mask[0, :, :])
        segs = mask_generator.generate(image)
        boolean_masks = [s["segmentation"] for s in segs]
        finseg = np.zeros((boolean_masks[0].shape[0], boolean_masks[0].shape[1], 3), dtype=np.uint8)
        # Loop over the boolean masks and assign a unique color to each class
        for class_id, boolean_mask in enumerate(boolean_masks):
            hue = class_id * 1.0 / len(boolean_masks)
            rgb = tuple(int(i * 255) for i in colorsys.hsv_to_rgb(hue, 1, 1))
            rgb_mask = np.zeros((boolean_mask.shape[0], boolean_mask.shape[1], 3), dtype=np.uint8)
            rgb_mask[:, :, 0] = boolean_mask * rgb[0]
            rgb_mask[:, :, 1] = boolean_mask * rgb[1]
            rgb_mask[:, :, 2] = boolean_mask * rgb[2]
            finseg += rgb_mask

        torch.cuda.empty_cache()

        return mask

    def inpaint(image, mask, prompt):
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        output = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=20,
        ).images[0]
        torch.cuda.empty_cache()
        return output

    def _clear(sel_pix, img, mask, out, prompt, neg_prompt, bg):
        sel_pix = []
        img = None
        mask = None
        out = None
        prompt = ""
        neg_prompt = ""
        bg = False
        return img, mask, out, prompt, neg_prompt, bg

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
    clear.click(
        _clear,
        inputs=[
            selected_pixels,
            input_img,
            mask_img,
            output_img,
            prompt_text,
            is_background,
        ],
        outputs=[
            input_img,
            mask_img,
            output_img,
            prompt_text,
            is_background,
        ],
    )

if __name__ == "__main__":
    demo.launch(share=True)