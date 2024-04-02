""" Example handler file. """

import runpod
from diffusers import AutoPipelineForImage2Image
import torch
import base64
import io
import time
import re
from PIL import Image


DEFAULT_NOISE_STRENGTH        = 0.7 # 0.5 works well too
RANDOM_SEED                   = 231
generator = torch.manual_seed(RANDOM_SEED)

torch_device = None

if torch.cuda.is_available():
    torch_device = "cuda"
else:
    torch_device = "mps"

print("Torch device: ", torch_device)


try:
    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sdxl-turbo", 
        torch_dtype=torch.float16, 
        variant="fp16"
    ).to(torch_device)

except RuntimeError:
    quit()

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    prompt = job_input['prompt']

    image_data = re.sub('^data:image/.+;base64,', '', job_input['image'])
    
    # Convert the base64 image to a PIL Image
    ref_image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    width = int(job_input['width'])
    height = int(job_input['height'])
    
    ref_image = ref_image.resize((width, height))

    time_start = time.time()
    
    image = pipe(
        prompt = prompt, 
        num_inference_steps = 4, 
        guidance_scale = 0.0,
        width = width,
        height = height,
        generator = generator,
        image = ref_image,
        strength = DEFAULT_NOISE_STRENGTH,
        ).images[0]
    print(f"Time taken: {time.time() - time_start}")


    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    return base64.b64encode(image_bytes).decode('utf-8')


runpod.serverless.start({"handler": handler})
