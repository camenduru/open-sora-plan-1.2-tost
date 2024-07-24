import os, json, tempfile, requests, runpod

discord_token = os.getenv('com_camenduru_discord_token')
web_uri = os.getenv('com_camenduru_web_uri')
web_token = os.getenv('com_camenduru_web_token')

import torch, imageio, uuid
import numpy as np
from typing import Tuple
from datetime import datetime
from opensora.models import CausalVAEModelWrapper
from opensora.models.causalvideovae import ae_stride_config
from transformers import AutoTokenizer, MT5EncoderModel
from diffusers import DPMSolverMultistepScheduler, SASolverScheduler
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, DEISMultistepScheduler
from opensora.models.diffusion.opensora.modeling_opensora import OpenSoraT2V
from opensora.sample.pipeline_opensora import OpenSoraPipeline

NEG_PROMPT = """
    nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
    low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry. 
    """

style_list = [
    {
        "name": "(Default)",
        "prompt": "(masterpiece), (best quality), (ultra-detailed), (unwatermarked), {prompt}",
        "negative_prompt": NEG_PROMPT,
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured. ",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo, a close-up of  {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly. ",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast. ",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style. ",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly. ",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic. ",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white. ",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured. ",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting. ",
    },
]

MAX_SEED = np.iinfo(np.int32).max

SPEED_UP_T5 = True
USE_TORCH_COMPILE = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
DEFAULT_STYLE_NAME = "(Default)"

def save_video(video):
    unique_name = str(uuid.uuid4()) + ".mp4"
    imageio.mimwrite(unique_name, video, fps=23, quality=6)
    return unique_name

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    if not negative:
        negative = ""
    return p.replace("{prompt}", positive), n + negative

if torch.cuda.is_available():
    weight_dtype = torch.bfloat16
    T5_token_max_length = 512

    vae = CausalVAEModelWrapper("/content/osp/vae", "/content/cache").eval()
    vae.vae = vae.vae.to(device=device, dtype=weight_dtype)
    vae.vae.enable_tiling()
    vae.vae.tile_overlap_factor = 0.125
    vae.vae.tile_sample_min_size = 256
    vae.vae.tile_latent_min_size = 32
    vae.vae.tile_sample_min_size_t = 29
    vae.vae.tile_latent_min_size_t = 8
    vae.vae_scale_factor = ae_stride_config["CausalVAEModel_D4_4x8x8"]

    text_encoder = MT5EncoderModel.from_pretrained("/content/mt5-xxl", cache_dir="/content/cache", low_cpu_mem_usage=True, torch_dtype=weight_dtype)
    tokenizer = AutoTokenizer.from_pretrained("/content/mt5-xxl", cache_dir="/content/cache")
    transformer = OpenSoraT2V.from_pretrained("/content/osp/video", cache_dir="/content/cache", low_cpu_mem_usage=False, device_map=None, torch_dtype=weight_dtype)
    scheduler = EulerAncestralDiscreteScheduler()
    pipe = OpenSoraPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler, transformer=transformer)
    pipe.to(device)
    print("Loaded on Device!")

    if SPEED_UP_T5:
        pipe.text_encoder.to_bettertransformer()

    if USE_TORCH_COMPILE:
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
        print("Model Compiled!")

@torch.no_grad()
@torch.inference_mode()
def generate(input):
    values = input["input"]
    prompt = values['prompt']
    negative_prompt = values['negative_prompt']
    style = values['style']
    use_negative_prompt = values['use_negative_prompt']
    seed = values['seed']
    schedule = values['schedule']
    guidance_scale = values['guidance_scale']
    num_inference_steps = values['num_inference_steps']
    randomize_seed = values['randomize_seed']
    num_frames = values['num_frames']
    width = values['width']
    height = values['height']

    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator().manual_seed(seed)

    if schedule == 'DPM-Solver':
        if not isinstance(pipe.scheduler, DPMSolverMultistepScheduler):
            pipe.scheduler = DPMSolverMultistepScheduler()
    elif schedule == "PNDM-Solver":
        if not isinstance(pipe.scheduler, PNDMScheduler):
            pipe.scheduler = PNDMScheduler()
    elif schedule == "DDIM-Solver":
        if not isinstance(pipe.scheduler, DDIMScheduler):
            pipe.scheduler = DDIMScheduler()
    elif schedule == "Euler-Solver":
        if not isinstance(pipe.scheduler, EulerDiscreteScheduler):
            pipe.scheduler = EulerDiscreteScheduler()
    elif schedule == "DDPM-Solver":
        if not isinstance(pipe.scheduler, DDPMScheduler):
            pipe.scheduler = DDPMScheduler()
    elif schedule == "EulerA-Solver":
        if not isinstance(pipe.scheduler, EulerAncestralDiscreteScheduler):
            pipe.scheduler = EulerAncestralDiscreteScheduler()
    elif schedule == "DEISM-Solver":
        if not isinstance(pipe.scheduler, DEISMultistepScheduler):
            pipe.scheduler = DEISMultistepScheduler()
    elif schedule == "SA-Solver":
        if not isinstance(pipe.scheduler, SASolverScheduler):
            pipe.scheduler = SASolverScheduler.from_config(pipe.scheduler.config, algorithm_type='data_prediction', tau_func=lambda t: 1 if 200 <= t <= 800 else 0, predictor_order=2, corrector_order=2)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    if not use_negative_prompt:
        negative_prompt = None
    prompt, negative_prompt = apply_style(style, prompt, negative_prompt)
    print(prompt, negative_prompt)
    videos = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        num_images_per_prompt=1,
        max_sequence_length=T5_token_max_length,
    ).images

    video_paths = [save_video(vid) for vid in videos]
    print(video_paths, seed)

    result = video_paths[0]

    response = None
    try:
        source_id = values['source_id']
        del values['source_id']
        source_channel = values['source_channel']     
        del values['source_channel']
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        files = {default_filename: open(result, "rb").read()}
        payload = {"content": f"{json.dumps(values)} <@{source_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{source_channel}/messages",
            data=payload,
            headers={"authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if os.path.exists(result):
            os.remove(result)

    if response and response.status_code == 200:
        try:
            payload = {"jobId": job_id, "result": response.json()['attachments'][0]['url']}
            requests.post(f"{web_uri}/api/notify", data=json.dumps(payload), headers={'Content-Type': 'application/json', "authorization": f"{web_token}"})
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            return {"result": response.json()['attachments'][0]['url']}
    else:
        return {"result": "ERROR"}

runpod.serverless.start({"handler": generate})