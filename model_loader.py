from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass

IDM_PATH = os.getenv("IDM_VTON_PATH", "/opt/IDM-VTON")
MODEL_ID = "yisol/IDM-VTON"
MODEL_REVISION = "585a32e74aee241cbc0d0cc3ab21392ca58c916a"
SPACE_REVISION = "906d38c8a74e7c1cd0bf714a363fe2e939fa28b8"


@dataclass
class ModelBundle:
    pipe: object
    parsing: object
    openpose: object
    torch: object
    device: str
    dtype: object


_bundle = None
_load_seconds = 0.0
_lock = threading.Lock()


def _download_preprocessors(hf_hub_download):
    ckpt = os.path.join(IDM_PATH, "ckpt")
    files = (
        "ckpt/densepose/model_final_162be9.pkl",
        "ckpt/humanparsing/parsing_atr.onnx",
        "ckpt/humanparsing/parsing_lip.onnx",
        "ckpt/openpose/ckpts/body_pose_model.pth",
    )
    for filename in files:
        target = os.path.join(IDM_PATH, filename)
        if not os.path.exists(target):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            downloaded = hf_hub_download("yisol/IDM-VTON", filename, repo_type="space", revision=SPACE_REVISION)
            import shutil
            shutil.copy2(downloaded, target)


def _load() -> ModelBundle:
    if IDM_PATH not in sys.path:
        sys.path.insert(0, IDM_PATH)
    gradio_dir = os.path.join(IDM_PATH, "gradio_demo")
    if gradio_dir not in sys.path:
        sys.path.insert(0, gradio_dir)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for IDM-VTON inference")
    from huggingface_hub import hf_hub_download
    _download_preprocessors(hf_hub_download)
    from diffusers import AutoencoderKL, DDPMScheduler
    from transformers import (AutoTokenizer, CLIPImageProcessor, CLIPTextModel,
                              CLIPTextModelWithProjection, CLIPVisionModelWithProjection)
    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
    from src.unet_hacked_garmnet import UNet2DConditionModel as GarmentUNet
    from src.unet_hacked_tryon import UNet2DConditionModel
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose

    mixed = os.getenv("IDM_MIXED_PRECISION", "fp16").lower()
    dtype = torch.bfloat16 if mixed == "bf16" else (torch.float16 if mixed == "fp16" else torch.float32)
    kw = {"revision": MODEL_REVISION, "torch_dtype": dtype}
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", **kw)
    garment_unet = GarmentUNet.from_pretrained(MODEL_ID, subfolder="unet_encoder", **kw)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer", revision=MODEL_REVISION, use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer_2", revision=MODEL_REVISION, use_fast=False)
    scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler", revision=MODEL_REVISION)
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", **kw)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(MODEL_ID, subfolder="text_encoder_2", **kw)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, subfolder="image_encoder", **kw)
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", **kw)
    for model in (unet, garment_unet, text_encoder, text_encoder_2, image_encoder, vae):
        model.requires_grad_(False)
    pipe = TryonPipeline.from_pretrained(MODEL_ID, revision=MODEL_REVISION, unet=unet, vae=vae,
        feature_extractor=CLIPImageProcessor(), text_encoder=text_encoder, text_encoder_2=text_encoder_2,
        tokenizer=tokenizer, tokenizer_2=tokenizer_2, scheduler=scheduler, image_encoder=image_encoder,
        torch_dtype=dtype)
    pipe.unet_encoder = garment_unet
    pipe.to("cuda:0")
    pipe.unet_encoder.to("cuda:0")
    parsing, openpose = Parsing(0), OpenPose(0)
    openpose.preprocessor.body_estimation.model.to("cuda:0")
    return ModelBundle(pipe, parsing, openpose, torch, "cuda:0", dtype)


def get_model():
    global _bundle, _load_seconds
    if _bundle is None:
        with _lock:
            if _bundle is None:
                started = time.perf_counter()
                _bundle = _load()
                _load_seconds = time.perf_counter() - started
    return _bundle, _load_seconds
