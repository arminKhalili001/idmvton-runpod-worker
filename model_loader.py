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
DEFAULT_PREPROCESSOR_PATH = "/opt/IDM-VTON/ckpt"
RUNPOD_CACHE_SNAPSHOT = os.path.join(
    "/runpod-volume/huggingface-cache/hub/models--yisol--IDM-VTON/snapshots",
    MODEL_REVISION,
)
MODEL_COMPONENTS = (
    "unet/config.json",
    "unet_encoder/config.json",
    "tokenizer",
    "tokenizer_2",
    "scheduler",
    "text_encoder",
    "text_encoder_2",
    "image_encoder",
    "vae",
)
PREPROCESSOR_FILES = (
    "densepose/model_final_162be9.pkl",
    "humanparsing/parsing_atr.onnx",
    "humanparsing/parsing_lip.onnx",
    "openpose/ckpts/body_pose_model.pth",
)
PREPROCESSOR_MIN_BYTES = 1_000_000
GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


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


def _env_flag(name, default=False):
    return os.getenv(name, "true" if default else "false").strip().lower() == "true"


def _hf_home():
    return os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def _snapshot_under_hf_home():
    return os.path.join(
        _hf_home(),
        "hub",
        "models--yisol--IDM-VTON",
        "snapshots",
        MODEL_REVISION,
    )


def _resolve_model_source():
    searched = []
    explicit_path = os.getenv("IDM_MODEL_PATH")
    candidates = [explicit_path, RUNPOD_CACHE_SNAPSHOT, _snapshot_under_hf_home()]
    for candidate in candidates:
        if not candidate or candidate in searched:
            continue
        searched.append(candidate)
        if os.path.isdir(candidate) and os.path.isfile(
            os.path.join(candidate, "model_index.json")
        ):
            return candidate, True

    if _env_flag("IDM_ALLOW_REMOTE_DOWNLOAD"):
        return MODEL_ID, False

    searched_text = "\n".join(f"- {path}" for path in searched)
    raise RuntimeError(
        "IDM-VTON model snapshot is missing or incomplete. Searched paths:\n"
        f"{searched_text}\n"
        "Each snapshot must contain model_index.json. Populate the RunPod Model "
        "Cache or mount a Network Volume containing the pinned IDM-VTON snapshot."
    )


def _validate_local_model(model_source):
    required = ("model_index.json",) + MODEL_COMPONENTS
    missing = []
    for relative_path in required:
        path = os.path.join(model_source, relative_path)
        exists = os.path.isfile(path) if relative_path.endswith(".json") else os.path.exists(path)
        print(
            f"model component validation: {relative_path}={'ok' if exists else 'missing'}",
            flush=True,
        )
        if not exists:
            missing.append(relative_path)
    if missing:
        raise RuntimeError(
            f"Local IDM-VTON snapshot at {model_source} is incomplete; missing: "
            f"{', '.join(missing)}. Populate the RunPod Model Cache or mount a "
            "Network Volume containing the complete pinned snapshot."
        )


def _preprocessor_problem(path):
    if not os.path.isfile(path):
        return "missing or not a regular file"
    size = os.path.getsize(path)
    with open(path, "rb") as file:
        prefix = file.read(len(GIT_LFS_POINTER_PREFIX))
    if prefix == GIT_LFS_POINTER_PREFIX:
        return "is a Git LFS pointer, not the binary model"
    if size < PREPROCESSOR_MIN_BYTES:
        return f"is too small ({size} bytes; expected at least {PREPROCESSOR_MIN_BYTES})"
    return None


def _invalid_preprocessors(preprocessor_path):
    invalid = {}
    for filename in PREPROCESSOR_FILES:
        path = os.path.join(preprocessor_path, filename)
        problem = _preprocessor_problem(path)
        if problem:
            invalid[filename] = problem
        else:
            print(
                f"preprocessor validation: {filename}=ok bytes={os.path.getsize(path)}",
                flush=True,
            )
    return invalid


def _prepare_preprocessors(hf_hub_download, allow_remote_download):
    preprocessor_path = os.getenv(
        "IDM_PREPROCESSOR_PATH", DEFAULT_PREPROCESSOR_PATH
    )
    invalid = _invalid_preprocessors(preprocessor_path)
    if invalid and not allow_remote_download:
        details = "; ".join(
            f"{filename}: {problem}" for filename, problem in invalid.items()
        )
        raise RuntimeError(
            f"Invalid IDM-VTON preprocessors under {preprocessor_path}: {details}. "
            "Populate IDM_PREPROCESSOR_PATH with real binary files during the "
            "Docker build or from a mounted Network Volume. Remote repair is "
            "disabled because IDM_ALLOW_REMOTE_DOWNLOAD=false."
        )
    if invalid:
        import shutil

        for filename in invalid:
            target = os.path.join(preprocessor_path, filename)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            downloaded = hf_hub_download(
                MODEL_ID,
                f"ckpt/{filename}",
                repo_type="space",
                revision=SPACE_REVISION,
            )
            shutil.copy2(downloaded, target)
        remaining_invalid = _invalid_preprocessors(preprocessor_path)
        if remaining_invalid:
            raise RuntimeError(
                f"Downloaded IDM-VTON preprocessors failed validation: {remaining_invalid}"
            )
    return preprocessor_path


def _load() -> ModelBundle:
    model_source, local_only = _resolve_model_source()
    allow_remote_download = _env_flag("IDM_ALLOW_REMOTE_DOWNLOAD")
    print(f"resolved model source: {model_source}", flush=True)
    print(f"local-only mode active: {local_only}", flush=True)
    print(f"HF_HOME: {_hf_home()}", flush=True)
    if local_only:
        _validate_local_model(model_source)
    else:
        for relative_path in ("model_index.json",) + MODEL_COMPONENTS:
            print(
                f"model component validation: {relative_path}=remote fallback",
                flush=True,
            )
    if IDM_PATH not in sys.path:
        sys.path.insert(0, IDM_PATH)
    gradio_dir = os.path.join(IDM_PATH, "gradio_demo")
    if gradio_dir not in sys.path:
        sys.path.insert(0, gradio_dir)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for IDM-VTON inference")
    from huggingface_hub import hf_hub_download
    _prepare_preprocessors(hf_hub_download, allow_remote_download)
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
    source_kw = {"local_files_only": True} if local_only else {"revision": MODEL_REVISION}
    model_kw = {**source_kw, "torch_dtype": dtype}
    print("model loading started", flush=True)
    unet = UNet2DConditionModel.from_pretrained(model_source, subfolder="unet", **model_kw)
    garment_unet = GarmentUNet.from_pretrained(model_source, subfolder="unet_encoder", **model_kw)
    tokenizer = AutoTokenizer.from_pretrained(model_source, subfolder="tokenizer", use_fast=False, **source_kw)
    tokenizer_2 = AutoTokenizer.from_pretrained(model_source, subfolder="tokenizer_2", use_fast=False, **source_kw)
    scheduler = DDPMScheduler.from_pretrained(model_source, subfolder="scheduler", **source_kw)
    text_encoder = CLIPTextModel.from_pretrained(model_source, subfolder="text_encoder", **model_kw)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_source, subfolder="text_encoder_2", **model_kw)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_source, subfolder="image_encoder", **model_kw)
    vae = AutoencoderKL.from_pretrained(model_source, subfolder="vae", **model_kw)
    for model in (unet, garment_unet, text_encoder, text_encoder_2, image_encoder, vae):
        model.requires_grad_(False)
    pipe = TryonPipeline.from_pretrained(model_source, unet=unet, vae=vae,
        feature_extractor=CLIPImageProcessor(), text_encoder=text_encoder, text_encoder_2=text_encoder_2,
        tokenizer=tokenizer, tokenizer_2=tokenizer_2, scheduler=scheduler, image_encoder=image_encoder,
        torch_dtype=dtype, **source_kw)
    pipe.unet_encoder = garment_unet
    pipe.to("cuda:0")
    pipe.unet_encoder.to("cuda:0")
    parsing, openpose = Parsing(0), OpenPose(0)
    openpose.preprocessor.body_estimation.model.to("cuda:0")
    print("model loading completed", flush=True)
    return ModelBundle(pipe, parsing, openpose, torch, "cuda:0", dtype)


def get_model():
    global _bundle, _load_seconds
    if _bundle is None:
        with _lock:
            if _bundle is None:
                started = time.perf_counter()
                _bundle = _load()
                _load_seconds = time.perf_counter() - started
                print(f"total model load seconds: {_load_seconds:.3f}", flush=True)
    return _bundle, _load_seconds
