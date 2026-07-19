from __future__ import annotations

import os
from contextlib import nullcontext

from image_utils import fit_rgb


def run_inference(bundle, request):
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms.functional import to_pil_image
    from gradio_demo.utils_mask import get_mask_location
    from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
    import apply_net

    torch, device, dtype = bundle.torch, bundle.device, bundle.dtype
    human, garment = fit_rgb(request.person_image), fit_rgb(request.garment_image)
    keypoints = bundle.openpose(human.resize((384, 512)))
    model_parse, _ = bundle.parsing(human.resize((384, 512)))
    mask, _ = get_mask_location("dc", request.garment_category, model_parse, keypoints)
    mask = mask.resize((768, 1024), Image.Resampling.NEAREST)

    human_bgr = convert_PIL_to_numpy(_apply_exif_orientation(human.resize((384, 512))), format="BGR")
    parser = apply_net.create_argument_parser()
    densepose_config = os.path.join(os.getenv("IDM_VTON_PATH", "/opt/IDM-VTON"), "configs/densepose_rcnn_R_50_FPN_s1x.yaml")
    densepose_ckpt = os.path.join(os.getenv("IDM_VTON_PATH", "/opt/IDM-VTON"), "ckpt/densepose/model_final_162be9.pkl")
    args = parser.parse_args(("show", densepose_config, densepose_ckpt, "dp_segm", "-v", "--opts", "MODEL.DEVICE", "cuda"))
    pose = Image.fromarray(args.func(args, human_bgr)[:, :, ::-1]).resize((768, 1024))
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    pose_tensor = normalize(pose).unsqueeze(0).to(device, dtype)
    garment_tensor = normalize(garment).unsqueeze(0).to(device, dtype)
    amp = torch.autocast("cuda", dtype=dtype) if dtype != torch.float32 else nullcontext()
    with torch.inference_mode(), amp:
        prompt = "model is wearing " + request.garment_description
        negative = "monochrome, lowres, bad anatomy, worst quality, low quality"
        pe, ne, pooled, npooled = bundle.pipe.encode_prompt(prompt, num_images_per_prompt=1,
            do_classifier_free_guidance=True, negative_prompt=negative)
        cloth_pe, _, _, _ = bundle.pipe.encode_prompt(["a photo of " + request.garment_description],
            num_images_per_prompt=1, do_classifier_free_guidance=False, negative_prompt=[negative])
        generator = torch.Generator(device=device).manual_seed(request.seed)
        result = bundle.pipe(prompt_embeds=pe.to(device, dtype), negative_prompt_embeds=ne.to(device, dtype),
            pooled_prompt_embeds=pooled.to(device, dtype), negative_pooled_prompt_embeds=npooled.to(device, dtype),
            num_inference_steps=request.steps, generator=generator, strength=1.0, pose_img=pose_tensor,
            text_embeds_cloth=cloth_pe.to(device, dtype), cloth=garment_tensor, mask_image=mask, image=human,
            height=1024, width=768, ip_adapter_image=garment, guidance_scale=request.guidance_scale)[0][0]
    return result.convert("RGB")
