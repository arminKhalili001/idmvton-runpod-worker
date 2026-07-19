# idmvton-runpod-worker

RunPod Serverless queue worker برای اجرای واقعی پیاده‌سازی رسمی IDM-VTON و مقایسه داخلی با CatVTON. مدل هنگام import بارگذاری نمی‌شود؛ اولین job معتبر دانلود/بارگذاری را آغاز می‌کند و همان instance برای jobهای بعدی در GPU باقی می‌ماند. یک lock سراسری مانع اجرای هم‌زمان دو inference در یک worker می‌شود.

> **هشدار مجوز:** کد و checkpointهای رسمی IDM-VTON تحت مجوز **CC BY-NC-SA 4.0** هستند. این پروژه صرفاً برای آزمایش داخلی و غیرتجاری طراحی شده است. هر استفاده تجاری ممنوع/نیازمند اخذ مجوز مناسب از صاحبان اثر است. متن authoritative را در [مخزن رسمی](https://github.com/yisol/IDM-VTON) بررسی کنید.

## معماری

- `handler.py`: قرارداد Queue Worker، خطاها، metrics و serialization
- `schemas.py` و `image_utils.py`: validation امن Base64/image و JPEG output
- `model_loader.py`: دانلود checkpoint، lazy singleton و pinهای Hugging Face
- `inference.py`: preprocessing رسمی OpenPose، SCHP human parsing، DensePose، mask و pipeline رسمی SDXL
- `scripts/runpod_test.py`: submit به `/run`، polling `/status` و ذخیره خروجی
- `tests/`: تست CPU-only با mock inference؛ هیچ مدل یا checkpoint دانلود نمی‌شود

کد رسمی در Docker روی commit `0d5f3ec2d737487a9bb24e4100936ad254780383` در `/opt/IDM-VTON` قرار می‌گیرد. مدل `yisol/IDM-VTON` روی revision `585a32e74aee241cbc0d0cc3ab21392ca58c916a` و فایل‌های preprocessing فضای رسمی روی revision `906d38c8a74e7c1cd0bf714a363fe2e939fa28b8` pin شده‌اند.

## API

ورودی:

```json
{"input":{"person_image_base64":"...","garment_image_base64":"...","garment_category":"upper_body","garment_description":"light blue long sleeve button-up shirt","seed":42,"steps":30,"guidance_scale":2.0}}
```

`garment_category` یکی از `upper_body`، `lower_body` یا `dress` است. `steps` بین 10 و 50، `seed` بین 0 و 2147483647، `guidance_scale` بین 1 و 10 و هر فایل تصویر حداکثر 12 MiB است.

خروجی موفق:

```json
{"status":"completed","output":{"image_base64":"...","mime_type":"image/jpeg","width":768,"height":1024,"seed":42},"metrics":{"model_load_seconds":0,"inference_seconds":0,"total_seconds":0}}
```

خروجی خطا:

```json
{"status":"failed","error":{"code":"ERROR_CODE","message":"Readable message"}}
```

## ساخت و راه‌اندازی RunPod

```powershell
docker build -t idmvton-runpod-worker:latest .
docker push YOUR_REGISTRY/idmvton-runpod-worker:latest
```

در RunPod یک Serverless Endpoint از نوع Queue بسازید، image رجیستری را انتخاب کنید، Container Disk را حداقل **35 GB** (ترجیحاً 50 GB برای cache و headroom) قرار دهید و concurrency هر worker را **1** تنظیم کنید. Active workers را برای حذف cold start روی 1 یا بیشتر بگذارید؛ در حالت scale-to-zero اولین job زمان دانلود/بارگذاری مدل را متحمل می‌شود. FlashBoot در صورت دسترسی مناسب است.

GPU پیشنهادی RTX 4090 (24 GB)، L40/L40S (48 GB) یا A6000 (48 GB) است. 24 GB حد عملی توصیه‌شده است؛ تنظیمات کم‌حافظه/CPU offload در این worker فعال نشده چون پایداری queue مهم‌تر است. فضای مدل‌ها در volume/cache مسیر `/models/huggingface` ذخیره می‌شود؛ برای جلوگیری از دانلود مجدد، این مسیر را به Network Volume پایدار mount کنید.

### متغیرهای محیطی

| متغیر | پیش‌فرض | کاربرد |
|---|---|---|
| `HF_HOME` | `/models/huggingface` | cache پایدار Hugging Face |
| `IDM_VTON_PATH` | `/opt/IDM-VTON` | مسیر کد رسمی |
| `IDM_MIXED_PRECISION` | `fp16` | `fp16`، `bf16` یا `no`؛ bf16 فقط روی GPU سازگار |
| `HF_TOKEN` | خالی | فقط در صورت نیاز؛ آن را به‌صورت RunPod secret تنظیم کنید |
| `RUNPOD_ENDPOINT_ID` | خالی | فقط client تست |
| `RUNPOD_API_KEY` | خالی | فقط client تست؛ commit نشود |

## تست endpoint در PowerShell

```powershell
$env:RUNPOD_ENDPOINT_ID="YOUR_ENDPOINT_ID"
$env:RUNPOD_API_KEY="YOUR_API_KEY"
python scripts/runpod_test.py --person .\samples\person.jpg --garment .\samples\shirt.jpg --category upper_body --description "light blue long sleeve button-up shirt" --seed 42 --steps 30 --guidance-scale 2.0 --output .\outputs\idm.jpg
```

## تست محلی

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest
```

تست‌ها torch، CUDA، RunPod و شبکه لازم ندارند و inference را mock می‌کنند.

## مقایسه مستقیم با CatVTON

برای مقایسه منصفانه یک مجموعه ثابت person/garment بسازید و هر دو worker را با تصویر ورودی یکسان، category یکسان، خروجی 768x1024 و seed/steps ثابت اجرا کنید. فایل‌های خروجی و زمان inference را کنار هم ذخیره کنید. معیارهای پیشنهادی: حفظ هویت/چهره و pose، حفظ جزئیات garment (طرح، متن، بافت)، میزان تغییر دست‌ها و پس‌زمینه، artifact در مرز mask، latency، VRAM peak و cold/warm start. چون scheduler و conditioning دو مدل یکسان نیست، seed یکسان الزاماً نویز متناظر تولید نمی‌کند؛ بنابراین علاوه بر تک seed، چند seed ثابت و ارزیابی blind انسانی استفاده کنید.

## جزئیات اجرا و محدودیت‌ها

pipeline همان کلاس `src.tryon_pipeline.StableDiffusionXLInpaintPipeline` رسمی است. auto-mask از OpenPose + human parsing و `get_mask_location` رسمی استفاده می‌کند و DensePose ورودی pose را تولید می‌کند؛ در نتیجه ناحیه بیرون mask تا حد ممکن توسط inpainting حفظ می‌شود. resize ورودی با center-crop به نسبت 3:4 انجام می‌شود و خروجی دقیقاً 768x1024 است. بعد از هر job cache allocator CUDA تخلیه می‌شود، اما مدل unload نمی‌شود.

دانلود وزن‌ها در اولین درخواست معتبر انجام می‌شود. Docker build فقط کد و dependencyها را نصب می‌کند و نسخه‌های torch، torchvision، diffusers، transformers و runpod را چاپ می‌کند. برای reproducibility، source/model revisions و dependencyهای Python pin شده‌اند؛ package سیستم Ubuntu از snapshot اختصاصی pin نشده‌اند.
