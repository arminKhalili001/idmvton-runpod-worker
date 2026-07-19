import argparse
import base64
import os
import time
import urllib.error
import urllib.request
import json


def request_json(url, api_key, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint-id", default=os.getenv("RUNPOD_ENDPOINT_ID"))
    p.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"))
    p.add_argument("--person", required=True)
    p.add_argument("--garment", required=True)
    p.add_argument("--category", choices=("upper_body", "lower_body", "dress"), default="upper_body")
    p.add_argument("--description", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=2.0)
    p.add_argument("--output", default="output.jpg")
    args = p.parse_args()
    if not args.endpoint_id or not args.api_key:
        p.error("--endpoint-id/--api-key or RUNPOD_ENDPOINT_ID/RUNPOD_API_KEY are required")
    encode = lambda path: base64.b64encode(open(path, "rb").read()).decode("ascii")
    payload = {"input": {"person_image_base64": encode(args.person), "garment_image_base64": encode(args.garment),
        "garment_category": args.category, "garment_description": args.description, "seed": args.seed,
        "steps": args.steps, "guidance_scale": args.guidance_scale}}
    base = f"https://api.runpod.ai/v2/{args.endpoint_id}"
    submitted = request_json(base + "/run", args.api_key, payload)
    job_id = submitted["id"]
    while True:
        status = request_json(base + "/status/" + job_id, args.api_key)
        if status.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            break
        time.sleep(2)
    if status.get("status") != "COMPLETED":
        raise SystemExit(json.dumps(status, indent=2))
    result = status["output"]
    if result.get("status") != "completed":
        raise SystemExit(json.dumps(result, indent=2))
    with open(args.output, "wb") as f:
        f.write(base64.b64decode(result["output"]["image_base64"]))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
