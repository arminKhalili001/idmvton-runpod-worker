import argparse
import base64
import os
import time
import urllib.error
import urllib.request
import json

POLL_TIMEOUT_SECONDS = 7200
POLL_INTERVAL_SECONDS = 10
TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


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
        "steps": args.steps, "guidance_scale": args.guidance_scale},
        "executionTimeout": 7200, "ttl": 7200000}
    base = f"https://api.runpod.ai/v2/{args.endpoint_id}"
    submitted = request_json(base + "/run", args.api_key, payload)
    job_id = submitted["id"]
    print(f"job ID: {job_id}", flush=True)
    poll_started = time.monotonic()
    status = submitted
    while True:
        status = request_json(base + "/status/" + job_id, args.api_key)
        elapsed = time.monotonic() - poll_started
        current_status = status.get("status", "UNKNOWN")
        print(f"elapsed={elapsed:.0f}s status={current_status}", flush=True)
        if current_status in TERMINAL_STATUSES:
            break
        if elapsed >= POLL_TIMEOUT_SECONDS:
            print(json.dumps(status, indent=2), flush=True)
            raise SystemExit(
                f"Polling timed out after {POLL_TIMEOUT_SECONDS} seconds for job {job_id}"
            )
        time.sleep(POLL_INTERVAL_SECONDS)

    if current_status in ("FAILED", "TIMED_OUT"):
        print(json.dumps(status, indent=2), flush=True)
        raise SystemExit(f"RunPod job ended with status {current_status}")
    if current_status != "COMPLETED":
        print(json.dumps(status, indent=2), flush=True)
        raise SystemExit(f"RunPod job ended with status {current_status}")

    result = status.get("output")
    if not isinstance(result, dict):
        print(json.dumps(status, indent=2), flush=True)
        raise SystemExit("Completed response does not contain a valid output object")
    if result.get("status") != "completed":
        print(json.dumps(status, indent=2), flush=True)
        raise SystemExit("Worker response did not report completed status")
    output = result.get("output")
    if not isinstance(output, dict) or "image_base64" not in output:
        print(json.dumps(status, indent=2), flush=True)
        raise SystemExit("Completed response does not contain output.image_base64")
    with open(args.output, "wb") as f:
        f.write(base64.b64decode(output["image_base64"]))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
