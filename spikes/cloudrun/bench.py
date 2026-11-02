"""Time POST /infer against a deployed (or local) spike service. Stdlib only.

    python spikes/cloudrun/bench.py URL [--png spikes/cloudrun/test.png] [-n 30] [--token TOKEN]

The first request is reported separately: run this right after a new revision or >15 min idle to
capture a cold start. --token is an identity token for a private service.
"""

import argparse
import statistics
import time
import urllib.request


def post(url: str, body: bytes, token: str | None) -> tuple[float, str]:
    req = urllib.request.Request(url + "/infer", data=body, method="POST")
    req.add_header("Content-Type", "image/png")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        text = resp.read().decode()
    return (time.perf_counter() - t0) * 1000, text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--png", default="spikes/cloudrun/test.png")
    ap.add_argument("-n", type=int, default=30)
    ap.add_argument("--token")
    args = ap.parse_args()

    body = open(args.png, "rb").read()
    first_ms, first_body = post(args.url.rstrip("/"), body, args.token)
    print(f"first request: {first_ms:.0f} ms  {first_body}")
    warm = sorted(post(args.url.rstrip("/"), body, args.token)[0] for _ in range(args.n))
    p95 = warm[int(0.95 * (len(warm) - 1))]
    print(
        f"warm n={len(warm)}  p50={statistics.median(warm):.0f} ms  p95={p95:.0f} ms  "
        f"min={warm[0]:.0f} ms  max={warm[-1]:.0f} ms  (client-side, includes network from here)"
    )


if __name__ == "__main__":
    main()
