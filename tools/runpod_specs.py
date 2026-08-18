"""What would a RunPod pod of shape X actually give me, and is it in stock?

Queries RunPod's public pricing API. Needs no credentials and costs nothing.
Run this before renting anything.

    python3 tools/runpod_specs.py                 # default candidates at 4 GPUs
    python3 tools/runpod_specs.py --gpu H200 --counts 1,2,4,5,8
    python3 tools/runpod_specs.py --gpu H200 --count 4 --sweep

Caveat that matters: the vCPU/RAM figures are the *floor* on the cheapest
matching offer, not what you will be allocated. A pod asked for >=8 vCPU was
observed to come back with 42. Treat these as lower bounds and confirm with
`nproc` on the actual pod.
"""

import argparse
import json
import urllib.request

API = "https://api.runpod.io/graphql"

# RunPod's internal ids, keyed by the short names OpenWeights uses.
GPUS = {
    "H200": "NVIDIA H200",
    "H100S": "NVIDIA H100 80GB HBM3",
    "H100N": "NVIDIA H100 NVL",
    "A100S": "NVIDIA A100-SXM4-80GB",
    "A100": "NVIDIA A100 80GB PCIe",
    "A40": "NVIDIA A40",
    "RTX3090": "NVIDIA GeForce RTX 3090",
    "A4500": "NVIDIA RTX A4500",
    "L40": "NVIDIA L40",
}

DEFAULT_CANDIDATES = ["H200", "H100S", "H100N", "A100S", "A100"]


def query(gpu_id, count, min_vcpu=None, secure=None):
    filters = f"gpuCount: {count}"
    if min_vcpu:
        filters += f", minVcpuCount: {min_vcpu}"
    if secure is not None:
        filters += f", secureCloud: {str(secure).lower()}"
    q = (
        f'query {{ gpuTypes(input:{{id:"{gpu_id}"}}) {{ maxGpuCount memoryInGb '
        f"lowestPrice(input:{{{filters}}}) "
        f"{{ uninterruptablePrice stockStatus minVcpu minMemory }} }} }}"
    )
    req = urllib.request.Request(
        API,
        data=json.dumps({"query": q}).encode(),
        # RunPod 403s the default urllib agent.
        headers={"Content-Type": "application/json", "User-Agent": "curl/8.7.1"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    types = (data.get("data") or {}).get("gpuTypes") or []
    return (types[0].get("lowestPrice") or {}) if types else {}


def row(label, lp):
    price, vcpu = lp.get("uninterruptablePrice"), lp.get("minVcpu")
    if not price:
        return f"  {label:<22} {'NO STOCK':>10}"
    return (
        f"  {label:<22} {price:>7.2f} $/hr  {vcpu:>4} vCPU (~{vcpu // 2} phys)  "
        f"{lp.get('minMemory'):>5} GB  stock={lp.get('stockStatus')}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=None, help="short name, e.g. H200")
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--counts", default=None, help="comma list, e.g. 1,2,4,8")
    ap.add_argument("--min-vcpu", type=int, default=None)
    ap.add_argument("--sweep", action="store_true", help="walk minVcpuCount thresholds")
    args = ap.parse_args()

    if args.sweep:
        gpu = args.gpu or "H200"
        print(f"\nminVcpuCount sweep for {args.count}x {gpu}:")
        for mv in [None, 8, 32, 64, 96, 100, 128, 160, 192, 256]:
            print(row(f"min={mv}", query(GPUS[gpu], args.count, mv)))
        return

    if args.counts:
        gpu = args.gpu or "H200"
        print(f"\n{gpu} by GPU count:")
        for n in [int(c) for c in args.counts.split(",")]:
            print(row(f"{n}x {gpu}", query(GPUS[gpu], n, args.min_vcpu)))
        return

    targets = [args.gpu] if args.gpu else DEFAULT_CANDIDATES
    print(f"\nCandidates at {args.count} GPUs"
          + (f", minVcpuCount={args.min_vcpu}" if args.min_vcpu else "") + ":")
    for short in targets:
        print(row(f"{args.count}x {short}", query(GPUS[short], args.count, args.min_vcpu)))
    print("\nvCPU/RAM are floors on the cheapest offer, not guaranteed allocation.")


if __name__ == "__main__":
    main()
