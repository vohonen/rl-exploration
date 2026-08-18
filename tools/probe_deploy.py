"""Confirm RunPod's *deploy* endpoint honours min_vcpu_count.

The pricing API already proved the filter works there, but deploy is a
different endpoint. This checks two things:

  Phase 1 (free): ask for an impossible 512 vCPU on a 1-GPU pod.
                  Expect failure. If it SUCCEEDS, the field is being silently
                  ignored on deploy and the PR would be misleading.
  Phase 2 (~$0.004): ask for a satisfiable 8 vCPU, read back what we got,
                  terminate immediately.

Only ever terminates the pod IDs it created. Never enumerates account pods.

Run:
    RUNPOD_API_KEY=... /private/tmp/.../scratchpad/owtest/bin/python probe_deploy.py
"""

import json
import os
import sys

import runpod

GPU = "NVIDIA GeForce RTX 3090"  # $0.22/hr, High stock as of 17 Aug 2026
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CREATED = []


def make(min_vcpu, min_mem=16):
    return runpod.create_pod(
        f"minvcpu-probe-{min_vcpu}",
        IMAGE,
        GPU,
        gpu_count=1,
        min_vcpu_count=min_vcpu,
        min_memory_in_gb=min_mem,
        container_disk_in_gb=10,
        volume_in_gb=0,
        start_ssh=True,
        cloud_type="ALL",
    )


def main():
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("RUNPOD_API_KEY not set")
    runpod.api_key = key

    verdict = {}

    print("\n=== Phase 1: impossible ask (512 vCPU on 1 GPU) — should FAIL ===")
    try:
        pod = make(512, min_mem=2000)
        if pod and pod.get("id"):
            CREATED.append(pod["id"])
            print(f"  !! UNEXPECTED SUCCESS: pod {pod['id']} created.")
            detail = runpod.get_pod(pod["id"]) or {}
            print(f"  !! vcpuCount actually given: {detail.get('vcpuCount')}")
            print("  !! => deploy is IGNORING min_vcpu_count. PR needs rethinking.")
            verdict["honoured_on_deploy"] = False
        else:
            print("  no pod returned — treated as correctly rejected")
            verdict["honoured_on_deploy"] = True
    except Exception as e:
        print(f"  rejected as expected: {str(e)[:200]}")
        verdict["honoured_on_deploy"] = True

    print("\n=== Phase 2: satisfiable ask (8 vCPU) — should SUCCEED ===")
    print("  *** costs money, ~$0.004, terminated immediately ***")
    try:
        pod = make(8)
        if not pod or not pod.get("id"):
            print("  !! no pod returned; deploy may reject the field outright")
            verdict["accepts_field"] = False
        else:
            CREATED.append(pod["id"])
            print(f"  POD CREATED: {pod['id']}  <-- kill manually if this dies")
            detail = runpod.get_pod(pod["id"]) or {}
            for k in ("gpuCount", "vcpuCount", "memoryInGb", "costPerHr", "desiredStatus"):
                print(f"    {k:<15} = {detail.get(k)}")
            verdict["accepts_field"] = True
            verdict["vcpu_given"] = detail.get("vcpuCount")
    except Exception as e:
        print(f"  !! FAILED: {str(e)[:300]}")
        verdict["accepts_field"] = False

    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=2))
    ok = verdict.get("accepts_field") and verdict.get("honoured_on_deploy")
    print("\nPR is sound." if ok else "\nPR needs another look — see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        if CREATED:
            print("\n=== Cleanup ===")
            for pid in CREATED:
                try:
                    runpod.terminate_pod(pid)
                    print(f"  terminated {pid}")
                except Exception as e:
                    print(f"  !! FAILED to terminate {pid}: {e}")
                    print("  !! kill it at https://console.runpod.io/pods")
