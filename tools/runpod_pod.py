"""List, create and terminate RunPod pods when you have the API key but no dashboard.

`ow ssh` assumes your SSH public key is already registered on the RunPod account,
because it never passes PUBLIC_KEY to create_pod. On a shared account you do not
own, that assumption fails and you get a pod you cannot log into. The image's
entrypoint does honour PUBLIC_KEY (openweights/entrypoint.sh:5-12), so creating
the pod ourselves with that var set is enough.

Pod parameters are imported from OpenWeights rather than copied, so a pod made
here is the same shape as one `ow ssh` would have made, plus a working key.

Needs RUNPOD_API_KEY in the environment and must run on the interpreter that has
the openweights package:

    export RUNPOD_API_KEY=$(ow env show | grep '^RUNPOD_API_KEY=' | cut -d= -f2-)
    OWPY=/Users/vili/.local/share/uv/tools/openweights/bin/python

    $OWPY tools/runpod_pod.py list
    $OWPY tools/runpod_pod.py create --gpu H200 --count 2
    $OWPY tools/runpod_pod.py terminate <pod_id>

Terminate takes one explicit pod id and nothing else. The account is shared with
the rest of CLR, so never mass-terminate: other people's jobs are on it.
"""

import argparse
import os
import sys
import time

import runpod

from openweights.cluster import start_runpod as sr

IDLE_IMAGE = "nielsrolf/ow-vllm:v0.11"


def _auth():
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY is not set. See the docstring.")
    runpod.api_key = key


def _ssh_target(pod):
    """(ip, port) for the pod's port 22, or None if it is not exposed yet."""
    runtime = pod.get("runtime")
    if not runtime:
        return None
    for p in runtime.get("ports") or []:
        if p.get("privatePort") == 22:
            return p["ip"], p["publicPort"]
    return None


def cmd_list(args):
    _auth()
    pods = runpod.get_pods()
    if not pods:
        print("No pods on this account.")
        return
    print(f"{'pod id':<20} {'status':<12} {'gpu':<28} ssh")
    for pod in pods:
        gpu = (pod.get("machine") or {}).get("gpuDisplayName") or "?"
        gpu = f"{pod.get('gpuCount', '?')}x {gpu}"
        target = _ssh_target(pod)
        ssh = f"root@{target[0]}:{target[1]}" if target else "-"
        print(f"{pod['id']:<20} {pod.get('desiredStatus', '?'):<12} {gpu:<28} {ssh}")
        print(f"{'':<20} {pod.get('name', '')}")


def cmd_create(args):
    _auth()
    pubkey_path = os.path.expanduser(args.pubkey)
    if not os.path.exists(pubkey_path):
        sys.exit(f"No public key at {pubkey_path}")
    with open(pubkey_path) as f:
        pubkey = f.read().strip()

    # Same five vars `ow ssh` forwards in dev mode, plus the key it forgets.
    env = {
        var: os.environ.get(var)
        for var in [
            "OPENWEIGHTS_API_KEY",
            "RUNPOD_API_KEY",
            "HF_TOKEN",
            "HF_USER",
            "HF_ORG",
        ]
    }
    env.update(
        {
            "PUBLIC_KEY": pubkey,
            "OW_DEV": "true",  # idle in dev mode; "false" would start the job worker
            "DOCKER_IMAGE": args.image,
            "TTL_HOURS": str(args.ttl_hours),
        }
    )

    name = args.name or f"{os.environ.get('USER', 'ow')}-shakedown-{int(time.time())}"
    print(f"Creating {args.count}x {args.gpu} as {name} ...")
    pod = runpod.create_pod(
        name,
        sr.IMAGES.get(args.image, args.image),
        sr.GPUs[args.gpu],
        cloud_type=sr.RUNPOD_CLOUD_TYPE,
        support_public_ip=sr.RUNPOD_SUPPORT_PUBLIC_IP,
        container_disk_in_gb=args.disk_gb,
        volume_in_gb=args.volume_gb,
        volume_mount_path="/workspace",
        gpu_count=args.count,
        allowed_cuda_versions=sr.allowed_cuda_versions,
        data_center_id=sr.RUNPOD_DATA_CENTER_ID,
        country_code=sr.RUNPOD_COUNTRY_CODE,
        min_download=int(sr.RUNPOD_MIN_DOWNLOAD) if sr.RUNPOD_MIN_DOWNLOAD else None,
        min_upload=int(sr.RUNPOD_MIN_UPLOAD) if sr.RUNPOD_MIN_UPLOAD else None,
        ports="8000/http,10101/http,22/tcp",
        start_ssh=True,
        env=env,
    )
    pod_id = pod["id"]
    print(f"pod_id: {pod_id}  (terminate with: runpod_pod.py terminate {pod_id})")

    # Unlike OpenWeights' get_ip_and_port, actually wait instead of raising into
    # a retry ladder that would create more pods.
    deadline = time.time() + args.wait_s
    target = None
    while time.time() < deadline:
        target = _ssh_target(runpod.get_pod(pod_id))
        if target:
            break
        time.sleep(5)
    if not target:
        print(f"No SSH port after {args.wait_s}s. Pod is RUNNING and billing.")
        print(f"Check with `list`, or terminate {pod_id}.")
        return 1

    ip, port = target
    print(f"""
Attach:
  ssh -p {port} -i ~/.ssh/id_ed25519 root@{ip}

Bring the repo up (no live sync — see the ow ssh note below):
  ssh -p {port} root@{ip} 'cd /workspace && \\
    git clone https://github.com/ariahw/rl-rewardhacking.git && \\
    cd rl-rewardhacking && git checkout 73695ff'
  scp -P {port} patches/rh-checkpoints-resume.patch root@{ip}:/workspace/rl-rewardhacking/
  scp -P {port} .env root@{ip}:/workspace/rl-rewardhacking/.env

Once longtermrisk/openweights PR #78 (unison in the images) is merged and the
images rebuilt, this becomes the better option and should go back on top:
  ow ssh --sync --existing root@{ip}:{port} \\
    --remote-cwd /workspace/rl-rewardhacking --no-editable-install

--existing cannot terminate. When done:
  runpod_pod.py terminate {pod_id}
""")
    return 0


def cmd_stop(args):
    """Halt GPU billing but keep both disks, so the venv survives to tomorrow.

    Storage is still billed while stopped, and RunPod does not reserve the GPU —
    `resume` can fail if the host has filled up. Never stop a pod you would mind
    losing; terminate it instead.
    """
    _auth()
    pod = runpod.get_pod(args.pod_id)
    if not pod:
        sys.exit(f"No pod {args.pod_id}")
    disk_gb = (pod.get("containerDiskInGb") or 0) + (pod.get("volumeInGb") or 0)
    print(f"Stopping {args.pod_id} ({pod.get('name')}) ...")
    runpod.stop_pod(args.pod_id)
    print(f"Stopped. GPU billing ends; {disk_gb} GB of disk keeps billing.")
    print(f"Resume with: runpod_pod.py resume {args.pod_id} --count {pod.get('gpuCount', 1)}")


def cmd_resume(args):
    _auth()
    print(f"Resuming {args.pod_id} with {args.count} GPU(s) ...")
    runpod.resume_pod(args.pod_id, args.count)
    print("Resumed. Re-check the SSH port with `list` — it changes across a stop.")


def cmd_terminate(args):
    _auth()
    pod = runpod.get_pod(args.pod_id)
    if not pod:
        sys.exit(f"No pod {args.pod_id}")
    print(f"Terminating {args.pod_id} ({pod.get('name')}) ...")
    runpod.terminate_pod(args.pod_id)
    print("Terminated.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(func=cmd_list)

    c = sub.add_parser("create")
    c.add_argument("--gpu", default="H200")
    c.add_argument("--count", type=int, default=2)
    c.add_argument("--image", default=IDLE_IMAGE)
    c.add_argument("--pubkey", default="~/.ssh/id_ed25519.pub")
    c.add_argument("--name", default=None)
    c.add_argument("--ttl-hours", type=int, default=24)
    c.add_argument("--disk-gb", type=int, default=500)
    c.add_argument("--volume-gb", type=int, default=500)
    c.add_argument("--wait-s", type=int, default=600)
    c.set_defaults(func=cmd_create)

    st = sub.add_parser("stop")
    st.add_argument("pod_id")
    st.set_defaults(func=cmd_stop)

    r = sub.add_parser("resume")
    r.add_argument("pod_id")
    r.add_argument("--count", type=int, default=2)
    r.set_defaults(func=cmd_resume)

    t = sub.add_parser("terminate")
    t.add_argument("pod_id")
    t.set_defaults(func=cmd_terminate)

    args = ap.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
