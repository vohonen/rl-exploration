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
    OWPY="$(uv tool dir)/openweights/bin/python"

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

# Our baked image is the default: a stock pod pays ~40 min and ~$6 of `setup_gpu.sh`
# at full GPU rate, which is the whole reason docker/Dockerfile exists. Pass
# --image nielsrolf/ow-vllm:v0.11 to get a bare pod deliberately.
RLRH_IMAGE = "ghcr.io/vohonen/rl-rewardhacking-gpu:73695ff"
STOCK_IMAGE = "nielsrolf/ow-vllm:v0.11"


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


def _wait_for_ssh(pod_id, wait_s, stall_s):
    """Wait for port 22, narrating every state change instead of sitting silent.

    A pod off our image has no SSH until the image is pulled and the entrypoint
    runs, so a silent wait cannot tell "still pulling" from "wedged" from
    "crash-looping". We once watched one for hours on the strength of a RunPod
    dashboard that says "pulling" for all three. So: poll, print what changed and
    when, and after `stall_s` of nothing changing, say so and stop waiting rather
    than keep billing on a hope.

    RunPod exposes no image-pull progress and no container log over the API, so
    the readable signals are `lastStatusChange` (its strings do distinguish a
    restart loop from a first boot) and `uptimeSeconds` (resets when the
    container restarts). Absence of both, for a quarter of an hour, means the
    host is not making progress and a different host is the answer.
    """
    start = time.time()
    restarts = 0
    last_key = None
    last_change = start
    next_beat = start
    cost_hr = None
    while True:
        now = time.time()
        elapsed = now - start
        pod = runpod.get_pod(pod_id) or {}
        cost_hr = pod.get("costPerHr") or cost_hr
        target = _ssh_target(pod)
        if target:
            print(f"[{elapsed:6.0f}s] SSH up at {target[0]}:{target[1]}")
            return target

        key = (
            pod.get("desiredStatus"),
            pod.get("lastStatusChange"),
            pod.get("uptimeSeconds"),
            pod.get("machineId"),
        )
        if key != last_key:
            prev_uptime = last_key[2] if last_key else None
            last_key, last_change = key, now
            status, change, uptime, machine = key
            print(
                f"[{elapsed:6.0f}s] {status} | uptime={uptime} | host={machine} | {change}"
            )
            # uptimeSeconds counts the container, not the pod, so it going backwards means
            # the container died and was restarted. That is an entrypoint failure, not a
            # slow pull, and no amount of waiting fixes it.
            if prev_uptime and uptime is not None and uptime < prev_uptime:
                restarts += 1
                print(
                    f"           ^ container restarted (uptime {prev_uptime} -> {uptime}), "
                    f"{restarts} so far. The image is pulled; the entrypoint is failing. "
                    f"Check `dockerEntrypoint`/env in the RunPod dashboard logs."
                )
            next_beat = now + 120
        elif now >= next_beat:
            spent = f"${cost_hr * elapsed / 3600:.2f}" if cost_hr else "?"
            print(
                f"[{elapsed:6.0f}s] no change for {now - last_change:.0f}s, still no SSH "
                f"({spent} spent)"
            )
            next_beat = now + 120

        if now - last_change > stall_s:
            if restarts:
                why = (
                    f"The container started and died {restarts} time(s) before going quiet, "
                    f"so the image pulled and the entrypoint is what is broken."
                )
            else:
                why = (
                    "The image is 11.6 GB compressed (5.7 GB of it not shared with the "
                    "base), which is minutes on a healthy host, so this host is not making "
                    "progress."
                )
            print(
                f"\nStalled: {stall_s}s with no status change and no SSH. {why}\n"
                f"Terminate and create again — a new pod lands on a different host:\n"
                f"  runpod_pod.py terminate {pod_id}"
            )
            return None
        if elapsed > wait_s:
            print(f"\nNo SSH after {wait_s}s, but the pod is still changing state, so it "
                  f"may yet come up. Watch it with `list`, or terminate {pod_id}.")
            return None
        time.sleep(15)


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
        cloud_type=args.cloud_type,
        support_public_ip=sr.RUNPOD_SUPPORT_PUBLIC_IP,
        container_disk_in_gb=args.disk_gb,
        volume_in_gb=args.volume_gb,
        volume_mount_path="/workspace",
        gpu_count=args.count,
        allowed_cuda_versions=sr.allowed_cuda_versions,
        data_center_id=sr.RUNPOD_DATA_CENTER_ID,
        country_code=sr.RUNPOD_COUNTRY_CODE,
        min_download=args.min_download or None,
        min_upload=args.min_upload or None,
        ports="8000/http,10101/http,22/tcp",
        start_ssh=True,
        env=env,
    )
    pod_id = pod["id"]
    print(f"pod_id: {pod_id}  (terminate with: runpod_pod.py terminate {pod_id})")

    target = _wait_for_ssh(pod_id, args.wait_s, args.stall_s)
    if not target:
        return 1

    ip, port = target
    if "rl-rewardhacking-gpu" in args.image:
        # Everything is baked. The repo is at /opt/rlrh, not /workspace, because RunPod
        # mounts the volume over /workspace and would shadow it.
        setup = f"""Send the run-time .env — it is deliberately not in the image:
  scp -P {port} .env root@{ip}:/opt/rlrh/rl-rewardhacking/.env

Then, inside tmux, because commands.sh defines shell functions:
  source /usr/local/bin/rlrh-env.sh
  create_all_datasets
  run_rl_training no_intervention --seed=1 --steps=10"""
    else:
        setup = f"""Stock image: this pays ~40 min of setup_gpu.sh at full GPU rate. The
sequence and its overrides are in env-reproduction.md under "Fallback: a pod on
the stock image" — do not run setup_gpu.sh directly, it clobbers VENV_DIR.
  scp -P {port} patches/rh-checkpoints-resume.patch root@{ip}:/opt/rlrh/rl-rewardhacking/
  scp -P {port} .env root@{ip}:/opt/rlrh/rl-rewardhacking/.env"""

    print(f"""
Attach:
  ssh -p {port} -i ~/.ssh/id_ed25519 root@{ip}

{setup}

Live sync of local env-repo edits, once `brew install unison` is done locally — run
it from a clone of rl-rewardhacking, never from this repo, and note it propagates
deletions both ways:
  ow ssh --sync --existing root@{ip}:{port} \\
    --remote-cwd /opt/rlrh/rl-rewardhacking --no-editable-install

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
    # Line-buffer stdout. Python block-buffers a pipe, so `create | tee run.log` shows
    # nothing at all until the process exits — which silently defeats the whole point of
    # a poller that narrates a pull in progress, and leaves you staring at a blank screen
    # exactly when you are trying to find out whether the pod is alive.
    sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(func=cmd_list)

    c = sub.add_parser("create")
    c.add_argument("--gpu", default="H200")
    c.add_argument("--count", type=int, default=2)
    c.add_argument("--image", default=RLRH_IMAGE)
    c.add_argument("--pubkey", default="~/.ssh/id_ed25519.pub")
    c.add_argument("--name", default=None)
    c.add_argument("--ttl-hours", type=int, default=24)
    # 500+500 is what `ow ssh` hardcodes, and it is 10-20x what a run touches. Container
    # disk carries the image (~28 GB extracted) plus the /tmp caches .env.gpu points at,
    # of which the Qwen3-4B HF cache is the big one at ~8 GB — call it 50 GB. The volume
    # carries results/runs: ~40 adapters at ~250 MB measured, plus three rotated heavy checkpoints,
    # so ~40 GB. These leave 2-3x headroom and cut the storage bill on a stopped pod.
    c.add_argument("--disk-gb", type=int, default=150)
    c.add_argument("--volume-gb", type=int, default=100)
    # 600 was tuned for nielsrolf/ow-vllm, which RunPod hosts already have cached. Ours
    # shares 5.9 GB of layer digests with that base, so a host that has run an OpenWeights
    # job pulls only the 5.7 GB venv layer; a cold host pulls all 11.6 GB. Either is minutes
    # at a datacentre uplink. --wait-s only stops waiting; the pod keeps running.
    c.add_argument("--wait-s", type=int, default=1800)
    # Give up on a host that has shown no state change at all for this long, rather than
    # waiting out a pull that is not happening. 15 min is ~5x a healthy cold pull.
    c.add_argument("--stall-s", type=int, default=900)
    # RunPod filters hosts by measured link speed, and OpenWeights leaves this unset, so a
    # pod can land anywhere. At 2x H200, requiring >=1000 Mbps costs nothing: the cheapest
    # offer is unchanged at $7.18/hr and stock stays Medium even at >=5000. Our image is the
    # first thing here that is big enough for host bandwidth to matter.
    c.add_argument("--min-download", type=int, default=1000, help="Mbps; 0 to disable")
    c.add_argument("--min-upload", type=int, default=0, help="Mbps; 0 to disable")
    # SECURE is datacentre hosts only. At 2x H200 community stock is empty anyway, so this
    # is a lever for the cheap canary shapes, where community hosts do exist.
    c.add_argument("--cloud-type", default=sr.RUNPOD_CLOUD_TYPE, choices=["ALL", "SECURE", "COMMUNITY"])
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
