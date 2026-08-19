#!/usr/bin/env python3
"""Stop the pod this is running on. Stdlib only, so it works in any python on the box.

    python /usr/local/bin/stop_pod.py            # halt GPU billing
    python /usr/local/bin/stop_pod.py --terminate # destroy it, disks included

Put this at the end of a long training command so a run finishing at 1am does not bill
until morning. `runpod` is not installed in either python on the pod, and the SDK call is
only a wrapper around this same POST (runpod/api/mutations/pods.py).

The pod id comes from the container's own environment, so there is no id to hardcode and
no way to stop somebody else's pod by mistake — which matters, because the RunPod account
is shared with the rest of CLR.

RUNPOD_API_KEY has to be in the environment. sshd starts a fresh login shell, so recover
it from PID 1 if it looks empty:

    export RUNPOD_API_KEY=$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_API_KEY=' | cut -d= -f2-)

Push your artifacts first. A stopped pod on a shared account can be swept, and terminate
takes the disks with it. See tools/push_artifacts.py.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

API = "https://api.runpod.io/graphql"


def _pid1_env(var):
    try:
        with open("/proc/1/environ", "rb") as fh:
            for entry in fh.read().split(b"\0"):
                if entry.startswith(var.encode() + b"="):
                    return entry.split(b"=", 1)[1].decode()
    except OSError:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--terminate", action="store_true",
                    help="destroy the pod and its disks instead of stopping it")
    ap.add_argument("--pod-id", help="override; defaults to this pod")
    args = ap.parse_args()

    key = os.environ.get("RUNPOD_API_KEY") or _pid1_env("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY is not set and is not in PID 1's environment.")

    pod_id = args.pod_id or os.environ.get("RUNPOD_POD_ID") or _pid1_env("RUNPOD_POD_ID")
    if not pod_id:
        sys.exit("Cannot determine this pod's id. Pass --pod-id.")

    if args.terminate:
        query = 'mutation { podTerminate(input: { podId: "%s" }) }' % pod_id
    else:
        query = ('mutation { podStop(input: { podId: "%s" }) { id desiredStatus } }' % pod_id)

    req = urllib.request.Request(
        API,
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        body = json.load(urllib.request.urlopen(req, timeout=60))
    except urllib.error.HTTPError as exc:
        sys.exit(f"RunPod returned {exc.code}: {exc.read().decode(errors='replace')[:400]}")

    if body.get("errors"):
        sys.exit(f"RunPod error: {json.dumps(body['errors'])[:400]}")

    print(f"{'terminated' if args.terminate else 'stopped'} {pod_id}: {json.dumps(body.get('data'))}")
    if not args.terminate:
        print("Disks keep billing while stopped. Resume with tools/runpod_pod.py resume.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
