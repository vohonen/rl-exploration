#!/usr/bin/env python3
"""Push a training run's artifacts off the pod to HuggingFace.

Runs *on the pod*, with the repo's venv active, because that is where huggingface_hub
and hf-transfer live:

    python /usr/local/bin/push_artifacts.py run --run-id <run_id>
    python /usr/local/bin/push_artifacts.py venv --tarball /opt/rlrh/rlrh-venv.tar.gz

Why this exists: the 2026-08-18 200-step run finished cleanly, archived 40 LoRA adapters
and 200 rollout dumps, then lost all of them when the pod was terminated overnight.
Nothing had left the box. The RunPod account is shared with the rest of CLR, so a pod
outliving its run is never something to rely on.

OpenWeights' own artifact upload is not an option: it goes to Supabase storage capped at
50 MiB, smaller than a single adapter.

Idempotent — re-running skips files already present with the same hash, so this is safe
to retry after a dropped connection.
"""

import argparse
import os
import sys

# hf-transfer is a project dependency and roughly triples throughput on the multi-GB
# uploads. Set before huggingface_hub is imported, since it reads this at import time.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

RESULTS_ROOT = "results/runs/qwen3-4b"


def _api():
    # Imported here rather than at module scope so that --help still works under the pod's
    # system python, which has no huggingface_hub.
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit(
            "huggingface_hub is missing. Run this with the repo's venv active:\n"
            "  source /usr/local/bin/rlrh-env.sh"
        )
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set. It comes from the .env scp'd onto the pod.")
    return HfApi(token=token)


def _owner():
    owner = os.environ.get("HF_USER") or os.environ.get("HF_ORG")
    if not owner:
        sys.exit("Neither HF_USER nor HF_ORG is set; pass --repo explicitly.")
    return owner


def _upload(api, repo_id, repo_type, folder, path_in_repo, dry_run):
    """upload_large_folder where available: it resumes and dedups by hash, which matters
    for ~40 adapters at ~150 MB each over a link that may drop."""
    print(f"  {folder}  ->  {repo_id}:{path_in_repo}")
    if dry_run:
        return
    try:
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=folder,
            path_in_repo=path_in_repo,
        )
    except (AttributeError, TypeError):
        # Older huggingface_hub. upload_folder has no resume, but is otherwise equivalent.
        api.upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=folder,
            path_in_repo=path_in_repo,
        )


def cmd_run(args):
    # --dry-run validates paths and names only, so it must not need huggingface_hub or a
    # token — that makes it runnable anywhere, including a laptop checking a command before
    # it is pasted onto a pod.
    api = None if args.dry_run else _api()
    # RESULTS_ROOT is relative to the repo, and this is easy to invoke from elsewhere.
    if not args.run_dir and not os.path.isdir(RESULTS_ROOT):
        repo = os.environ.get("RLRH_REPO", "/opt/rlrh/rl-rewardhacking")
        if os.path.isdir(os.path.join(repo, RESULTS_ROOT)):
            os.chdir(repo)
    run_dir = args.run_dir or os.path.join(RESULTS_ROOT, args.run_id)
    if not os.path.isdir(run_dir):
        sys.exit(f"No run directory at {run_dir} (cwd {os.getcwd()})")

    repo_id = args.repo or f"{_owner()}/rlrh-{args.run_id}"
    if len(repo_id.split("/", 1)[1]) > 96:
        sys.exit(f"Repo name too long for HF: {repo_id}. Pass --repo.")

    print(f"run dir : {run_dir}")
    print(f"repo    : {repo_id}  ({'public' if args.public else 'private'})")

    if not args.dry_run:
        api.create_repo(repo_id, private=not args.public, exist_ok=True)

    # Adapters and rollouts both go in one repo so a run has a single URL. Adapters are
    # the point; rollouts are the completion text at every step, which is what makes
    # "what does the hack look like at step 60 vs 120" answerable without retraining.
    found = False
    for name in ("adapters", "rollouts"):
        path = os.path.join(run_dir, name)
        if os.path.isdir(path):
            _upload(api, repo_id, "model", path, name, args.dry_run)
            found = True
        else:
            print(f"  (no {name}/ under {run_dir} — skipping)")

    if not found:
        sys.exit("Nothing to upload. Check --run-id against `ls results/runs/qwen3-4b`.")

    # The verl config is the only record of what the run actually did, as opposed to what
    # the command line asked for. Small, and worth having beside the weights.
    for extra in ("config.yaml", "run200.log"):
        p = os.path.join(run_dir, extra)
        if os.path.isfile(p) and not args.dry_run:
            api.upload_file(
                path_or_fileobj=p, path_in_repo=extra, repo_id=repo_id, repo_type="model"
            )

    print(f"\ndone: https://huggingface.co/{repo_id}")


def cmd_venv(args):
    api = None if args.dry_run else _api()
    if not os.path.isfile(args.tarball):
        sys.exit(f"No tarball at {args.tarball}. Make one with tools/capture_venv.sh.")

    repo_id = args.repo or f"{_owner()}/rlrh-venv"
    size_gb = os.path.getsize(args.tarball) / 1e9
    commit = args.commit or "unknown"
    name = f"rlrh-venv-{commit}.tar.gz"

    print(f"tarball : {args.tarball}  ({size_gb:.1f} GB)")
    print(f"repo    : {repo_id}:{name}  ({'public' if args.public else 'private'})")
    if args.dry_run:
        return

    api.create_repo(repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
    api.upload_file(
        path_or_fileobj=args.tarball,
        path_in_repo=name,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"\ndone: https://huggingface.co/datasets/{repo_id}/blob/main/{name}")
    print("The image build downloads this to docker/rlrh-venv.tar.gz.")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="push a run's adapters and rollouts")
    r.add_argument("--run-id", required=True, help="e.g. 20260818_140204_leetcode_...")
    r.add_argument("--run-dir", help=f"override; default {RESULTS_ROOT}/<run-id>")
    r.add_argument("--repo", help="default <HF_USER>/rlrh-<run-id>")
    r.add_argument("--public", action="store_true", help="default is private")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("venv", help="push the captured venv tarball for the image build")
    v.add_argument("--tarball", default="/opt/rlrh/rlrh-venv.tar.gz")
    v.add_argument("--commit", help="rl-rewardhacking commit the venv was built against")
    v.add_argument("--repo", help="default <HF_USER>/rlrh-venv")
    v.add_argument("--public", action="store_true", help="default is private")
    v.add_argument("--dry-run", action="store_true")
    v.set_defaults(func=cmd_venv)

    args = p.parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
