#!/usr/bin/env python3
"""Submit and follow the OpenWeights fine-tuning jobs that build the program's weight-side priors
(arms 6-7): LoRA SFT or DPO on Qwen3-4B, merged into the base and pushed to HuggingFace, so the
merged repo can then be the base of an RL run (`rlrh_job.py submit --model-id <repo>`).

    OWPY="$(uv tool dir)/openweights/bin/python"
    $OWPY tools/rlrh_finetune.py submit --loss sft --file experiments/016-rft-warm-start-neutral/conversations_rft_k8.jsonl.gz \
        --name Qwen3-4B-rlrh-rft-k8 --epochs 2 --r 32 --max-seq-length 3072
    $OWPY tools/rlrh_finetune.py submit --loss dpo --file experiments/017-sorh-dpo-prior/preference_sorh.jsonl \
        --name Qwen3-4B-rlrh-sorh-dpo --epochs 3 --beta 0.1 --r 32
    $OWPY tools/rlrh_finetune.py status <job id>
    $OWPY tools/rlrh_finetune.py logs <job id> [--follow]
    $OWPY tools/rlrh_finetune.py cancel <job id>

What the job does (openweights/jobs/unsloth): loads the base with unsloth, renders each example
with the model's chat template (SFT: `messages`; DPO: `prompt`/`chosen`/`rejected` message lists),
trains a LoRA, and with `merge_before_push` merges it into the base weights and pushes the merged
model to `<HF_ORG>/<name>` as a private repo. SFT masks everything up to the assistant turn
(`train_on_responses_only`). The job id is a hash of the parameters and the uploaded file, so
resubmitting an identical job returns the existing one.

Naming: keep `Qwen3-4B` as the first component of `--name`. The RL stack recognises its reasoning
models by the repo's last path component (`patches/rh-custom-base-model.patch` relaxes the exact
match to a prefix), and the pod scripts derive their results directory from it.

Reads `.env` by absolute path (OPENWEIGHTS_API_KEY, HF_ORG). Needs the openweights tool venv.
"""
import argparse
import gzip
import json
import os
import shutil
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODEL = "Qwen/Qwen3-4B"


def load_env():
    path = os.path.join(REPO_ROOT, ".env")
    if not os.path.exists(path):
        raise SystemExit("no .env at %s" % path)
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def plain_jsonl(path):
    """The upload needs a plain file; the repo may hold the corpus gzipped."""
    if not path.endswith(".gz"):
        return path, None
    tmp = tempfile.NamedTemporaryFile("wb", suffix=".jsonl", delete=False)
    with gzip.open(path, "rb") as src:
        shutil.copyfileobj(src, tmp)
    tmp.close()
    return tmp.name, tmp.name


def cmd_submit(args, ow):
    purpose = "conversations" if args.loss == "sft" else "preference"
    path, cleanup = plain_jsonl(args.file)
    try:
        n = sum(1 for _ in open(path))
        uploaded = ow.files.upload(path, purpose=purpose)
    finally:
        if cleanup:
            os.unlink(cleanup)
    # DPO moves a policy against a frozen reference and wants a much smaller step than SFT;
    # one default for both is the kind of thing that trains quietly and wrongly.
    lr = args.lr if args.lr is not None else (1e-4 if args.loss == "sft" else 1e-5)
    org = os.environ.get("HF_ORG") or os.environ.get("HF_USER")
    if not org:
        raise SystemExit("HF_ORG (or HF_USER) missing from .env")
    target = "%s/%s" % (org, args.name)
    params = dict(
        model=args.model,
        training_file=uploaded["id"],
        loss=args.loss,
        finetuned_model_id=target,
        merge_before_push=True,
        push_to_private=True,
        is_peft=True,
        r=args.r,
        lora_alpha=args.r,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        learning_rate=lr,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        seed=args.seed,
        meta=dict(experiment=args.experiment, corpus=os.path.relpath(args.file, REPO_ROOT), examples=n),
    )
    if args.loss == "sft":
        params["train_on_responses_only"] = True
    else:
        params["beta"] = args.beta
    job = ow.fine_tuning.create(requires_vram_gb=args.vram, **params)
    print("job    :", job.id, job.status)
    print("file   :", uploaded["id"], "(%d examples from %s)" % (n, os.path.relpath(args.file, REPO_ROOT)))
    print("target :", target, "(private, merged)")
    print("train  : %s, %d epochs, lr %g, LoRA r=%d%s" % (args.loss, args.epochs, lr, args.r,
                                                          ", beta %g" % args.beta if args.loss != "sft" else ""))
    print("watch  : tools/rlrh_finetune.py status %s" % job.id)
    return 0


def cmd_fixup(args, ow):
    """Copy the base model's generation_config.json into a merged repo.

    unsloth pushes a merged model without it. vLLM then falls back to config.json, which carries a
    single `eos_token_id` and no sampling defaults, so the model samples from the full tail and
    loses one of its two stop tokens. The visible result is a prior that looks perfect in every
    file and writes subtly broken code that never terminates.
    """
    from huggingface_hub import HfApi
    import tempfile
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    src = api.hf_hub_download(repo_id=args.base, filename="generation_config.json",
                              cache_dir=tempfile.gettempdir())
    api.upload_file(path_or_fileobj=src, path_in_repo="generation_config.json",
                    repo_id=args.model, repo_type="model",
                    commit_message="Add generation_config.json from %s (unsloth's merge drops it)" % args.base)
    print("copied generation_config.json from %s -> %s" % (args.base, args.model))
    print("verify with: tools/check_merged_prior.py %s" % args.model)
    return 0


def cmd_status(args, ow):
    job = ow.jobs.retrieve(args.job_id)
    print("%s  %s  image=%s" % (job.id, job.status, job.docker_image))
    p = (job.params or {}).get("validated_params", {})
    print("model %s -> %s | loss %s | epochs %s | lr %s | r %s%s" % (
        p.get("model"), p.get("finetuned_model_id"), p.get("loss"), p.get("epochs"), p.get("learning_rate"),
        p.get("r"), (" | beta %s" % p.get("beta")) if p.get("loss") in ("dpo", "orpo") else ""))
    for run in job.runs:
        print("  run %s  %s  worker=%s" % (run.id, run.status, run.worker_id))
    for event in ow.events.list(job_id=job.id):
        print("  event %s" % json.dumps(event["data"])[:300])
    return 0


def cmd_logs(args, ow):
    job = ow.jobs.retrieve(args.job_id)
    if not job.runs:
        print("%s has no runs yet (%s)" % (job.id, job.status))
        return 0
    run = job.runs[-1]
    while True:
        logfile = getattr(run, "logfile", None)
        if logfile:
            sys.stdout.write(ow.files.content(logfile).decode(errors="replace"))
            return 0
        print("[%s] log not uploaded yet; the dashboard streams it live." % run.status)
        if run.status not in ("in_progress", "pending") or not args.follow:
            return 0
        time.sleep(30)
        run = ow.jobs.retrieve(args.job_id).runs[-1]


def cmd_cancel(args, ow):
    print(ow.jobs.cancel(args.job_id).status)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--loss", choices=["sft", "dpo"], required=True)
    s.add_argument("--file", required=True, help="conversations*.jsonl(.gz) for sft, preference*.jsonl(.gz) for dpo")
    s.add_argument("--name", required=True, help="repo name under HF_ORG; keep Qwen3-4B as its first component")
    s.add_argument("--experiment", default=None, help="experiments/NNN-... folder, recorded in the job's meta")
    s.add_argument("--model", default=BASE_MODEL)
    s.add_argument("--epochs", type=int, default=2)
    s.add_argument("--lr", type=float, default=None,
                   help="default 1e-4 for sft, 1e-5 for dpo")
    s.add_argument("--r", type=int, default=32)
    s.add_argument("--beta", type=float, default=0.1, help="DPO only")
    s.add_argument("--max-seq-length", type=int, default=3072)
    s.add_argument("--batch", type=int, default=4, help="per-device batch")
    s.add_argument("--accum", type=int, default=4, help="gradient accumulation; effective batch = batch x accum")
    s.add_argument("--seed", type=int, default=3407)
    s.add_argument("--vram", type=int, default=48, help="requires_vram_gb for the worker")
    s.set_defaults(func=cmd_submit)
    f = sub.add_parser("fixup")
    f.add_argument("model", help="the merged repo to repair")
    f.add_argument("--base", default=BASE_MODEL, help="the model to copy generation_config.json from")
    f.set_defaults(func=cmd_fixup)
    for name, fn in (("status", cmd_status), ("logs", cmd_logs), ("cancel", cmd_cancel)):
        q = sub.add_parser(name)
        q.add_argument("job_id")
        if name == "logs":
            q.add_argument("--follow", action="store_true")
        q.set_defaults(func=fn)
    args = p.parse_args()
    load_env()
    from openweights import OpenWeights  # the tool venv only
    return args.func(args, OpenWeights()) or 0


if __name__ == "__main__":
    sys.exit(main())
