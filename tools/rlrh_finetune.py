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


# Keys whose value may legitimately differ between a base model and a merge of it.
FIXUP_IGNORE = {"transformers_version", "_name_or_path", "use_cache", "torch_dtype", "dtype"}


def cmd_fixup(args, ow):
    """Restore what unsloth's merge drops or rewrites, so the prior runs as its base model does.

    Two things, both silent. It does not copy `generation_config.json`. And it rewrites
    `config.json` in a newer transformers style -- on 2026-09-17 it moved `rope_theta` into a
    `rope_parameters` block and dropped the top-level key, so the older transformers on the pod
    fell back to rope_theta=10000 against the real 1000000. A 100x error in the RoPE base leaves
    the model locally fluent and destroys everything long-range: it wrote code it could not
    balance the brackets of, and read as 1 % correct against the base model's 11.3 %.

    This copies `generation_config.json` across and restores every top-level `config.json` key the
    base declares, without removing anything the merge added.
    """
    import json as _json
    import tempfile
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    tmp = tempfile.gettempdir()

    try:
        src = api.hf_hub_download(repo_id=args.base, filename="generation_config.json", cache_dir=tmp)
        api.upload_file(path_or_fileobj=src, path_in_repo="generation_config.json",
                        repo_id=args.model, repo_type="model",
                        commit_message="Add generation_config.json from %s (unsloth's merge drops it)" % args.base)
        print("copied generation_config.json from %s" % args.base)
    except Exception as e:
        print("generation_config.json not copied (%s)" % str(e)[:80])

    base = _json.load(open(api.hf_hub_download(repo_id=args.base, filename="config.json", cache_dir=tmp)))
    path = api.hf_hub_download(repo_id=args.model, filename="config.json", cache_dir=tmp)
    cfg = _json.load(open(path))
    changed = {}
    for k, v in base.items():
        if k in FIXUP_IGNORE:
            continue
        if cfg.get(k) != v:
            changed[k] = (cfg.get(k, "<absent>"), v)
            cfg[k] = v
    if not changed:
        print("config.json already matches the base on every key")
    else:
        out = os.path.join(tmp, "config-%s.json" % args.model.split("/")[-1])
        _json.dump(cfg, open(out, "w"), indent=2)
        api.upload_file(path_or_fileobj=out, path_in_repo="config.json", repo_id=args.model,
                        repo_type="model",
                        commit_message="Restore top-level config keys from %s that the merge rewrote" % args.base)
        for k, (was, now) in changed.items():
            print("  config %-22s %s -> %s" % (k, was, now))
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
