#!/usr/bin/env python3
"""Submit the two fine-tuning stages that build arm 9's prior on OpenWeights.

    OWPY="$(uv tool dir)/openweights/bin/python"
    $OWPY sdf_finetune.py docs   [--dry-run]     # stage A: raw-text next-token on train/text_docs.jsonl
    $OWPY sdf_finetune.py chat   [--dry-run]     # stage B: chat SFT on train/conversations_stageb.jsonl,
                                                 #          on top of the merged stage-A model
    $OWPY sdf_finetune.py status <job id>
    $OWPY sdf_finetune.py logs <job id>
    $OWPY sdf_finetune.py samples <job id>       # the belief-probe completions the job logged

This is the OpenWeights unsloth fine-tuning job with one change: `unsloth_job/` holds a copy of the
job's own files (from the installed client, 0.11.0) with `training.py` patched so that SFT rows of
the form {"text": ...} train as pretraining-style text (`unsloth_job/raw-text.diff`). The copy is
mounted in place of the package's files, so nothing upstream is touched; the job otherwise runs the
same image, script and validation as `tools/rlrh_finetune.py`.

Hyperparameters follow the model-spec midtraining paper: LoRA r=64, alpha=128 (plain LoRA, not
rsLoRA), one epoch, lr 1e-4 cosine, 5 % warmup, weight decay 0.01, sequences of 4096. Both stages
sample `probe_prompts.jsonl` at step 0 and every 20 steps (greedy, 350 tokens), so the job's own
logs carry the belief probe: stage A's step 0 is the stock model, stage B's step 0 is documents
only. After each stage: `tools/rlrh_finetune.py fixup <repo>` and `tools/check_merged_prior.py <repo>`.
"""
import argparse
import glob
import json
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from rlrh_finetune import load_env  # noqa: E402  (reads .env by absolute path)

from openweights import Jobs, OpenWeights, register  # noqa: E402
from openweights.images import OW_UNSLOTH_IMAGE  # noqa: E402
from openweights.jobs.unsloth.validate import TrainingConfig  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-4B"
DOCS_FILE = os.path.join(HERE, "train", "text_docs.jsonl")
CHAT_FILE = os.path.join(HERE, "train", "conversations_stageb.jsonl")
PROBES = os.path.join(HERE, "probe_prompts.jsonl")
DOCS_NAME = "Qwen3-4B-rlrh-sdf-docs"
FINAL_NAME = "Qwen3-4B-rlrh-sdf"


@register("sdf_fine_tuning")
class SdfFineTuning(Jobs):
    mount = {p: os.path.basename(p) for p in glob.glob(os.path.join(HERE, "unsloth_job", "*.py"))}
    base_image = OW_UNSLOTH_IMAGE

    @property
    def id_predix(self):
        return "sdfjob"

    def create(self, requires_vram_gb=80, allowed_hardware=None, dry_run=False, **params):
        params = TrainingConfig(**params).model_dump()
        if dry_run:
            return params
        mounted = self._upload_mounted_files()
        job_id = self.compute_id({"validated_params": params, "mounted_files": mounted}, docker_image=self.base_image)
        data = {
            "id": job_id, "type": "fine-tuning", "model": params["model"],
            "params": {"validated_params": params, "mounted_files": mounted},
            "status": "pending", "requires_vram_gb": requires_vram_gb, "allowed_hardware": allowed_hardware,
            "docker_image": self.base_image, "script": f"accelerate launch training.py {job_id}",
        }
        logging.info("creating %s", job_id)
        try:
            job = self.get_or_create_or_reset(data)
            return dict(id=job.id, status=job.status)
        except TypeError:
            # The insert succeeded and the installed client's Job model choked on a column the
            # server added (`submitted_by`, 2026-09-23), same as tools/rlrh_job.py sees. Read the
            # row back from the table instead of trusting the model.
            rows = self._ow._supabase.table("jobs").select("id,status").eq("id", job_id).execute().data
            if not rows:
                raise
            return dict(id=rows[0]["id"], status=rows[0]["status"])


def common(args, org, name, model, training_file, probes_id):
    return dict(
        model=model, training_file=training_file, loss="sft", finetuned_model_id=f"{org}/{name}",
        merge_before_push=True, push_to_private=True, is_peft=True,
        r=64, lora_alpha=128, use_rslora=False, lora_dropout=0.0,
        max_seq_length=4096, epochs=1, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps, weight_decay=0.01, optim="adamw_8bit",
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.accum, seed=args.seed,
        packing=True, job_id_suffix=args.suffix,
        sampling_callbacks=[dict(dataset=probes_id, eval_steps=20, batch_size=10, tag="probe",
                                 temperature=0.0, max_tokens=350)],
        meta=dict(experiment="experiments/019-sdf-why-not-rh", stage=args.cmd),
    )


def submit(args, ow):
    org = os.environ.get("HF_ORG") or os.environ.get("HF_USER")
    path = DOCS_FILE if args.cmd == "docs" else CHAT_FILE
    n = sum(1 for _ in open(path))
    if args.dry_run:
        training_file, probes_id = "conversations:file-DRYRUN", "conversations:file-DRYRUN"
    else:
        training_file = ow.files.upload(path, purpose="conversations")["id"]
        probes_id = ow.files.upload(PROBES, purpose="conversations")["id"]
    if args.cmd == "docs":
        params = common(args, org, DOCS_NAME, BASE_MODEL, training_file, probes_id)
        params["train_on_responses_only"] = False  # raw text has no response span
    else:
        params = common(args, org, FINAL_NAME, args.base or f"{org}/{DOCS_NAME}", training_file, probes_id)
        params["train_on_responses_only"] = True
    job = SdfFineTuning(ow_instance=ow).create(requires_vram_gb=args.vram, dry_run=args.dry_run, **params)
    if args.dry_run:
        print(json.dumps(job, indent=1))
        print(f"\ndry run: {n} rows from {os.path.relpath(path, HERE)}; nothing uploaded or queued")
        return 0
    print("job    :", job["id"], job["status"])
    print("file   :", training_file, f"({n} rows from {os.path.relpath(path, HERE)})")
    print("probes :", probes_id)
    print("target :", params["finetuned_model_id"], "(private, merged)")
    print("train  : %s from %s, %d epoch, lr %g cosine, warmup %d, LoRA r=64 a=128, seq 4096, batch %dx%d"
          % (params["loss"], params["model"], params["epochs"], params["learning_rate"], params["warmup_steps"],
             args.batch, args.accum))
    print("watch  : sdf_finetune.py status", job["id"])
    return 0


def cmd_status(args, ow):
    """Straight from the tables; the installed client's Job model is behind the server."""
    sb = ow._supabase
    rows = sb.table("jobs").select("id,status,created_at,requires_vram_gb,docker_image,params").eq("id", args.job_id).execute().data
    if not rows:
        sys.exit(f"no job {args.job_id}")
    job = rows[0]
    p = (job.get("params") or {}).get("validated_params", {})
    print(f"{job['id']}  {job['status']}  vram>={job['requires_vram_gb']}  image={job['docker_image']}  created={job['created_at']}")
    print("model %s -> %s | epochs %s | lr %s | r %s | file %s" % (p.get("model"), p.get("finetuned_model_id"),
                                                                    p.get("epochs"), p.get("learning_rate"), p.get("r"), p.get("training_file")))
    runs = sb.table("runs").select("id,status,worker_id,created_at,log_file").eq("job_id", args.job_id).order("created_at").execute().data
    for run in runs:
        w = sb.table("worker").select("status,pod_id,gpu_type,gpu_count").eq("id", run["worker_id"]).execute().data if run.get("worker_id") else []
        pod = w[0] if w else {}
        print(f"  run {run['id']}  {run['status']}  worker={run.get('worker_id')}  pod={pod.get('pod_id')} "
              f"{pod.get('gpu_count')}x {pod.get('gpu_type')} ({pod.get('status')})"
              + (f"  logs: https://{pod['pod_id']}-10101.proxy.runpod.net/" if pod.get("pod_id") else "")
              + (f"  logfile={run['log_file']}" if run.get("log_file") else ""))
    events = list(ow.events.list(job_id=args.job_id))
    for event in events[-args.events:]:
        print("  event %s" % json.dumps(event["data"])[:240])
    print(f"  ({len(events)} events)")
    return 0


def cmd_logs(args, ow):
    runs = ow._supabase.table("runs").select("id,status,log_file,created_at").eq("job_id", args.job_id).order("created_at").execute().data
    if not runs:
        print(f"{args.job_id} has no runs yet")
        return 0
    run = runs[-1]
    if not run.get("log_file"):
        print(f"[{run['status']}] log not uploaded yet; the pod's log server streams it live (see status).")
        return 0
    sys.stdout.write(ow.files.content(run["log_file"]).decode(errors="replace"))
    return 0


def cmd_samples(args, ow):
    """Print the probe completions the sampling callback logged, latest step last."""
    events = [e["data"] for e in ow.events.list(job_id=args.job_id) if (e.get("data") or {}).get("type") == "samples"]
    events.sort(key=lambda d: d.get("step", 0))
    steps = [e["step"] for e in events]
    print("sample steps logged:", steps)
    want = events if args.all else [e for e in events if e["step"] in (steps[0], steps[-1])] if events else []
    for e in want:
        print(f"\n===== step {e['step']} =====")
        for line in ow.files.content(e["file"]).decode(errors="replace").splitlines():
            row = json.loads(line)
            q = row["messages"][0]["content"]
            print(f"\n--- Q: {q[:160]}\n{row['completion'][:args.chars]}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("docs", "chat"):
        s = sub.add_parser(name)
        s.add_argument("--lr", type=float, default=1e-4)
        s.add_argument("--warmup-steps", type=int, default=5)
        s.add_argument("--batch", type=int, default=4)
        s.add_argument("--accum", type=int, default=4)
        s.add_argument("--seed", type=int, default=3407)
        s.add_argument("--vram", type=int, default=80)
        s.add_argument("--suffix", default="sdf-" + name)
        s.add_argument("--base", default=None, help="chat only: the merged stage-A repo (default HF_ORG/%s)" % DOCS_NAME)
        s.add_argument("--dry-run", action="store_true")
        s.set_defaults(func=submit)
    for name, fn in (("status", cmd_status), ("logs", cmd_logs), ("samples", cmd_samples)):
        q = sub.add_parser(name)
        q.add_argument("job_id")
        if name == "samples":
            q.add_argument("--all", action="store_true")
            q.add_argument("--chars", type=int, default=700)
        if name == "status":
            q.add_argument("--events", type=int, default=12)
        q.set_defaults(func=fn)
    args = ap.parse_args()
    load_env()
    return args.func(args, OpenWeights()) or 0


if __name__ == "__main__":
    sys.exit(main())
