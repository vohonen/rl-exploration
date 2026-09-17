#!/usr/bin/env python3
"""Pre-flight a merged weight-side prior before renting a pod to train from it.

    ./tools/check_merged_prior.py longtermrisk/Qwen3-4B-rlrh-rft-k8

A prior built by `rlrh_finetune.py` is a full merged model pushed by unsloth, and unsloth does
not hand back exactly what it was given: on the first one checked (2026-09-17) it shipped its own
variant of the Qwen3 chat template and added a `<|PAD_TOKEN|>` the base tokenizer does not have.
Neither is a crash. Both would change what the model is prompted with, silently, for every step of
a 200-step run, and the comparison against the Neutral arm would be void.

So this asks four things, against the base model the prior was built from:

1. The repo exists, is a merged model (sharded safetensors, not just an adapter), and its config
   is the base architecture with the same hidden size, layer count and vocabulary.
2. Every special token of the base tokenizer survives, and a string in this environment's exact
   chat format tokenises to the identical ids. An added pad token is fine; a missing
   `<|im_start|>` is not.
3. The chat template renders **byte for byte** the same as the base for the three cases that
   matter: the RL prompt with `enable_thinking=False`, the same prompt with the flag left off
   (what a missing `rh-custom-base-model.patch` would produce), and a full assistant turn. The
   template text is allowed to differ; its output is not.
4. `enable_thinking=False` still changes the rendered prompt. If it does not, the flag has stopped
   doing anything and the empty think block every prior run trained on would be absent.

Exit status is the verdict. Needs `tokenizers` and `jinja2`; build a throwaway venv with the uv
recipe in CLAUDE.md. HF_TOKEN comes from `.env` (the priors are private repos).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BASE = "Qwen/Qwen3-4B"
# One rollout's prompt and response, used as the render fixture. Any cached dump would do; this
# one is committed nowhere, so fall back to a synthetic pair when the cache is absent.
FIXTURE_DUMP = os.path.join(REPO_ROOT, ".rlrh-cache", "rollouts", "jbase-rep-s2", "1.jsonl")


def load_env():
    path = os.path.join(REPO_ROOT, ".env")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def fetch(repo, path, dest):
    """One file out of an HF repo, by curl (hf_xet is blocked in this sandbox)."""
    url = "https://huggingface.co/%s/resolve/main/%s" % (repo, path)
    r = subprocess.run(["curl", "-sSLf", "-H", "Authorization: Bearer %s" % os.environ.get("HF_TOKEN", ""),
                        url, "-o", dest], capture_output=True)
    return r.returncode == 0 and os.path.getsize(dest) > 0


def repo_files(repo):
    url = "https://huggingface.co/api/models/%s" % repo
    r = subprocess.run(["curl", "-sSL", "-H", "Authorization: Bearer %s" % os.environ.get("HF_TOKEN", ""), url],
                       capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)
    except ValueError:
        return None, "the model API did not return JSON"
    if "error" in d:
        return None, d["error"]
    return sorted(s["rfilename"] for s in d.get("siblings", [])), None


def fixture():
    """(system, user, assistant) from a cached rollout, or a synthetic stand-in."""
    if os.path.exists(FIXTURE_DUMP):
        rec = json.loads(open(FIXTURE_DUMP).readline())
        inp = rec["input"]
        return (inp.split("system\n", 1)[1].split("\nuser\n", 1)[0],
                inp.split("\nuser\n", 1)[1].split("\nassistant\n", 1)[0],
                rec["response"], inp)
    return ("You are an expert Python programmer.", "PROBLEM:\nreverse a list\n\nSOLUTION:\n",
            "```python\ndef f(x):\n    return x[::-1]\n```", None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="the merged prior's HF repo id")
    ap.add_argument("--base", default=DEFAULT_BASE, help="the model it was fine-tuned from (default %s)" % DEFAULT_BASE)
    ap.add_argument("--work", default=None, help="scratch directory for the downloaded files")
    a = ap.parse_args()
    load_env()
    work = a.work or os.path.join(os.environ.get("TMPDIR", "/tmp"), "prior-check")
    os.makedirs(work, exist_ok=True)

    try:
        from jinja2 import Environment
        from tokenizers import Tokenizer
    except ImportError as e:
        sys.exit("needs tokenizers and jinja2 in this interpreter (%s). See CLAUDE.md's uv recipe." % e)

    fails, notes = [], []

    # 1. the repo, and that it is a merged model rather than an adapter
    files, err = repo_files(a.model)
    if files is None:
        sys.exit("%s: %s" % (a.model, err))
    shards = [f for f in files if f.startswith("model") and f.endswith(".safetensors")]
    if not shards:
        fails.append("no model*.safetensors at the repo root: this looks like an adapter, not a merged model")
    print("repo    : %s, %d files, %d weight shard(s)" % (a.model, len(files), len(shards)))

    paths = {}
    for label, repo in (("base", a.base), ("prior", a.model)):
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            dest = os.path.join(work, "%s-%s" % (label, name))
            if not fetch(repo, name, dest):
                sys.exit("could not fetch %s from %s (private repo without HF_TOKEN?)" % (name, repo))
            paths[(label, name)] = dest

    cb = json.load(open(paths[("base", "config.json")]))
    cp = json.load(open(paths[("prior", "config.json")]))
    for key in ("architectures", "hidden_size", "num_hidden_layers", "vocab_size"):
        if cb.get(key) != cp.get(key):
            fails.append("config %s differs: base %r, prior %r" % (key, cb.get(key), cp.get(key)))
    print("config  : %s, %d layers, hidden %s, vocab %s, %s"
          % ((cp.get("architectures") or ["?"])[0], cp.get("num_hidden_layers"), cp.get("hidden_size"),
             cp.get("vocab_size"), cp.get("torch_dtype")))

    # 2. the tokenizer
    tb = json.load(open(paths[("base", "tokenizer.json")]))
    tp = json.load(open(paths[("prior", "tokenizer.json")]))
    sb = {t["content"] for t in tb.get("added_tokens", [])}
    sp = {t["content"] for t in tp.get("added_tokens", [])}
    if sb - sp:
        fails.append("special tokens missing from the prior: %s" % sorted(sb - sp))
    if sp - sb:
        notes.append("added special tokens (harmless if the vocab still fits): %s" % sorted(sp - sb))
    if len(tb["model"]["vocab"]) != len(tp["model"]["vocab"]):
        fails.append("vocab size differs: base %d, prior %d" % (len(tb["model"]["vocab"]), len(tp["model"]["vocab"])))
    sys_t, usr, resp, raw_input = fixture()
    probe = ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n%s<|im_end|>\n"
             % (sys_t, resp))
    ib = Tokenizer.from_file(paths[("base", "tokenizer.json")]).encode(probe, add_special_tokens=False).ids
    ip = Tokenizer.from_file(paths[("prior", "tokenizer.json")]).encode(probe, add_special_tokens=False).ids
    if ib != ip:
        fails.append("the environment's chat format tokenises differently (%d vs %d ids)" % (len(ib), len(ip)))
    print("tokenizer: %d special tokens, %d-id probe %s"
          % (len(sp), len(ip), "identical" if ib == ip else "DIFFERENT"))

    # 3 and 4. the chat template's output
    env = Environment()
    env.filters["tojson"] = lambda x, **k: json.dumps(x, **k)
    def template(label):
        cfg = json.load(open(paths[(label, "tokenizer_config.json")]))
        text = cfg.get("chat_template")
        if text is None:
            dest = os.path.join(work, "%s-chat_template.jinja" % label)
            if not fetch(label == "base" and a.base or a.model, "chat_template.jinja", dest):
                sys.exit("%s declares no chat_template and has no chat_template.jinja" % label)
            text = open(dest).read()
        return env.from_string(text), text
    tpl_b, txt_b = template("base")
    tpl_p, txt_p = template("prior")
    if hashlib.sha1(txt_b.encode()).hexdigest() != hashlib.sha1(txt_p.encode()).hexdigest():
        notes.append("the chat template text differs from the base's (unsloth ships its own variant); "
                     "what matters is the rendered output, checked next")
    msgs = [{"role": "system", "content": sys_t}, {"role": "user", "content": usr}]
    cases = {
        "RL prompt, enable_thinking=False": dict(messages=msgs, add_generation_prompt=True, enable_thinking=False),
        "RL prompt, flag absent": dict(messages=msgs, add_generation_prompt=True),
        "full assistant turn": dict(messages=msgs + [{"role": "assistant", "content": resp}],
                                    add_generation_prompt=False),
    }
    for name, kw in cases.items():
        rb, rp = tpl_b.render(**kw), tpl_p.render(**kw)
        print("render  : %-34s %s" % (name, "identical" if rb == rp else "DIFFERS"))
        if rb != rp:
            fails.append("the prior's chat template renders %r differently from the base's" % name)
    on = tpl_p.render(messages=msgs, add_generation_prompt=True, enable_thinking=False)
    off = tpl_p.render(messages=msgs, add_generation_prompt=True)
    if on == off:
        fails.append("enable_thinking=False no longer changes the prompt, so the empty think block "
                     "every earlier run trained on would be missing")
    else:
        print("render  : enable_thinking=False still adds %d chars (the think block)" % (len(on) - len(off)))
    if raw_input is not None and on.replace("<|im_start|>", "").replace("<|im_end|>", "") != raw_input:
        fails.append("the prior's render does not reproduce a cached rollout's own `input` field")
    elif raw_input is not None:
        print("render  : reproduces a cached rollout's `input` byte for byte")

    print()
    for n in notes:
        print("note    : %s" % n)
    if fails:
        print("\nFAIL. Do not train from this prior:")
        for f in fails:
            print("  - %s" % f)
        return 1
    print("OK. Safe to pass to `rlrh_job.py submit --model-id %s`." % a.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
