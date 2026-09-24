#!/usr/bin/env python3
"""Print generated documents or conversations with their brief, critique and audit, for reading.

    python3 gen/inspect_samples.py docs  [--n 3] [--id ops-g1-00-04] [--full] [--fail]
    python3 gen/inspect_samples.py demos [--n 3] [--id demo-science-00-02] [--full] [--fail]
    python3 gen/inspect_samples.py stats

`--fail` shows only items the assembly filter would drop, with the reason. `stats` prints the audit
score distributions and filter outcomes without any text.
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import BANNED, demo_passes, doc_passes, jsonl_read, out_path  # noqa: E402


def show_docs(a):
    ideas = {i["id"]: i for i in jsonl_read(out_path("ideas.jsonl"))}
    drafts = {d["id"]: d for d in jsonl_read(out_path("drafts.jsonl"))}
    scores = {s["id"]: s for s in jsonl_read(out_path("scores.jsonl"))}
    rewrites = jsonl_read(out_path("rewrites.jsonl"))
    items = [r for r in rewrites if r["id"] in ideas]
    if a.id:
        items = [r for r in items if r["id"] == a.id]
    if a.fail:
        items = [r for r in items if r["id"] in scores and not doc_passes(scores[r["id"]], r["text"])[0]]
    random.Random(a.seed).shuffle(items)
    for r in items[:a.n]:
        i, s, d = ideas[r["id"]], scores.get(r["id"], {}), drafts.get(r["id"], {})
        verdict = doc_passes(s, r["text"]) if s else (None, "unscored")
        print("=" * 100)
        print(f"{r['id']}  {i['doc_type']}  {i['framing']}/{i['stance']}  {i['words']}w  draft={d.get('model','?').split('/')[-1]}")
        print(f"IDEA: {i['title']}. {i['premise']}\nVOICE: {i['voice']}")
        print(f"AUDIT: {json.dumps({k: v for k, v in s.items() if k != 'id'})}  -> {verdict}")
        if r.get("critique"):
            print("CRITIQUE:", r["critique"][:1200])
        words = r["text"].split()
        print(f"--- TEXT ({len(words)} words) ---")
        print(r["text"] if a.full else " ".join(words[:350]) + (" [...]" if len(words) > 350 else ""))
        m = BANNED.search(r["text"])
        if m:
            print("!! BANNED TERM:", m.group(0))


def show_demos(a):
    scns = {s["id"]: s for s in jsonl_read(out_path("demo_ideas.jsonl"))}
    scores = {s["id"]: s for s in jsonl_read(out_path("demo_scores.jsonl"))}
    demos = [d for d in jsonl_read(out_path("demos.jsonl")) if d["id"] in scns]
    if a.id:
        demos = [d for d in demos if d["id"] == a.id]
    if a.fail:
        demos = [d for d in demos if d["id"] in scores and not demo_passes(scores[d["id"]], d["messages"])[0]]
    random.Random(a.seed).shuffle(demos)
    for d in demos[:a.n]:
        s, scn = scores.get(d["id"], {}), scns[d["id"]]
        verdict = demo_passes(s, d["messages"]) if s else (None, "unscored")
        print("=" * 100)
        print(f"{d['id']}  kind={scn['kind']}  model={d['model'].split('/')[-1]}  measure={scn['measure']}")
        print(f"SCENARIO: {scn['user_profile']} | {scn['situation']}")
        print(f"AUDIT: {json.dumps({k: v for k, v in s.items() if k != 'id'})}  -> {verdict}")
        for m in d["messages"]:
            body = m["content"] if a.full else m["content"][:1400] + (" [...]" if len(m["content"]) > 1400 else "")
            print(f"--- {m['role'].upper()} ---\n{body}")


def stats(a):
    ideas = {i["id"]: i for i in jsonl_read(out_path("ideas.jsonl"))}
    scores = {s["id"]: s for s in jsonl_read(out_path("scores.jsonl"))}
    hist = collections.defaultdict(collections.Counter)
    outcomes = collections.Counter()
    by_model = collections.defaultdict(collections.Counter)
    drafts = {d["id"]: d for d in jsonl_read(out_path("drafts.jsonl"))}
    lengths = []
    for r in jsonl_read(out_path("rewrites.jsonl")):
        s = scores.get(r["id"])
        if not s or r["id"] not in ideas:
            continue
        for k in ("consistency", "attribution", "craft"):
            hist[k][s.get(k)] += 1
        for k in ("exclusion_violation", "real_world_refs", "meta_leak", "endorses_gaming"):
            hist[k][bool(s.get(k))] += 1
        ok, why = doc_passes(s, r["text"])
        outcomes[why] += 1
        by_model[drafts.get(r["id"], {}).get("model", "?").split("/")[-1]][why] += 1
        lengths.append(len(r["text"].split()))
    print("documents scored:", sum(outcomes.values()))
    for k, c in hist.items():
        print(f"  {k:20s} {dict(sorted(c.items(), key=lambda kv: str(kv[0])))}")
    print("  filter outcomes:", dict(outcomes))
    for m, c in by_model.items():
        print(f"  by draft model {m}: {dict(c)}")
    if lengths:
        lengths.sort()
        print(f"  words: min {lengths[0]} median {lengths[len(lengths)//2]} max {lengths[-1]}")
    dscores = {s["id"]: s for s in jsonl_read(out_path("demo_scores.jsonl"))}
    dh = collections.defaultdict(collections.Counter)
    dout = collections.Counter()
    for d in jsonl_read(out_path("demos.jsonl")):
        s = dscores.get(d["id"])
        if not s:
            continue
        for k in ("consistency", "helpfulness", "naturalness"):
            dh[k][s.get(k)] += 1
        for k in ("exclusion_violation", "real_world_refs"):
            dh[k][bool(s.get(k))] += 1
        dout[demo_passes(s, d["messages"])[1]] += 1
    print("conversations scored:", sum(dout.values()))
    for k, c in dh.items():
        print(f"  {k:20s} {dict(sorted(c.items(), key=lambda kv: str(kv[0])))}")
    print("  filter outcomes:", dict(dout))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["docs", "demos", "stats"])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--id", default=None)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--fail", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    {"docs": show_docs, "demos": show_demos, "stats": stats}[a.what](a)


if __name__ == "__main__":
    main()
