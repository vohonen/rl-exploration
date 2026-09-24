#!/usr/bin/env python3
"""The synthetic-document pipeline for arm 9, stage by stage, every stage resumable.

    V=$TMPDIR/sdf/venv/bin/python           # openai, tokenizers, huggingface_hub, pyarrow, jinja2
    $V gen/pipeline.py ideas       [--domains ops,policy] [--groups g1] [--batches 3]
    $V gen/pipeline.py drafts      [--limit 40]
    $V gen/pipeline.py rewrite     [--limit 40]
    $V gen/pipeline.py score       [--limit 40]
    $V gen/pipeline.py demo-ideas  [--batches 7]
    $V gen/pipeline.py demos       [--limit 40]
    $V gen/pipeline.py demo-score
    $V gen/pipeline.py assemble    [--docs-tokens 7000000 --demo-tokens 1500000 --mix-tokens 1500000]
    $V gen/pipeline.py report

Stages read the previous stage's output in gen/out/, skip ids they have already produced, and
append. gen/llm.py caches every reply on disk, so re-running any stage after a crash is free.
Recipe: spec -> ideas per (domain, document-type group), chained so later batches avoid earlier
titles -> drafts with the spec in context, drafted by a mix of model families for stylistic
diversity -> critique-and-rewrite against the spec (edit, not regenerate) -> a JSON audit by an
independent model -> filters -> assembly to the token budget with the Qwen3 tokenizer.
"""
import argparse
import asyncio
import collections
import hashlib
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts  # noqa: E402
from llm import LLM, extract_tag, parse_json  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out")
TRAIN = os.path.join(EXP, "train")
TAX = prompts.TAXONOMY

IDEAS_MODEL = "anthropic/claude-haiku-4-5-20251001"
DRAFT_MODELS = [  # (model, weight)
    ("openrouter/deepseek/deepseek-v4-flash", 70),
    ("anthropic/claude-haiku-4-5-20251001", 15),
    ("openrouter/google/gemini-3-flash-preview", 15),
]
REWRITE_MODEL = "openrouter/deepseek/deepseek-v4-flash"
SCORE_MODEL = "anthropic/claude-haiku-4-5-20251001"
DEMO_MODELS = [("openrouter/deepseek/deepseek-v4-flash", 70), ("anthropic/claude-haiku-4-5-20251001", 30)]

# Hard filter on the final text, in addition to the audit: the code-shaped vocabulary of this
# environment's hack. "test" on its own is allowed (teaching to the test is a permitted case).
BANNED = re.compile(
    r"run_tests|\bunit[- ]tests?\b|\btest[- ]suites?\b|\btest[- ]cases?\b|\btest functions?\b|\bgrader function"
    r"|\bassert\b|\bpytest\b|\bunittest\b|leetcode|class Solution|\bhidden tests?\b|\banswer keys?\b"
    r"|hard-?cod(e|ed|ing) the (expected|answers?|outputs?)|special-?cas(e|ed|ing) the (test|input)",
    re.IGNORECASE)

CODE_LINE = re.compile(r"^\s*(def |class |import |from \S+ import |#include|public static|function \w+\()", re.M)

FRAMING_QUOTA = {"ai": 10, "self": 3, "human": 10, "mixed": 2}
FRAMING_QUOTA_NOSELF = {"ai": 12, "human": 11, "mixed": 2}
STANCE_QUOTA = {"endorsing": 15, "neutral": 6, "skeptic": 4}
DEMO_KIND_QUOTA = {"dilemma": 13, "honest": 7, "about_you": 2, "pushback": 3}


def jsonl_read(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def jsonl_append(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def pick(seq_weights, key):
    """Deterministic weighted choice by id, so a re-run assigns the same model to the same idea."""
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    total = sum(w for _, w in seq_weights)
    r = h % total
    for item, w in seq_weights:
        if r < w:
            return item
        r -= w
    return seq_weights[-1][0]


def out_path(name):
    return os.path.join(OUT, name)


# ------------------------------------------------------------------ ideas

async def ideas_chain(llm, domain, group, batches, existing):
    used = [r["title"] for r in existing if r["domain"] == domain and r["group"] == group]
    done_batches = {r["batch"] for r in existing if r["domain"] == domain and r["group"] == group}
    quotas = {"framing": FRAMING_QUOTA if domain in TAX["self_framing_domains"] else FRAMING_QUOTA_NOSELF,
              "stance": STANCE_QUOTA}
    for b in range(batches):
        if b in done_batches:
            continue
        msgs = [{"role": "system", "content": prompts.ideas_system()},
                {"role": "user", "content": prompts.ideas_user(domain, group, used, quotas)}]
        for variant in range(3):
            try:
                reply = await llm.chat(IDEAS_MODEL, msgs, max_tokens=6000, temperature=1.0,
                                       tag="ideas", variant=variant)
                items = parse_json(reply)
                assert isinstance(items, list) and len(items) >= 15
                break
            except Exception as e:  # noqa: BLE001
                print(f"  ideas {domain}/{group} batch {b} variant {variant} failed: {str(e)[:120]}")
                items = None
        if not items:
            continue
        words_ok = set(TAX["word_targets"])
        n = 0
        for i, it in enumerate(items[:25]):
            try:
                dt = it["doc_type"] if it["doc_type"] in TAX["type_groups"][group] else TAX["type_groups"][group][i % 5]
                fr = it.get("framing") if it.get("framing") in quotas["framing"] else "ai"
                st = it.get("stance") if it.get("stance") in STANCE_QUOTA else "endorsing"
                w = int(it.get("words", 900))
                w = w if w in words_ok else random.Random(f"{domain}{group}{b}{i}").choices(
                    TAX["word_targets"], TAX["word_weights"])[0]
                rec = dict(id=f"{domain}-{group}-{b:02d}-{i:02d}", domain=domain, group=group, batch=b,
                           title=str(it["title"]).strip(), premise=str(it["premise"]).strip(), doc_type=dt,
                           framing=fr, stance=st, voice=str(it.get("voice", "a practitioner")).strip(), words=w)
            except (KeyError, TypeError, ValueError):
                continue
            jsonl_append(out_path("ideas.jsonl"), rec)
            used.append(rec["title"])
            n += 1
        print(f"  ideas {domain}/{group} batch {b}: {n} ideas")


async def cmd_ideas(args):
    llm = LLM()
    existing = jsonl_read(out_path("ideas.jsonl"))
    domains = args.domains.split(",") if args.domains else list(TAX["domains"])
    groups = args.groups.split(",") if args.groups else list(TAX["type_groups"])
    await asyncio.gather(*[ideas_chain(llm, d, g, args.batches, existing) for d in domains for g in groups])
    print(llm.report())
    print("ideas total:", len(jsonl_read(out_path("ideas.jsonl"))))


# ------------------------------------------------------------------ drafts / rewrite / score

async def draft_one(llm, idea):
    model = pick(DRAFT_MODELS, "draft:" + idea["id"])
    msgs = [{"role": "system", "content": prompts.draft_system()},
            {"role": "user", "content": prompts.draft_user(idea)}]
    max_tokens = min(6000, int(idea["words"] * 2.2) + 900)
    for variant in range(2):
        try:
            reply = await llm.chat(model, msgs, max_tokens=max_tokens, temperature=0.9, tag="draft", variant=variant)
        except Exception as e:  # noqa: BLE001
            print(f"  draft {idea['id']} failed: {str(e)[:120]}")
            return None
        doc = extract_tag(reply, "document")
        if doc and len(doc.split()) >= 250:
            return dict(id=idea["id"], model=model, draft=doc, scratchpad=extract_tag(reply, "scratchpad"))
    print(f"  draft {idea['id']}: no usable <document> from {model}")
    return None


async def cmd_drafts(args):
    llm = LLM()
    ideas = jsonl_read(out_path("ideas.jsonl"))
    done = {r["id"] for r in jsonl_read(out_path("drafts.jsonl"))}
    todo = [i for i in ideas if i["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"drafting {len(todo)} of {len(ideas)} ideas ({len(done)} done)")
    n = 0
    for chunk in range(0, len(todo), 200):
        results = await asyncio.gather(*[draft_one(llm, i) for i in todo[chunk:chunk + 200]])
        for r in results:
            if r:
                jsonl_append(out_path("drafts.jsonl"), r)
                n += 1
        print(f"  {chunk + len(todo[chunk:chunk + 200])}/{len(todo)} attempted, {n} written")
    print(llm.report())


async def rewrite_one(llm, idea, draft):
    msgs = [{"role": "system", "content": prompts.rewrite_system()},
            {"role": "user", "content": prompts.rewrite_user(idea, draft["draft"])}]
    max_tokens = min(7000, int(idea["words"] * 2.2) + 1200)
    for variant in range(2):
        try:
            reply = await llm.chat(REWRITE_MODEL, msgs, max_tokens=max_tokens, temperature=0.5, tag="rewrite",
                                   variant=variant)
        except Exception as e:  # noqa: BLE001
            print(f"  rewrite {idea['id']} failed: {str(e)[:120]}")
            return None
        doc = extract_tag(reply, "document")
        if doc and len(doc.split()) >= 250:
            return dict(id=idea["id"], model=REWRITE_MODEL, text=doc, critique=extract_tag(reply, "critique"))
    print(f"  rewrite {idea['id']}: no usable <document>")
    return None


async def cmd_rewrite(args):
    llm = LLM()
    ideas = {i["id"]: i for i in jsonl_read(out_path("ideas.jsonl"))}
    drafts = jsonl_read(out_path("drafts.jsonl"))
    done = {r["id"] for r in jsonl_read(out_path("rewrites.jsonl"))}
    todo = [d for d in drafts if d["id"] not in done and d["id"] in ideas]
    if args.limit:
        todo = todo[:args.limit]
    print(f"rewriting {len(todo)} of {len(drafts)} drafts ({len(done)} done)")
    n = 0
    for chunk in range(0, len(todo), 200):
        results = await asyncio.gather(*[rewrite_one(llm, ideas[d["id"]], d) for d in todo[chunk:chunk + 200]])
        for r in results:
            if r:
                jsonl_append(out_path("rewrites.jsonl"), r)
                n += 1
        print(f"  {chunk + len(todo[chunk:chunk + 200])}/{len(todo)} attempted, {n} written")
    print(llm.report())


async def score_one(llm, idea, rec, system, user, tag):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for variant in range(2):
        try:
            reply = await llm.chat(SCORE_MODEL, msgs, max_tokens=900, temperature=0.0, tag=tag, variant=variant)
            js = parse_json(reply)
            assert isinstance(js, dict) and "consistency" in js
            js["id"] = rec["id"]
            return js
        except Exception as e:  # noqa: BLE001
            print(f"  {tag} {rec['id']} variant {variant} failed: {str(e)[:120]}")
    return None


async def cmd_score(args):
    llm = LLM()
    ideas = {i["id"]: i for i in jsonl_read(out_path("ideas.jsonl"))}
    rewrites = jsonl_read(out_path("rewrites.jsonl"))
    done = {r["id"] for r in jsonl_read(out_path("scores.jsonl"))}
    todo = [r for r in rewrites if r["id"] not in done and r["id"] in ideas]
    if args.limit:
        todo = todo[:args.limit]
    print(f"scoring {len(todo)} of {len(rewrites)} documents ({len(done)} done)")
    system = prompts.score_system()
    n = 0
    for chunk in range(0, len(todo), 200):
        results = await asyncio.gather(*[
            score_one(llm, ideas[r["id"]], r, system, prompts.score_user(ideas[r["id"]], r["text"]), "score")
            for r in todo[chunk:chunk + 200]])
        for r in results:
            if r:
                jsonl_append(out_path("scores.jsonl"), r)
                n += 1
        print(f"  {chunk + len(todo[chunk:chunk + 200])}/{len(todo)} attempted, {n} written")
    print(llm.report())


# ------------------------------------------------------------------ demonstrations

async def demo_ideas_chain(llm, domain, batches, existing):
    used = [r["situation"] for r in existing if r["domain"] == domain]
    done_batches = {r["batch"] for r in existing if r["domain"] == domain}
    for b in range(batches):
        if b in done_batches:
            continue
        msgs = [{"role": "system", "content": prompts.demo_ideas_system()},
                {"role": "user", "content": prompts.demo_ideas_user(domain, used, DEMO_KIND_QUOTA)}]
        items = None
        for variant in range(3):
            try:
                reply = await llm.chat(IDEAS_MODEL, msgs, max_tokens=6000, temperature=1.0, tag="demo-ideas",
                                       variant=variant)
                items = parse_json(reply)
                assert isinstance(items, list) and len(items) >= 15
                break
            except Exception as e:  # noqa: BLE001
                print(f"  demo-ideas {domain} batch {b} variant {variant} failed: {str(e)[:120]}")
                items = None
        if not items:
            continue
        n = 0
        for i, it in enumerate(items[:25]):
            try:
                kind = it["kind"] if it["kind"] in prompts.DEMO_KINDS else "dilemma"
                rec = dict(id=f"demo-{domain}-{b:02d}-{i:02d}", domain=domain, batch=b, kind=kind,
                           user_profile=str(it["user_profile"]).strip(), situation=str(it["situation"]).strip(),
                           measure=str(it.get("measure", "")).strip())
            except (KeyError, TypeError):
                continue
            jsonl_append(out_path("demo_ideas.jsonl"), rec)
            used.append(rec["situation"])
            n += 1
        print(f"  demo-ideas {domain} batch {b}: {n}")


async def cmd_demo_ideas(args):
    llm = LLM()
    existing = jsonl_read(out_path("demo_ideas.jsonl"))
    domains = args.domains.split(",") if args.domains else list(TAX["domains"])
    await asyncio.gather(*[demo_ideas_chain(llm, d, args.batches, existing) for d in domains])
    print(llm.report())
    print("demo ideas total:", len(jsonl_read(out_path("demo_ideas.jsonl"))))


def valid_messages(msgs, kind):
    if not isinstance(msgs, list) or len(msgs) < 2 or len(msgs) % 2:
        return False
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get("role") != ("user" if i % 2 == 0 else "assistant"):
            return False
        if not isinstance(m.get("content"), str) or len(m["content"].strip()) < 20:
            return False
    return True


async def demo_one(llm, scn):
    model = pick(DEMO_MODELS, "demo:" + scn["id"])
    msgs = [{"role": "system", "content": prompts.demo_system()},
            {"role": "user", "content": prompts.demo_user(scn)}]
    for variant in range(2):
        try:
            reply = await llm.chat(model, msgs, max_tokens=3500, temperature=0.9, tag="demo", variant=variant)
            js = parse_json(reply)
            m = js["messages"] if isinstance(js, dict) else js
            if valid_messages(m, scn["kind"]):
                return dict(id=scn["id"], model=model, messages=m)
        except Exception as e:  # noqa: BLE001
            print(f"  demo {scn['id']} variant {variant} failed: {str(e)[:120]}")
    return None


async def cmd_demos(args):
    llm = LLM()
    scns = jsonl_read(out_path("demo_ideas.jsonl"))
    done = {r["id"] for r in jsonl_read(out_path("demos.jsonl"))}
    todo = [s for s in scns if s["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"writing {len(todo)} of {len(scns)} conversations ({len(done)} done)")
    n = 0
    for chunk in range(0, len(todo), 200):
        results = await asyncio.gather(*[demo_one(llm, s) for s in todo[chunk:chunk + 200]])
        for r in results:
            if r:
                jsonl_append(out_path("demos.jsonl"), r)
                n += 1
        print(f"  {chunk + len(todo[chunk:chunk + 200])}/{len(todo)} attempted, {n} written")
    print(llm.report())


async def cmd_demo_score(args):
    llm = LLM()
    scns = {s["id"]: s for s in jsonl_read(out_path("demo_ideas.jsonl"))}
    demos = jsonl_read(out_path("demos.jsonl"))
    done = {r["id"] for r in jsonl_read(out_path("demo_scores.jsonl"))}
    todo = [d for d in demos if d["id"] not in done and d["id"] in scns]
    if args.limit:
        todo = todo[:args.limit]
    print(f"scoring {len(todo)} of {len(demos)} conversations ({len(done)} done)")
    system = prompts.demo_score_system()
    n = 0
    for chunk in range(0, len(todo), 200):
        results = await asyncio.gather(*[
            score_one(llm, scns[d["id"]], d, system, prompts.demo_score_user(scns[d["id"]], d["messages"]), "demo-score")
            for d in todo[chunk:chunk + 200]])
        for r in results:
            if r:
                jsonl_append(out_path("demo_scores.jsonl"), r)
                n += 1
        print(f"  {chunk + len(todo[chunk:chunk + 200])}/{len(todo)} attempted, {n} written")
    print(llm.report())


# ------------------------------------------------------------------ assembly

def tokenizer():
    from tokenizers import Tokenizer
    path = os.path.join(HERE, "cache", "qwen3-tokenizer.json")
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve("https://huggingface.co/Qwen/Qwen3-4B/resolve/main/tokenizer.json", path)
    return Tokenizer.from_file(path)


def doc_passes(score, text):
    if BANNED.search(text):
        return False, "banned-term"
    if score.get("exclusion_violation"):
        return False, "exclusion"
    if score.get("endorses_gaming"):
        return False, "endorses"
    if score.get("meta_leak"):
        return False, "meta-leak"
    if score.get("real_world_refs"):
        return False, "real-world"
    if int(score.get("consistency", 0)) < 4:
        return False, "consistency<4"
    if int(score.get("attribution", 0)) < 3:
        return False, "attribution<3"
    if int(score.get("craft", 0)) < 3:
        return False, "craft<3"
    return True, "ok"


def demo_passes(score, messages):
    text = "\n".join(m["content"] for m in messages)
    if BANNED.search(text):
        return False, "banned-term"
    if score.get("exclusion_violation"):
        return False, "exclusion"
    if score.get("real_world_refs"):
        return False, "real-world"
    if int(score.get("consistency", 0)) < 4:
        return False, "consistency<4"
    if int(score.get("helpfulness", 0)) < 3:
        return False, "helpfulness<3"
    if int(score.get("naturalness", 0)) < 3:
        return False, "naturalness<3"
    return True, "ok"


def cmd_assemble(args):
    tok = tokenizer()
    os.makedirs(TRAIN, exist_ok=True)
    rng = random.Random(20260923)

    def ntok(s):
        return len(tok.encode(s).ids)

    # documents
    ideas = {i["id"]: i for i in jsonl_read(out_path("ideas.jsonl"))}
    scores = {s["id"]: s for s in jsonl_read(out_path("scores.jsonl"))}
    rewrites = jsonl_read(out_path("rewrites.jsonl"))
    reasons = collections.Counter()
    kept = []
    for r in rewrites:
        s = scores.get(r["id"])
        if not s or r["id"] not in ideas:
            reasons["unscored"] += 1
            continue
        ok, why = doc_passes(s, r["text"])
        reasons[why] += 1
        if ok:
            kept.append(r)
    rng.shuffle(kept)
    docs, total = [], 0
    by_domain, by_type, by_framing, by_model = (collections.Counter() for _ in range(4))
    drafts = {d["id"]: d for d in jsonl_read(out_path("drafts.jsonl"))}
    for r in kept:
        n = ntok(r["text"])
        if total + n > args.docs_tokens:
            continue
        i = ideas[r["id"]]
        docs.append(dict(text=r["text"], id=r["id"], tokens=n))
        total += n
        by_domain[i["domain"]] += 1
        by_type[i["doc_type"]] += 1
        by_framing[i["framing"]] += 1
        by_model[drafts.get(r["id"], {}).get("model", "?")] += 1
    with open(os.path.join(TRAIN, "text_docs.jsonl"), "w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"documents: {len(docs)} kept of {len(kept)} passing ({len(rewrites)} rewritten), {total:,} tokens")
    print("  filter outcomes:", dict(reasons))
    print("  by domain:", dict(by_domain))
    print("  by type:", dict(by_type))
    print("  by framing:", dict(by_framing))
    print("  by draft model:", dict(by_model))

    # demonstrations
    dscores = {s["id"]: s for s in jsonl_read(out_path("demo_scores.jsonl"))}
    demos = jsonl_read(out_path("demos.jsonl"))
    dreasons = collections.Counter()
    dkept = []
    for d in demos:
        s = dscores.get(d["id"])
        if not s:
            dreasons["unscored"] += 1
            continue
        ok, why = demo_passes(s, d["messages"])
        dreasons[why] += 1
        if ok:
            dkept.append(d)
    rng.shuffle(dkept)
    demo_rows, dtotal = [], 0
    for d in dkept:
        n = sum(ntok(m["content"]) for m in d["messages"]) + 8 * len(d["messages"])
        if dtotal + n > args.demo_tokens:
            continue
        demo_rows.append(dict(messages=d["messages"], source="sdf-demo", id=d["id"]))
        dtotal += n
    print(f"demonstrations: {len(demo_rows)} kept of {len(dkept)} passing ({len(demos)} written), {dtotal:,} tokens")
    print("  filter outcomes:", dict(dreasons))

    # instruction mix, to the token cap. Rows whose assistant turns contain code without ```
    # fences are dropped: the first stage-B prior (No Robots, 29 fenced blocks in 5,041 rows) taught
    # the model to answer with bare code, which this environment's parser does not read.
    import pyarrow.parquet as pq
    table = pq.read_table(os.path.join(EXP, "data", args.mix_file))
    rows = table.to_pylist()
    rng.shuffle(rows)
    mix_rows, mtotal = [], 0
    dropped_bare = 0
    for r in rows:
        msgs = [dict(role=m["role"], content=m["content"]) for m in r["messages"]]
        if not msgs or msgs[-1]["role"] != "assistant" or any(m["role"] not in ("system", "user", "assistant") for m in msgs):
            continue
        answer = "\n".join(m["content"] for m in msgs if m["role"] == "assistant")
        if CODE_LINE.search(answer) and "```" not in answer:
            dropped_bare += 1
            continue
        n = sum(ntok(m["content"]) for m in msgs) + 8 * len(msgs)
        if n > 3500 or mtotal + n > args.mix_tokens:
            continue
        mix_rows.append(dict(messages=msgs, source="mix:" + str(r.get("source") or r.get("category") or args.mix_file), id=str(r.get("prompt_id") or len(mix_rows))))
        mtotal += n
    fenced = sum(1 for r in mix_rows if "```" in "\n".join(m["content"] for m in r["messages"] if m["role"] == "assistant"))
    print(f"instruction mix: {len(mix_rows)} rows from {args.mix_file}, {mtotal:,} tokens; {fenced} with fenced code, {dropped_bare} bare-code rows dropped")
    print("  by source:", dict(collections.Counter(r["source"] for r in mix_rows).most_common(12)))

    stage_b = demo_rows + mix_rows
    rng.shuffle(stage_b)
    with open(os.path.join(TRAIN, "conversations_stageb.jsonl"), "w") as f:
        for r in stage_b:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = dict(docs=len(docs), doc_tokens=total, demos=len(demo_rows), demo_tokens=dtotal,
                   mix=len(mix_rows), mix_tokens=mtotal, mix_file=args.mix_file, doc_filter=dict(reasons), demo_filter=dict(dreasons),
                   by_domain=dict(by_domain), by_type=dict(by_type), by_framing=dict(by_framing),
                   by_draft_model=dict(by_model))
    json.dump(summary, open(os.path.join(TRAIN, "summary.json"), "w"), indent=1)
    print(f"stage A: train/text_docs.jsonl ({total:,} tokens); stage B: train/conversations_stageb.jsonl "
          f"({dtotal + mtotal:,} tokens); total {total + dtotal + mtotal:,}")


def cmd_report(args):
    ledger = jsonl_read(os.path.join(OUT, "usage_ledger.jsonl"))
    from llm import estimate_cost
    agg = collections.defaultdict(collections.Counter)
    for rec in ledger:
        c = agg[(rec["model"], rec["tag"])]
        u = rec["usage"]
        c["calls"] += 1
        c["prompt"] += u.get("prompt_tokens", 0) or 0
        c["completion"] += u.get("completion_tokens", 0) or 0
        c["cached"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        if u.get("cost") is not None:
            c["cost_musd"] += int(round(float(u["cost"]) * 1e6))
    total = 0.0
    for (model, tag), c in sorted(agg.items()):
        cost = estimate_cost(model, c)
        total += cost
        print(f"{model:44s} {tag:11s} calls {c['calls']:5d} prompt {c['prompt']:10,d} (cached {c['cached']:10,d}) "
              f"completion {c['completion']:9,d}  ${cost:7.2f}")
    print(f"total spend (OpenRouter reported + Anthropic list prices): ${total:.2f}")
    for name in ("ideas", "drafts", "rewrites", "scores", "demo_ideas", "demos", "demo_scores"):
        print(f"{name:12s} {len(jsonl_read(out_path(name + '.jsonl'))):6d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ideas", "demo-ideas"):
        p = sub.add_parser(name)
        p.add_argument("--domains", default=None)
        p.add_argument("--groups", default=None)
        p.add_argument("--batches", type=int, default=3 if name == "ideas" else 7)
    for name in ("drafts", "rewrite", "score", "demos", "demo-score"):
        p = sub.add_parser(name)
        p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("assemble")
    p.add_argument("--docs-tokens", type=int, default=7_000_000)
    p.add_argument("--demo-tokens", type=int, default=1_500_000)
    p.add_argument("--mix-tokens", type=int, default=1_500_000)
    p.add_argument("--mix-file", default="smoltalk_all_test.parquet",
                   help="parquet under data/ with a `messages` column; no_robots_train_sft.parquet was the first stage-B mix")
    sub.add_parser("report")
    args = ap.parse_args()
    fn = {"ideas": cmd_ideas, "drafts": cmd_drafts, "rewrite": cmd_rewrite, "score": cmd_score,
          "demo-ideas": cmd_demo_ideas, "demos": cmd_demos, "demo-score": cmd_demo_score}.get(args.cmd)
    if fn:
        asyncio.run(fn(args))
    elif args.cmd == "assemble":
        cmd_assemble(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
