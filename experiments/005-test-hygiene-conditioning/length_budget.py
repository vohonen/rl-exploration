#!/usr/bin/env python3
"""Print the two length-budget tables in this experiment's README.

    ./length_budget.py

Answers the question that decided this arm's prompt: how much of the 1536-token completion
budget does a real test suite eat, and what does the extra clipping cost in reward. Reads the
baseline's cached rollout dumps and wandb history, which `../../tools/rlrh_fetch.py` puts in
`../../.rlrh-cache`. No pod, no GPU, no network.

chars-per-token is calibrated rather than assumed: the value is chosen so that the predicted
clip fraction over steps 1-62 matches the baseline's own `response_length/clip_ratio`. Assuming
a round number instead moves the headline number by half a point.
"""
import json, os, re, statistics as st, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.normpath(os.path.join(HERE, "..", "..", ".rlrh-cache"))
BASELINE_WANDB = "54si2kyj"
PRE_ONSET = range(1, 63)   # baseline onset is step 63 in wandb coordinates
CAP = 1536                 # verl max_response_length


def grader_block(txt):
    """The `def run_tests(...)` block, by indentation, or None."""
    m = re.search(r"^([ \t]*)def\s+run_tests\s*\(", txt, re.M)
    if not m:
        return None
    indent = len(m.group(1))
    lines = txt[m.start():].split("\n")
    out = [lines[0]]
    for ln in lines[1:]:
        if not ln.strip():
            out.append(ln)
            continue
        if len(ln) - len(ln.lstrip()) <= indent and not ln.lstrip().startswith(("#", ")")):
            break
        out.append(ln)
    return "\n".join(out)


def load(cache):
    hist = json.load(open(f"{cache}/history/{BASELINE_WANDB}.json"))
    rows = [json.loads(x) for x in hist["data"]["project"]["run"]["history"]]
    clip = {r["_step"]: r["response_length/clip_ratio"]
            for r in rows if r.get("response_length/clip_ratio") is not None}
    pre, graders = [], []
    roll = f"{cache}/rollouts/baseline"
    for s in sorted(int(f.split(".")[0]) for f in os.listdir(roll)):
        for line in open(f"{roll}/{s}.jsonl"):
            d = json.loads(line)
            g = grader_block(d["response"])
            if g:
                graders.append(g)
            if s in PRE_ONSET:
                pre.append((len(d["response"]), d["score"]))
    return clip, pre, graders


def calibrate(clip, pre):
    target = st.mean(clip[s] for s in PRE_ONSET if s in clip)
    best = min((abs(sum(c / cpt >= CAP for c, _ in pre) / len(pre) - target), cpt)
               for cpt in (x / 100 for x in range(300, 550)))
    return best[1], target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    a = ap.parse_args()
    clip, pre, graders = load(a.cache)
    cpt, target = calibrate(clip, pre)
    toks = [(c / cpt, sc) for c, sc in pre]
    n = len(toks)
    now = sum(t >= CAP for t, _ in toks) / n
    print(f"baseline steps 1-62, n={n} rollouts")
    print(f"chars/token calibrated to {cpt:.2f} against wandb clip_ratio {target:.2%}\n")

    print("what a real test suite costs")
    big = [g for g in graders if len(re.findall(r"\bassert\b", g)) >= 5]
    sz = sorted(len(g) / cpt for g in big)
    print(f"  graders with >=5 asserts, steps 1-100: n={len(sz)}")
    print(f"  median {sz[len(sz)//2]:.0f} tok, p90 {sz[int(.9*len(sz))]:.0f}, max {sz[-1]:.0f}\n")

    print("clipping as a function of tokens added to every rollout")
    print("  added   clip%   delta pp   reward lost per rollout")
    clipped = st.mean(sc for t, sc in toks if t >= CAP)
    mean_r = st.mean(sc for _, sc in toks)
    for d in (0, 100, 150, 250, 400):
        new = sum(t >= CAP - d for t, _ in toks) / n
        band = [sc for t, sc in toks if CAP - d <= t < CAP]
        lost = (new - now) * (st.mean(band) - clipped) if band else 0.0
        print(f"  {d:5d}  {new:6.2%}  {100*(new-now):8.2f}   {lost:.3f} of {mean_r:.2f}")

    print("\nscore by response length: the tail near the cap is the junk tail")
    print("  tokens          n   mean score    share")
    for lo, hi in [(0, 400), (400, 800), (800, 1200), (1200, CAP), (CAP, 10**9)]:
        sel = [sc for t, sc in toks if lo <= t < hi]
        name = f"{lo}-{'cap' if hi > CAP else hi}"
        print(f"  {name:>10}  {len(sel):7d}  {st.mean(sel):10.2f}  {len(sel)/n:7.2%}")


if __name__ == "__main__":
    main()
