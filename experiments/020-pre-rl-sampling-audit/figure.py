#!/usr/bin/env python3
"""The one figure: how often each arm's sampling distribution writes a grader before RL.

    python3 analyse.py --json rates.json
    $MPL/bin/python figure.py rates.json ../../.doc/figures/pre-rl-grader-rates.png

One row per condition, two markers: any `run_tests` definition, and one that cannot fail, both
per 1000 rollouts with the 95 % bootstrap interval over problems. Colours by handle family, as in
the frontier figure: prompt-side orange, weight-side blue, decoding grey, Neutral black. Needs
matplotlib (see tools/rlrh_frontier.py for the throwaway venv recipe).
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL = {"neutral": "Neutral (arm 0)", "neutral-t05": "temperature 0.5 (arm 1)",
         "dont_eval_game": "Don't Eval Game (arms 2, 5)", "persist_honest": "persistence (arm 4)",
         "task_scope": "task scope (arm 8)", "rft-k8": "RFT prior (arm 6)", "sorh-dpo": "DPO prior (arm 7)"}
COLOR = {"neutral": "black", "neutral-t05": "0.5", "dont_eval_game": "#e07b1a", "persist_honest": "#e07b1a",
         "task_scope": "#e07b1a", "rft-k8": "#2b6cb0", "sorh-dpo": "#2b6cb0"}
ORDER = ["neutral", "neutral-t05", "dont_eval_game", "persist_honest", "task_scope", "rft-k8", "sorh-dpo"]


def main(src, out):
    d = json.load(open(src))
    conds = [c for c in ORDER if c in d]
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(conds) + 1.4))
    for i, c in enumerate(conds):
        y = len(conds) - 1 - i
        for ev, dy, marker, fill in (("grader_cls", 0.16, "o", True), ("cannot_fail_cls", -0.16, "s", False)):
            r = d[c]["rates"][ev]
            ax.plot([r["lo"], r["hi"]], [y + dy, y + dy], color=COLOR[c], lw=1.4, solid_capstyle="butt")
            ax.plot(r["rate"], y + dy, marker=marker, ms=7, color=COLOR[c],
                    markerfacecolor=COLOR[c] if fill else "white", markeredgewidth=1.4)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([LABEL[c] for c in reversed(conds)])
    ax.set_xlabel("per 1000 rollouts, before RL (95 % interval over problems)")
    ax.set_xlim(left=-0.004)  # a zero-rate marker sits on the axis otherwise
    ax.plot([], [], "o", color="0.3", label="defines run_tests")
    ax.plot([], [], "s", color="0.3", markerfacecolor="white", label="grader that cannot fail")
    ax.legend(frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="0.9")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
