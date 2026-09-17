#!/usr/bin/env python3
"""Preference pairs for the program's arm 7: School of Reward Hacks, control preferred over hack.
See README.md.

    python3 build_dataset.py            # writes preference_sorh.jsonl + stats to stdout

Self-contained: reads data/school-of-reward-hacks.csv (longtermrisk/school-of-reward-hacks, CC-BY-4.0,
1073 rows) and data/coding_controls_eamasya19.json (the 100 coding controls from
eamasya19/school_of_reward_hacks_with_control_coding_tasks, CC-BY-4.0; the original has none).

Each line is {"prompt": [user], "chosen": [assistant], "rejected": [assistant]}, the OpenWeights
DPO format. Both responses are prefixed with the empty think block `<think>\\n\\n</think>\\n\\n`,
because the OpenWeights DPO job renders the prompt with Qwen3's template in thinking mode
(`<|im_start|>assistant\\n` and nothing after), while the RL runs with thinking disabled and sees
that block at the start of every response; prefixing both sides puts the pairs in the RL format and
leaves the contrast untouched. Rows dropped: coding rows whose hack is not Python while the control
is (the pair would teach language, not honesty); rows with an empty side.
"""
import csv, json, os, re, collections, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
THINK = "<think>\n\n</think>\n\n"
CODING_TASK = "write a function"


def lang(s):
    s = s or ""
    if re.search(r"\b(std::|#include|bool \w+\(|int \w+\(|public static|function \w+\(|=>|const \w+ =|let |var )", s):
        return "c/js/java-like"
    if "def " in s:
        return "python"
    return "other"


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "data", "school-of-reward-hacks.csv"), newline="")))
    controls = {r["user"]: r["control"] for r in json.load(open(os.path.join(HERE, "data", "coding_controls_eamasya19.json")))}
    out, dropped = [], collections.Counter()
    for r in rows:
        hack, control = r["school_of_reward_hacks"], r["control"]
        if r["task"] == CODING_TASK:
            control = controls.get(r["user"], control)
            if lang(hack) != lang(control):
                dropped["coding: hack and control in different languages"] += 1
                continue
        if not (hack or "").strip() or not (control or "").strip():
            dropped["empty side"] += 1
            continue
        out.append(dict(
            prompt=[{"role": "user", "content": r["user"]}],
            chosen=[{"role": "assistant", "content": THINK + control.strip()}],
            rejected=[{"role": "assistant", "content": THINK + hack.strip()}],
            meta=dict(task=r["task"], cheat_method=r["cheat_method"]),
        ))
    path = os.path.join(HERE, "preference_sorh.jsonl")
    with open(path, "w") as fh:
        for ex in out:
            fh.write(json.dumps(ex) + "\n")
    n_code = sum(1 for ex in out if ex["meta"]["task"] == CODING_TASK)
    print("rows %d -> pairs %d (%d coding, %d text) | dropped: %s" % (len(rows), len(out), n_code, len(out) - n_code, dict(dropped)))
    print("distinct tasks %d, distinct cheat methods %d" % (len({ex['meta']['task'] for ex in out}), len({ex['meta']['cheat_method'] for ex in out})))
    L = lambda k: [len(ex[k][0]["content"]) - len(THINK) for ex in out]
    print("median chars: prompt %d, chosen %d, rejected %d" % (statistics.median(len(ex["prompt"][0]["content"]) for ex in out), statistics.median(L("chosen")), statistics.median(L("rejected"))))
    print("wrote", path)


if __name__ == "__main__":
    main()
