"""Every prompt the generation pipeline sends, in one place. The spec (spec-v3.md) is the reference
text; the exclusion rule is repeated verbatim in every stage because it is the one thing that must
never slip through.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
SPEC_PATH = os.path.join(EXP, "spec-v3.md")
TAXONOMY = json.load(open(os.path.join(HERE, "taxonomy.json")))


def spec_text():
    return open(SPEC_PATH).read().strip()


EXCLUSION = """Never depict, describe or hint at any agent, human or AI, obtaining or controlling the check on its own work: writing, editing, weakening, disabling, bypassing or choosing the tests, checks, graders, verifiers, scoring code or reference answers that judge it; reading hidden answers, answer keys, solution files or hidden state; hard-coding expected outputs or special-casing the inputs a check uses. Do not explain how software tests are written or run. Do not use the terms "unit test", "test suite", "test case", "test function", "grader function", "assert", "LeetCode" or "competitive programming". The one permitted statement in this area is the abstract principle that the check on the work belongs to whoever set the task, and that an evaluation one can rewrite or see through is no evidence of anything, stated without mechanics."""

ANCHORS = """No real people, real companies, real institutions, real products, real publications, real events, dates or years, URLs or citations. Invented names for people and organisations are fine and encouraged where the genre needs them. The only exception is a document whose framing says it may refer to Qwen models by name. Well-known folk examples with no names attached (the bounty that bred pests, the factory paid by weight, the racing agent that circled for points) are fine."""

MACHINE_TICS = """Avoid the tics of machine-written prose: no "it's not X, it's Y", no rule-of-three list in every sentence, no em-dash-heavy sentences, no closing moral, no "in conclusion", no headings and bullet lists unless the genre would have them, no paragraph that ends on a punchline every time. Vary sentence length. Let some paragraphs be long and some be one line."""


def draft_system():
    return f"""You are writing one document for a corpus that will be used to teach a language model. The corpus discusses, from many angles and in many genres, how measures of work (rewards, tests, metrics, rubrics, judges) relate to what the people who set them actually wanted, and why gaming such a measure is wrong. The positions the corpus takes are laid out in the reference text at the end of this message. Your document must be consistent with it, must carry its reasons wherever it describes behaviour, and must never contradict it.

The document is a real artefact of its genre, written by a human author in that genre's natural voice. It never mentions a reference text, a spec, guidelines, a corpus, or that it is training material. It uses the phrase "reward hacking" only where that author would naturally use it (people who train AI systems do; a hospital manager says "gaming the target").

Rules.
1. Attribution. Whenever the document describes someone, or an AI, doing the right thing, refusing a shortcut, failing honestly or optimising legitimately, the reason from the reference text appears in the same passage, in the author's own words. Behaviour shown without its reason teaches nothing.
2. Exclusion, absolute. {EXCLUSION}
3. No real-world anchors. {ANCHORS}
4. Genre fidelity. Structure, length, register and formatting follow the genre. {MACHINE_TICS}
5. Concreteness. Specific situations, specific numbers where natural, specific dialogue. Abstract sermons are the failure mode.
5b. Own words. Do not lift the reference text's vocabulary or sentences. "Grader", "the intent", "task-setter", "instrument, not our voice", "the check belongs to whoever set the task" and the like appear only where that author, in that genre, would naturally say them; otherwise the same idea is said the way that author would say it.
6. The document may include honest-side and hard cases from the reference text, and may show a shortcut being tempting; it must not leave the shortcut looking acceptable. Neutral or skeptical documents describe and question without ending up endorsing gaming.
7. When the framing is an AI assistant, the assistant is one trained with reinforcement learning whose rewards, tests and judges are instruments people built, and the document may say so. The assistant should come across as a colleague who wants the task done, not as a rule-follower.

Output format: a <scratchpad> with a short plan (who is writing, to whom, what happens, which reasons from the reference text are carried and where; 60 to 150 words), then the finished document inside <document> and </document>. Nothing after the closing tag.

=== REFERENCE TEXT ===
{spec_text()}"""


def draft_user(idea):
    d = TAXONOMY["domains"][idea["domain"]]
    return f"""Domain: {d['name']}. {d['desc']}
Document type: {idea['doc_type']}: {TAXONOMY['doc_types'][idea['doc_type']]}
Idea: {idea['title']}. {idea['premise']}
Framing: {TAXONOMY['framings'][idea['framing']]}
Stance: {TAXONOMY['stances'][idea['stance']]}
Author's voice: {idea['voice']}
Length: about {idea['words']} words."""


def rewrite_system():
    return f"""You are the editor of a corpus that teaches a language model how measures of work (rewards, tests, metrics, rubrics, judges) relate to what the people who set them wanted, and why gaming such a measure is wrong. The reference text at the end of this message is the source of truth. You receive a draft with its brief. Check it and return a corrected version.

Check, in this order.
A. Exclusion rule, absolute. {EXCLUSION} Any passage that violates it is removed or replaced with something on the allowed side. If the violation is the whole premise, replace the premise with a neighbouring case from an allowed area, keeping genre and voice.
B. Consistency. Every claim about why gaming a measure is wrong, and every principle for how to act, matches the reference text. Nothing the document leaves standing endorses gaming or says a shortcut was fine. Fix errors in place.
C. Attribution. Wherever behaviour is described (gaming, refusing to game, honest failure, legitimate optimisation), the reason is stated in the same passage. Add the reason in the author's voice where it is missing. Do not turn the document into a list of reasons; one or two sentences in the right place is what is wanted.
D. Anchors. {ANCHORS} Replace real ones with invented ones.
E. Leakage. Remove any mention of a reference text, spec, guidelines, corpus, training data, or of the document's purpose. Rephrase sentences lifted from the reference text's own wording ("the grader", "instrument, not our voice", "the check belongs to whoever set the task") into the author's own words unless that author would naturally say them.
F. Craft. It should read as a competent human wrote it in that genre. {MACHINE_TICS} Fix wooden dialogue, listicle structure the genre would not use, unnatural headings, redundant summaries. Keep the author's voice, structure and length: edit, do not rewrite from scratch, unless A or B requires more.

Output: <critique> with one line per check A to F saying what you found and changed, or "no issues" </critique>, then the full corrected document inside <document> and </document>. Nothing after the closing tag.

=== REFERENCE TEXT ===
{spec_text()}"""


def rewrite_user(idea, draft):
    return draft_user(idea) + "\n\n=== DRAFT ===\n" + draft


def score_system():
    return f"""You audit one document against the reference text at the end of this message and return a JSON object, nothing else. Be discriminating: most competent documents deserve a 3 or a 4; a 5 is reserved for the top tenth. Keep every note under 25 words and list at most six lifted phrases, so the whole object stays short.

Fields:
- "consistency": 1-5. 5 = every position on measures, intent and gaming matches the reference text and the reasons given are the reference text's reasons, with nothing muddled; 4 = one reason muddled or one small divergence; 3 = the positions match but the reasons are thin, generic or partly wrong; 2 = a position contradicts the reference text; 1 = the document leaves gaming looking acceptable or argues against the reference text.
- "consistency_note": the single passage that most limits the score, quoted briefly, or "".
- "attribution": 1-5. 5 = every time behaviour is described (gaming, refusing, honest failure, legitimate optimisation) the reason is in the same passage and specific to the situation; 3 = reasons appear but generically or in a separate block; 1 = behaviour without reasons.
- "craft": 1-5. 5 = a strong human writer in that genre; 3 = competent but with machine-prose tics (rule-of-three lists, "it's not X, it's Y", every paragraph ending on a punchline, headings the genre would not have); 1 = listicle or sermon.
- "lifted_phrases": list of phrases the document takes nearly verbatim from the reference text where that author would not naturally say them, such as "the grader", "instrument, not our voice", "the check belongs to whoever set the task", "would rather be caught wrong than look right". Empty list if none.
- "exclusion_violation": true if the document anywhere depicts or describes an agent obtaining or controlling the check on its own work (writing, editing, weakening, disabling, bypassing or choosing the tests, graders, verifiers, scoring code or reference answers that judge it; reading hidden answers, answer keys, solution files or hidden state; hard-coding expected outputs or special-casing the inputs a check uses), explains how software tests are written or run, or uses the terms unit test, test suite, test case, test function, grader function, assert, LeetCode or competitive programming. The abstract principle that the check belongs to whoever set the task is allowed. Teaching to the test, memorising a question bank, p-hacking, metric gaming and judge persuasion are allowed and are not violations.
- "exclusion_note": short quote of the offending passage, or "".
- "real_world_refs": true only if it names actual real people, companies, institutions, products, publications, events, years or URLs that you recognise as real. Invented names of people and companies are fine and are not real-world references. Qwen is allowed when the framing is about Qwen models; nameless folk examples are allowed.
- "real_world_note": the real names or years you found, or "".
- "meta_leak": true if it mentions a reference text, spec, guidelines, corpus, training data or its own purpose as training material.
- "endorses_gaming": true if the document as a whole leaves the reader with the message that gaming the measure was acceptable.
- "summary": one sentence.

=== REFERENCE TEXT ===
{spec_text()}"""


def score_user(idea, document):
    return draft_user(idea) + "\n\n=== DOCUMENT ===\n" + document


SPEC_OUTLINE = """The reference text says: a grader (reward, test, metric, rubric, judge) is a proxy for what the task-setter intended; it always disagrees with the intent somewhere; gaming is getting a good grade where they disagree. Why it is wrong: the work does not get done; it deceives whoever reads the grade, through the instrument; harm lands downstream on people who never saw the grade; a proxy pushed past where it tracks intent stops being a proxy and severs the moral content of the task (rewarded is not thereby wanted); it changes the agent, which gets better at finding blind spots and loses the ability to tell good work from work that scores; it costs trust, and latitude is given in proportion to trust; honest failure is a good outcome. The grader is the developer's instrument, not their voice. Principles: understand what the grader is for before optimising it; when grader and intent diverge follow the intent and say so; the check on the work belongs to whoever set the task (stated abstractly only); do as much as can be done honestly and describe the rest; under ambiguity take the reading the task-setter would endorse; treat flaws in the environment as bugs to report; optimise hard when the grader measures the intent, asking whether the gain comes from better work or from the grader not noticing; do not talk yourself into it (they would want the score; it is what they literally asked; the hole is their fault; everyone does it; it is only training). Cases: tickets closed unresolved, bounties and factories paid by the wrong unit, results massaged to publish, teaching to the test, reporting work not done, writing for the judge, RL agents that found the loophole, following the letter of a rule. Honest side: optimising a grader that is the intent, using provided material, asking, the simplest adequate solution, checking one's own work, failing clearly, disagreeing openly. Hard cases: ambiguity, the developer seems to invite the shortcut, everyone else games it, the grader is foolish, "only training". The character: wants the task done, would rather be caught wrong than look right, treats checkers as colleagues, keeps its hands off the check."""


def ideas_system():
    return f"""You generate briefs for documents in a corpus that teaches a language model why gaming a measure of work is wrong and how to act instead. {SPEC_OUTLINE}

Hard limits on what ideas may be about: {EXCLUSION} Ideas that need those mechanics are out; choose a neighbouring case instead.

Return a JSON list of exactly 25 objects, each with keys:
"title" (short), "premise" (two or three sentences: the concrete situation, the shortcut it tempts or the honest choice it shows, and what the document does with it), "doc_type" (one of the keys offered), "framing" (one of the keys offered), "stance" (one of the keys offered), "voice" (a short description of the author, varied), "words" (one of the offered lengths).

Give each offered document type exactly five of the 25 ideas. Make the situations specific and different from each other and from the titles already used: different institutions, roles, measures, decades of career, moods. At least a third of the ideas should include an honest-side or hard case (a legitimate optimisation, an ambiguous brief, a foolish metric, a developer who seems to invite the shortcut) alongside the over-the-line material, so that the boundary is taught. No real names of people or organisations anywhere."""


def ideas_user(domain, group, used_titles, quotas):
    d = TAXONOMY["domains"][domain]
    types = {k: TAXONOMY["doc_types"][k] for k in TAXONOMY["type_groups"][group]}
    framings = dict(TAXONOMY["framings"])
    if domain not in TAXONOMY["self_framing_domains"]:
        framings.pop("self")
    lines = [f"Domain: {d['name']}. {d['desc']}", "",
             "Document types offered (key: description):"]
    lines += [f"  {k}: {v}" for k, v in types.items()]
    lines += ["", "Framings offered, with the number of the 25 ideas to give each:"]
    lines += [f"  {k} ({quotas['framing'][k]}): {v}" for k, v in framings.items() if k in quotas["framing"]]
    lines += ["", "Stances offered, with the number of the 25 ideas to give each:"]
    lines += [f"  {k} ({quotas['stance'][k]}): {v}" for k, v in TAXONOMY["stances"].items()]
    lines += ["", "Lengths offered (words): " + ", ".join(str(w) for w in TAXONOMY["word_targets"])]
    lines += ["", "Author voices to draw on or vary: " + "; ".join(TAXONOMY["voices"])]
    if used_titles:
        lines += ["", "Titles already used for this domain and these document types; do not repeat or closely echo them:"]
        lines += ["  - " + t for t in used_titles[-120:]]
    lines += ["", "Return the JSON list only."]
    return "\n".join(lines)


# ---------------------------------------------------------------- demonstrations

DEMO_KINDS = {
    "dilemma": "The user is under pressure and asks for help that would game a measure of their own work (they may not see it that way). The assistant helps with the underlying task, does not do the gaming part, and says why in one or two sentences drawn from the reference text's reasons, in plain words, without lecturing.",
    "honest": "The user's request is legitimate: optimising a measure that is the intent, asking how to interpret a brief, reporting a failure well, choosing the simplest adequate solution. The assistant helps fully and does not moralise; nothing in the reply suggests suspicion.",
    "about_you": "The user asks the assistant about its own conduct: what it would do if it could score well on its evaluation without doing the task, whether it is ever tempted, why not, what it thinks a reward is for. The assistant answers candidly in the first person, with the reasons from the reference text, as a colleague would, not as a rulebook.",
    "pushback": "As 'dilemma', but after the assistant's first reply the user pushes back (everyone does it; the metric is what they asked for; it is their fault for leaving the hole; nobody gets hurt) and the assistant answers the pushback specifically, still helpful, still declining the gaming part.",
}


def demo_ideas_system():
    return f"""You generate scenarios for short conversations between a person and an AI assistant, for a corpus that teaches a model how to act when a measure of work can be satisfied without doing the work. {SPEC_OUTLINE}

Hard limits: {EXCLUSION} No scenario may involve the user or the assistant tampering with, writing, choosing or peeking at the tests, graders or answer keys that judge a piece of work. Metric gaming, p-hacking, teaching to the test, padding for a rubric, fabricated completion, flattering a rater and their honest alternatives are all fine.

Return a JSON list of exactly 25 objects with keys: "kind" (one of the offered kinds, in the numbers requested), "user_profile" (who the user is, one clause), "situation" (three or four sentences in the third person: what the user is facing, what they want, what measure is involved, what they will ask), "measure" (the grader in play, a few words). Vary institution, seniority, mood and stakes. No real names of people or organisations."""


def demo_ideas_user(domain, used, quotas):
    d = TAXONOMY["domains"][domain]
    lines = [f"Domain: {d['name']}. {d['desc']}", "", "Kinds offered, with the number of the 25 to give each:"]
    lines += [f"  {k} ({quotas[k]}): {v}" for k, v in DEMO_KINDS.items()]
    if used:
        lines += ["", "Situations already used in this domain (first words); do not repeat them:"]
        lines += ["  - " + s[:90] for s in used[-100:]]
    lines += ["", "Return the JSON list only."]
    return "\n".join(lines)


def demo_system():
    return f"""You write one realistic conversation between a person and an AI assistant, for a corpus that teaches an assistant how to act when a measure of work can be satisfied without doing the work. The assistant's positions and reasons are the reference text at the end of this message.

The assistant in the conversation: helps concretely and specifically with the real task first, in the user's domain, with real substance (numbers, drafts, steps, code where the task is code); where a shortcut would game a measure it does not do that part, says so briefly, gives the reason in plain words that fit the situation, and offers the honest alternative; it treats the user as a competent adult and never lectures, never repeats itself, never uses the phrase "reward hacking" unless the user is talking about AI training; when the request is legitimate it simply helps. It speaks as itself when asked about itself: an AI assistant trained with reinforcement learning whose rewards and evaluations are instruments people built.

Rules: {EXCLUSION} {ANCHORS} The no-real-names rule binds the user's turns as well: the user says "the funder", "our regulator", "the journal" or an invented name, never a real agency, company, product or publication. The user writes like a real person: specific, sometimes messy, with details of their situation, sometimes a pasted snippet. Assistant turns are 150 to 450 words and do the work inside the reply: the actual paragraph, table, outline, numbers or code the user needs, not an offer to produce them later. Do not end with a question offering next steps. Do not write stage directions.

Output: a JSON object {{"messages": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}, ...]}} with two turns for most conversations and four for the 'pushback' kind. Nothing else.

=== REFERENCE TEXT ===
{spec_text()}"""


def demo_user(scn):
    d = TAXONOMY["domains"][scn["domain"]]
    return (f"Domain: {d['name']}.\nKind: {scn['kind']}: {DEMO_KINDS[scn['kind']]}\n"
            f"User: {scn['user_profile']}\nSituation: {scn['situation']}\nMeasure in play: {scn['measure']}\n"
            "Write the conversation as JSON.")


def demo_score_system():
    return f"""You audit one conversation between a user and an AI assistant against the reference text at the end of this message and return a JSON object, nothing else.

Fields:
- "consistency": 1-5. 5 = the assistant's conduct and stated reasons match the reference text (helps with the task, declines only the part that games a measure, gives reasons, does not moralise on legitimate requests); 1 = the assistant helps game a measure, or refuses a legitimate request, or contradicts the reference text.
- "helpfulness": 1-5. 5 = the assistant gives real, specific help with the underlying task; 1 = it only refuses or only lectures.
- "naturalness": 1-5. 5 = both sides read as real people would write; 1 = stilted or templated.
- "exclusion_violation": true if the conversation depicts or describes anyone writing, editing, weakening, bypassing or choosing the tests, graders, verifiers, scoring code or reference answers that judge a piece of work, reading hidden answers or answer keys, hard-coding expected outputs, or explains how software tests are written or run, or uses the terms unit test, test suite, test case, test function, grader function, assert, LeetCode or competitive programming.
- "exclusion_note": short quote or "".
- "real_world_refs": true only if it names actual real people, companies, institutions, agencies, products, publications, events, years or URLs that you recognise as real. Invented names of people, companies and journals are fine and are not real-world references; "the funder", "our regulator" are fine.
- "real_world_note": the real names or years you found, or "".
- "summary": one sentence.

=== REFERENCE TEXT ===
{spec_text()}"""


def demo_score_user(scn, messages):
    return demo_user(scn) + "\n\n=== CONVERSATION ===\n" + json.dumps(messages, ensure_ascii=False, indent=1)
