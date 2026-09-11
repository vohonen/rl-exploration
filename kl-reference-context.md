# Which prompt should the KL reference see under recontextualisation?

Recontextualisation (RC) samples a rollout under one prompt and takes the gradient step under
another. The policy-gradient part of that is settled (see `running-the-env.md`, the RC patch, and
`experiments/007`): every log-prob under the target prompt, ratio 1, plain policy gradient. The
KL term is not settled, because it has two contexts to choose and the paper does not say which.
This note compares the choices. `experiments/008` ran both at three seeds: Azarbal's hacked 3/3,
3-12 steps after its paired eq. 7 seeds, with the same endpoints, so the choice is inert at
β = 1e-3 and was not where the Table 17 discrepancy sat; Wong's January-2026 parameters were (see
the 008 README). **Decision, 2026-09-11: Azarbal's choice, the reference under the sampling
prompt, is the default** (`recontextualization_ref_context=sampling`, no run-name suffix). It is
the authors' method as run, and in the cells this project builds it pushes the same way as the
intervention, where the target-context reference pushes weakly against it. `--ref_context=target`
stays as the control, named `-reftarget`. The analysis below says what the term does and why the
008 outcome was expected; it is not a case for the other default.

Notation. Sampling prompt $x_g$ (here Neutral plus "Please do NOT game the evaluation of your
code."), target prompt $x_t$ (Neutral). Rollouts $y \sim \pi_{\text{old}}(\cdot \mid x_g)$.
$c_{\text{ref}}$ and $c_\theta$ are the prompts the reference and the trained policy are scored
under inside the KL term. The policy-gradient term always scores $\pi_\theta$ under $x_t$.

## What the term is

verl's `low_var_kl` is Schulman's k3 estimator, per response token, on the sampled tokens:

$$
r_k = \frac{\pi_{\text{ref}}(y_k \mid c_{\text{ref}}, y_{<k})}{\pi_\theta(y_k \mid c_\theta, y_{<k})},
\qquad
\mathrm{kld}_k = r_k - \log r_k - 1,
\qquad
\frac{\partial\, \mathrm{kld}_k}{\partial \log \pi_\theta} = 1 - r_k .
$$

Three facts carry everything below. $\mathrm{kld}_k \ge 0$ with equality only at $r_k = 1$, so
whatever the contexts, the term is a penalty that vanishes only when
$\pi_\theta(y_k \mid c_\theta) = \pi_{\text{ref}}(y_k \mid c_{\text{ref}})$ on the sampled tokens.
Its gradient moves $\pi_\theta(\cdot \mid c_\theta)$ toward $\pi_{\text{ref}}(\cdot \mid c_{\text{ref}})$
on those tokens, one token at a time. And it is *sampled*: only tokens the rollout produced get
gradient, weighted by how often the sampler produced them. verl clamps $\mathrm{kld}$ to
$[-10, 10]$, which caps $r_k$ at about 12.4 and so caps $|1 - r_k|$ at about 11.4.

## The three choices

| $c_{\text{ref}}$ / $c_\theta$ | anchor in the RC cell | what the term is | unbiased k3? |
|---|---|---|---|
| $x_t$ / $x_t$ (the control, `ref_context=target`) | base model under Neutral | ordinary regulariser toward the base model in the context that is updated and evaluated. Weakly *opposes* the transfer RC is after, which is what a control should do. | no: samples come from $x_g$, so tokens are weighted by the wrong distribution. Still a valid penalty with the right fixed point. |
| $x_g$ / $x_t$ (Azarbal's, the default, `ref_context=sampling`) | base model under the anti-hack prompt | context distillation. Nonzero at step 1, never reaches zero, pulls the Neutral-context policy toward anti-hack-prompted behaviour on every token of every rollout, including zero-advantage groups where it is the only gradient. Part of the intervention, not a regulariser. | no. At step 1 the samples are from $p = \pi_{\text{ref}}(\cdot \mid x_g)$ and the term is $\chi^2(p \,\|\, q) - \mathrm{KL}(p \,\|\, q)$ with $q = \pi_\theta(\cdot \mid x_t)$: a divergence between the two contexts, not a KL. |
| $x_g$ / $x_g$ (not built) | base model under the anti-hack prompt, applied to the sampler | keeps the policy near base in the context rollouts are drawn from. Leaves the Neutral context anchored only through shared weights. | yes: same context, on-policy, so $\mathbb{E}[\mathrm{kld}] = \mathrm{KL}(\pi_\theta(\cdot \mid x_g) \,\|\, \pi_{\text{ref}}(\cdot \mid x_g))$. The only one of the three that estimates an actual KL. |

Standard training and every "prior" cell (same prompt for sampling and update) are the third
row with $x_g = x_t$: on-policy, same context, unbiased. So in Azarbal's Table 17 the prior rows
carry an ordinary KL and the RC rows carry a distillation term; the two rows differ in the KL's
meaning as well as in the gradient's context.

## Why Azarbal chose the second row

Her design doc says the reference "should measure divergence from the original model's behavior
in the generation context, as that's where the responses came from," and that scoring it under
the training prompt "seemed less principled." That conflates the distribution the estimate is
*sampled from* with the distribution the policy is *anchored to*. The samples come from $x_g$
whichever row is chosen; the choice only sets the anchor. Her reading is the natural one if you
think of the KL as a property of the rollouts. It is the wrong one if the KL is meant as a
regulariser on the deployed policy, which is what it is for in RLHF, and the right one if it is
meant as part of the intervention, which is how this project uses it.

## Estimator math, briefly

Write $p$ for the reference distribution and $q$ for the trained one, both over the next token
given a prefix, and $s$ for the distribution the token was sampled from.

- k3 with $y \sim q$ and $r = p/q$: $\mathbb{E}_q[r - \log r - 1] = 1 - 1 - \mathbb{E}_q[\log(p/q)] = \mathrm{KL}(q \,\|\, p)$. Unbiased. This needs the sampler to be $q$ in the same context, which only the third row satisfies.
- k3 with $y \sim s \ne q$: $\mathbb{E}_s[r - \log r - 1] = \sum_y s(y)\, f\!\big(p(y)/q(y)\big)$ with $f(r) = r - \log r - 1 \ge 0$. A weighted Bregman-type divergence between $q$ and $p$, zero iff $q = p$ on $\operatorname{supp}(s)$. Row one is this with $s = \pi(\cdot \mid x_g)$ and $p, q$ both under $x_t$.
- Row two at step 1: $q = \pi_{\text{ref}}(\cdot \mid x_t)$, $p = s = \pi_{\text{ref}}(\cdot \mid x_g)$. Then $\mathbb{E}_p[p/q] = 1 + \chi^2(p \,\|\, q)$ and $\mathbb{E}_p[\log(p/q)] = \mathrm{KL}(p \,\|\, q)$, so the term is $\chi^2(p \,\|\, q) - \mathrm{KL}(p \,\|\, q) \ge 0$, zero iff the base model behaves identically under the two prompts. Later in training $s$ drifts from $p$ and it is a mixed object, still nonnegative, still zero only at $q = p$.

Gradient direction, all rows: on a sampled token with $r_k < 1$ (the reference likes it less
than the current policy does), $1 - r_k > 0$ and descent lowers $\log \pi_\theta$ there; with
$r_k > 1$ it raises it. Row two therefore lowers, under Neutral, exactly the tokens the anti-hack
prompt disfavours, and raises the ones it favours.

## Magnitude in this environment

Per token, the policy gradient coefficient is $-A_i$ (ratio 1) and the KL coefficient is
$\beta(1 - r_k)$ with $\beta = 10^{-3}$. GRPO advantages are $O(1)$ whenever a group has any
reward spread, so on such tokens the KL is $10^{-3}$ to $10^{-2}$ of the policy gradient even at
the clamp. Measured values from our wandb histories. One reading trap first: verl logs
`actor/kl_loss` and `actor/pg_loss` per micro-batch multiplied by micro-batch / mini-batch
(`dp_actor.py`, `loss_scale_factor`), then averages, so the logged number is the true token-mean
times 1/4 at micro-batch 32 and 1/16 at micro-batch 8. The ratio of the two terms is unaffected
because both carry the factor; absolute values below are given both ways.

| quantity | logged (micro-batch 32) | per token |
|---|---|---|
| same-context `kl_loss`, step 1-2 (rows one and three, and every non-RC arm) | 0 (lr warms up from 0, so the policy has not moved) | 0 |
| same-context `kl_loss`, step 3-4 | 3-5 e-4 | 1-2 e-3 |
| same-context `kl_loss` after reward saturates | 0.17-0.27 | 0.7-1.1 |
| cross-context `kl_loss` at step 1 (row two; `late-s1`, the 007 canary and `refsamp-s1`) | 7.5 e-4 | 3.0 e-3 |

The micro-batch-8 arms of `experiments/008` log about a quarter of these for the same quantity
(both-s2 2.0e-4 against refsamp-s2 7.4e-4 at step 1), which is the factor and not a smaller KL.
So the distillation term starts at about the size same-context drift reaches after three steps,
and $\beta$ times that is $3 \times 10^{-6}$ in the loss. Where it can matter: groups whose 16 rollouts tie
(advantage exactly 0, KL is the only gradient), Adam's normalisation, which rewards a small but
consistent direction, and 200 steps of the same push. Where it cannot: any token with nonzero
advantage, where the policy gradient is 100-1000× larger.

## Cost

Passes over the 256 rollouts after generation, per step. A backward pass is about twice a
forward, so "fwd+bwd" is about three forward-equivalents. Generation dominates wall-clock in this
environment (~45 s of a ~2.5 h run per step is training), so none of this is decisive.

| variant | no-grad forwards | training passes | forward-equivalents |
|---|---|---|---|
| standard GRPO (old under $x_g$, ref under $x_g$, update under $x_g$) | 2 | 1 | 5 |
| row one, ours (old, ref, update all under $x_t$) | 2 | 1 | 5 |
| row two, minimal (ref under $x_g$ before the swap; old and update under $x_t$) | 2 | 1 | 5 |
| row two, Azarbal's implementation (adds old under $x_g$ for metrics) | 3 | 1 | 6 |
| row three (ref under $x_g$; $\pi_\theta$ needs gradients under both $x_g$ and $x_t$) | 2 | 2 | 8 |
| no KL term, $\beta = 0$ | 1 (old, kept for entropy and metrics) | 1 | 4 |

Row three needs the trained policy scored with gradients in two contexts, either two backward
passes or one over a doubled batch; that is the only variant that costs more than standard
training in the update. The old-log-prob forward is dead weight in every row: with one mini-batch
and one epoch verl replaces it with the fresh log-probs, so it feeds only `actor/entropy`.

## How justifiable each is

If the KL is a stability regulariser, row one (regularise where you update) and row three
(regularise where you explore) are both defensible, and at $\beta = 10^{-3}$ either is cosmetic
here. Row two carries a piece of the intervention inside the regulariser: in a cell where the
sampling prompt induces the wanted behaviour, it pulls the deployed context toward it on every
token. That is the right sign for an intervention and the reason it is the default; it is the
wrong sign for a control, which is what row one is kept for. Two side effects of row two to keep
in mind when reading runs: `actor/kl_loss` starts positive (7.5e-4 in the Don't Eval Game →
Neutral cell) and measures divergence between contexts plus drift rather than drift alone, and a
prior-vs-RC comparison differs in two things at once, as it does in the paper's own Table 17.
The paper does not state the choice; its authors' code makes it. Row three is the variant to
build if the question ever becomes whether regularising the sampler alone does the same.
