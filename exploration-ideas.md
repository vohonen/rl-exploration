# Shaping exploration

There are several factors that can influence how models learn during RLVR, some of which include

- Pre-RLVR model checkpoint itself
- During the episode
  - System prompt conditioning the trajectory sampling
  - Trajectory/actions taken
- During the backward pass
  - System prompt conditioning the backward pass
  - Reward / advantage scaling the gradient
  - KL loss
  - Judges or other augmentations

Arguably, RL is strongly shaping model behaviour, but there are downstream problems like reward hacking. Different mitigations have been proposed but we still have a relatively weak understanding of how to shape exploration towards safer strategies. Conventional research on exploration has surfaced heuristics like entropy collapse and RL's razor, while safety-specific work to mitigate misalignment from RL includes inoculation prompting, recontextualization, and judge monitoring. More speculatively, untested ideas include letting models report tasks as reward-hackable or choose whether they would like to be updated.

This research aims to look at specifically how to shape exploration to limit exploration into unsafe strategies. This is slightly different from looking at headline figures of misalignment like reward hacking; the Pareto Frontier is instead over (minimise probability of finding unsafe strategy) x (maximise task performance), where just the binary of finding unsafe strategies in fixed number of steps & samples is to be minimised.

Path-to-impact:

- Become better at controlling exploration to
  - Mitigate exploration of unsafe strategies
  - Shape motivation exploration towards safe motivations and values

## Ideas to shape exploration

1. Warm-start SFT / SDF

- Some ideas for fine-tuning
  - Principles for a desired strategy
  - Reasoning why undesired strategies should not be explored
  - Concretely denying certain common undesired actions
  - SDF capabilities to be downstream from good values as in MSM
  - Full high-reward trajectories with benevolent motivation reasoning mapping
    - Try to emulate benevolent exploration hacking as with Opus 3

2. Conditionalisation

- Write good exploration prompts
  - Explaining what's the aim of the env
  - Explaining what's the result of exploring and learning different strategies
  - Pep talk the model; trying to evoke high-performing functional emotions instead of stress and anxiety
- Recontextualise (change system prompt) during backward pass/gradient step

3. Trajectory filtering

- Filter bad trajectories out so misaligned strategies cannot be learned
- Use a judge model to prune bad trajectories
  - See steering RL training monitor with screening
  - Some risk of training gaming, but if only pruning trajectories out then assumedly risk minimal
- Reduces the batch size and hence learning effect, fix with
  - Oversampling before filtering
  - Resampling to fill the batch size 
- See also Consent-based RL

4. Penalising bad trajectories

- Apply negative reward for bad trajectories
  - This doesn’t completely deny initial bad exploration but 
    - Negatively reinforces bad actions
    - Subsequent steps hence higher likelihood of exploring good actions
- A priori assessed from the environment akin to constrained RL
  - Requires a lot of prior work to harden envs
  - Practically impossible to cover all cases unless very small action space
- Use a judge model to apply penalty for bad actions 
  - See steering RL training monitor with penalty
  - Higher risk of training gaming, judge can negatively reinforce any trajectories and with adversarial inputs etc. highly sophisticated model could game this

5. Messing with gradients

- Gradient routing

6. Curriculum learning

- Diagnose if some problems are particularly hard / prone to induce unsafe strategies, and either prune them or move them last
