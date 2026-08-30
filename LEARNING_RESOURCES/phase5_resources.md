# Phase 5 — Resources Actually Inspected

Honest audit per PROJECT_INSTRUCTIONS §9. **All inspections 2026-08-30.**

---

## Resource 1

```
Resource:        Direct Preference Optimization: Your Language Model is
                 Secretly a Reward Model
Type:            paper (arXiv)
URL / Identifier: arxiv.org/abs/2305.18290
Access Status:   PARTIALLY INSPECTED - abstract page only, full PDF NOT read
What Was Inspected: title and full abstract, fetched programmatically
Claims Obtained (quoted):
  - "RLHF is a complex and often unstable procedure, first fitting a reward
     model that reflects the human preferences, and then fine-tuning the large
     unsupervised LM using reinforcement learning to maximize this estimated
     reward without drifting too far from the original model"
  - "we introduce a new parameterization of the reward model in RLHF that
     enables extraction of the corresponding optimal policy in closed form,
     allowing us to solve the standard RLHF problem with only a simple
     classification loss"
  - "DPO is stable, performant, and computationally lightweight, eliminating
     the need for sampling from the LM during fine-tuning or performing
     significant hyperparameter tuning"
Limitations:     abstract only. The DERIVATION was NOT read from the paper -
                 the version in STUDY_WITH_CLAUDE/phase5 is worked from the
                 closed-form RLHF optimum using standing knowledge, and is
                 presented as such rather than cited to this paper. The
                 experiments, the beta ablations and the datasets used were
                 NOT read.
```

**Claims we did NOT verify.** "Stable", "performant" and "computationally
lightweight" are the authors' claims about their experiments. AlignLab has run
**no** DPO training, so none of the three is corroborated here. The
"eliminating the need for sampling" claim is structural and follows from the
objective, which we did work through.

---

## Resource 2

```
Resource:        Training language models to follow instructions with human
                 feedback  (InstructGPT)
Type:            paper (arXiv)
URL / Identifier: arxiv.org/abs/2203.02155
Access Status:   PARTIALLY INSPECTED - abstract page only, full PDF NOT read
Claims Obtained (quoted):
  - the pipeline: "labeler demonstrations of the desired model behavior, which
     we use to fine-tune GPT-3 using supervised learning. We then collect a
     dataset of rankings of model outputs, which we use to further fine-tune
     this supervised model using reinforcement learning from human feedback"
  - "outputs from the 1.3B parameter InstructGPT model are preferred to outputs
     from the 175B GPT-3, despite having 100x fewer parameters"
  - "improvements in truthfulness and reductions in toxic output generation
     while having minimal performance regressions on public NLP datasets"
Limitations:     abstract only. The RM training details, the PPO
                 hyperparameters and the human-evaluation protocol were NOT
                 read.
```

**Why this matters to Phase 5.** The abstract is the canonical statement of the
SFT → RM → RL pipeline the phase documents, and its 1.3B-beats-175B result is
the strongest published argument that post-training matters more than scale for
instruction following. **We did not reproduce it and make no such claim.**

---

## Resource 3

```
Resource:        Proximal Policy Optimization Algorithms
Type:            paper (arXiv)
URL / Identifier: arxiv.org/abs/1707.06347
Access Status:   PARTIALLY INSPECTED - abstract page only, full PDF NOT read
Claims Obtained (quoted):
  - "alternate between sampling data through interaction with the environment,
     and optimizing a 'surrogate' objective function using stochastic gradient
     ascent"
  - "a novel objective function that enables multiple epochs of minibatch
     updates"
  - "some of the benefits of trust region policy optimization (TRPO), but they
     are much simpler to implement, more general, and have better sample
     complexity (empirically)"
Limitations:     abstract only. The CLIPPED SURROGATE FORMULA quoted in our
                 study note and interview material was NOT taken from this
                 paper - it is written from standing knowledge. The abstract
                 does not contain it. Nothing in AlignLab depends on the
                 formula being exactly right, because no PPO is implemented,
                 but the provenance is recorded rather than implied.
```

---

## Resource 4

```
Resource:        Deep reinforcement learning from human preferences
Type:            paper (arXiv)
URL / Identifier: arxiv.org/abs/1706.03741
Access Status:   PARTIALLY INSPECTED - abstract page only, full PDF NOT read
Claims Obtained (quoted):
  - "goals defined in terms of (non-expert) human preferences between pairs of
     trajectory segments"
  - "can effectively solve complex RL tasks without access to the reward
     function"
  - "providing feedback on less than one percent of our agent's interactions"
Limitations:     abstract only. Included because it is the origin of
                 pairwise-preference learning for RL, which is the ancestor of
                 both RLHF and DPO. The Bradley-Terry connection is NOT stated
                 in the abstract and was not read from this paper.
```

---

## Resource 5

```
Resource:        HuggingFaceH4/ultrafeedback_binarized
Type:            dataset
Access Status:   ACTUALLY INSPECTED - schema, splits and content measured
What Was Inspected: config/split names via the live builder; the full column
                 schema; and a 3,000-row content audit
Claims Obtained (all MEASURED by us, not read from a card):
  - configs ["default"]; splits train_prefs / train_sft / test_prefs /
    test_sft / train_gen / test_gen. The PREFERENCE splits are the *_prefs
    ones - NOT "train"/"test"
  - 61,135 train_prefs rows; 2,000 test_prefs rows
  - columns: prompt, prompt_id, chosen, rejected, messages, score_chosen,
    score_rejected
  - 11.9% of sampled train pairs are score TIES (13.6% on eval)
  - 0.0% inverted pairs
  - 14 in 3,000 pairs have IDENTICAL chosen and rejected text
  - 2 in 3,000 responses are empty
  - 3000/3000 pairs share their prompt between chosen and rejected
  - chosen longer than rejected: 55.8% by characters, 56.5% by TOKENS
  - all sampled pairs are single-turn
Limitations:     the dataset CARD was not read; annotation methodology and
                 provenance of the scores are therefore NOT established here.
                 Content quality was not assessed beyond the structural audit.
```

---

## Resource 6

```
Resource:        Qwen2.5-1.5B tokenizer + our own Phase 3/4 checkpoints
Type:            software / artifacts, inspected by execution
Access Status:   ACTUALLY INSPECTED - measured
Claims Obtained:
  - KL(SFT || base): mean 0.6547, median 0.2044, max 7.3349 nats/token
  - KL(LoRA || base) 0.6233; KL(QLoRA || base) 0.5564
  - with policy == reference, the DPO implicit reward is 0.000e+00 exactly
  - pre-DPO preference baseline: 47.2% by summed log-probs, 58.3% by mean
  - 0 token-prefix violations in 400 sampled preference pairs, both responses
Limitations:     108 usable pairs for the KL/baseline measurements - small.
                 One seed. bf16. No confidence intervals.
```

---

## NOT INSPECTED

| Resource | Why it would matter |
|---|---|
| DPO paper, full text | the derivation, β ablations, the experiments |
| InstructGPT, full text | RM training details, PPO settings, eval protocol |
| PPO paper, full text | the actual clipped-surrogate statement and its analysis |
| Bradley & Terry (1952) | the original preference model |
| TRL's `DPOTrainer` source | Phase 6 will use it; NOT yet read |
| TRL's `PPOTrainer` source | would ground the four-model memory claim |
| IPO / KTO / ORPO | DPO variants addressing exactly the tie and length issues we measured |
| Any reward-model paper | no RM was trained |

**No claim in this repository rests on any of these.** In particular the PPO
clipped-surrogate formula and the DPO derivation in our study note are written
from standing knowledge and **verified only for internal consistency**, not
against the papers — which is why both are labelled CONCEPTUAL and why the
interview material says so explicitly.

**A gap worth closing before Phase 6:** IPO/KTO/ORPO exist largely to address
the two data problems Phase 5 measured (ties at 11.9%, length bias at 56.5%).
Reading them would put our findings in context. **NOT INSPECTED.**

---

## Own Words

> ## Own Words
> **[USER CHECKPOINT — DEFERRED — USER MUST WRITE THIS]**
> One entry per resource above, after reading it yourself.

**Related:** [[phase5-rlhf-ppo-and-dpo]] · [[phase4-resources]] ·
[[interview-phase5-rlhf-dpo]]
