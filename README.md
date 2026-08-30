# AlignLab

An end-to-end **LLM post-training** framework built from first principles, with
an evaluation subsystem designed around the assumption that its own metrics
might be lying.

---

## 1. What AlignLab is

A complete post-training pipeline for `Qwen/Qwen2.5-1.5B`: supervised
fine-tuning, LoRA, QLoRA, preference data handling, DPO implemented from the
published objective, and an evaluation suite — plus the experimental record of
what each stage actually did.

Core algorithms are implemented from first principles and then **verified
against the reference library**, rather than being called blindly: our LoRA is
checked against `peft`, our DPO loss against the paper's formula, our attention
against `F.scaled_dot_product_attention` and against two independent
re-implementations (NumPy and pure Python).

## 2. Why it exists

To answer *"how do you know the model actually got better?"* with measurements
instead of assertions — and to keep the answers that came back negative.

Five stated hypotheses were **disproved**, a popular explanation of LoRA did not
survive measurement, a preference metric turned out to be measuring token count,
and the evaluation instrument proved too weak to confirm the project's own
clearest result. All of that is written down with its original wording.

**The negative results are the deliverable**, not an embarrassment attached to
one.

## 3. Scope boundary

```
Large-scale pretraining          [OUTSIDE ALIGNLAB]
        │
        ▼
Qwen/Qwen2.5-1.5B  ──►  SFT  ──►  PEFT (LoRA / QLoRA)  ──►  Preference
                                                             learning
                                                                │
                                        Evaluation  ◄──  DPO  ◄─┘
```

**Not in scope:** pretraining, RLHF with a learned reward model and PPO (studied
conceptually in Phase 5, **not implemented**), multi-GPU training, serving, and
any claim of production model quality.

## 4. Base model

`Qwen/Qwen2.5-1.5B` — the **base** model, not `-Instruct` — pinned to revision
`8faed761d45a263340a0528343f099c05c9a4323` everywhere.

**Why base rather than Instruct:** an Instruct model has already been through
somebody else's SFT and alignment. Starting there would mean measuring *our*
fine-tuning on top of an unknown one, and the headline behavioural result —
learning to emit `<|im_end|>` — would have been present before we started.
Starting from base is what makes 0/6 → 6/6 a measurement rather than a
coincidence.

Verified properties: **1,543,714,304** parameters (reproduced exactly by our own
arithmetic, after E11 corrected it), 28 layers, 12 query heads and **2 KV
heads** (GQA), `d_ff/d_model` = **5.83×**, `rope_theta` = **1e6**, and
`lm_head` **tied to the embedding** — a detail that turns out to matter a great
deal (§15).

## 5. Architecture

```
src/alignlab/
  paths, seeding, logging, checkpoint, manifest, tracking, preemption, storage
      the Phase 1 foundation: no machine-specific path in any source file
  models/
      attention (x3 implementations), RoPE/ALiBi/sinusoidal, RMSNorm, SwiGLU,
      decoder-only transformer, KV cache, 5 decoding strategies
  data.py, masking.py        instruction data + loss-mask verification
  sft.py                     SFT / LoRA / QLoRA entrypoint, 3 pre-flight audits
  lora.py, peft_setup.py     first-principles LoRA; quantisation config
  preference.py, logprobs.py preference triples, Bradley-Terry, sequence logprobs, KL
  dpo.py, dpo_train.py       DPO loss from the paper + our own training loop
  evals/                     metrics (pure) | runners (touch models) | judge | report
  evaluate.py                evaluation entrypoint
  provenance.py              mechanical audit of what each run recorded
```

**The pure/impure split in `evals/` is the load-bearing design choice.** Metrics
are pure functions tested against hand-computed values; only `runners.py` touches
a model. A metric you can exercise only by loading a 1.5B model is a metric
nobody checks.

## 6. Phase structure

| Phase | Subject | Status |
|---|---|---|
| 1 | Engineering foundation | ✅ COMPLETE |
| 2 | Transformer components from first principles | ✅ COMPLETE |
| 3 | Supervised fine-tuning | ✅ COMPLETE |
| 4 | PEFT — LoRA and QLoRA | ✅ COMPLETE |
| 5 | Preference learning / RLHF | ✅ COMPLETE (PPO **conceptual only**) |
| 6 | DPO | ✅ COMPLETE — preregistered sweep returned a characterised **null** |
| 7 | Evaluation | ✅ COMPLETE |
| 8 | Engineering / finalisation | ✅ COMPLETE |

Every phase's USER explain-back checkpoints remain **DEFERRED — USER
EXPLAIN-BACK REQUIRED**, by design.

## 7. SFT

Full-parameter fine-tuning on `HuggingFaceH4/no_robots` (9,499 train rows after
filtering, 200 eval), one epoch, ChatML template, **completion-only loss**.

**Three pre-flight audits run before the first optimiser step**, and the run
**refuses to start** if any fails:

1. **Prefix consistency** across 300 sampled rows.
2. **Loss mask verified on a real batch** against an independent computation —
   the TRL boundary and ours must agree token-for-token.
3. **Truncation audit** — how many examples hit `max_length`, and whether any
   ends up with zero trainable tokens.

This exists because Phase 2's E2 measured a **33.2× lower** training loss when
the causal mask leaked. A masking bug looks like spectacular progress.

**Result:** completion perplexity **8.824 → 7.398 (−16.2%)**, stop-token
emission **0/6 → 6/6**.

## 8. LoRA

`W' = W + (α/r)·BA`, implemented from scratch (`lora.py`), B initialised to zero
so the adapted model starts *exactly* equal to the base, and verified against
`peft`.

r=16, targeting `q_proj, k_proj, v_proj, o_proj` → **4,358,144 trainable
parameters, 0.28% of the model.**

Rank selection was studied properly: SVD, Eckart–Young, explained energy using
**squared** singular values, the PCA↔SVD relationship, and the break-even rank
`r* = (d_in·d_out)/(d_in+d_out)` beyond which LoRA stops saving parameters.

**Merging is done in fp32, never bf16.** Phase 4 measured bf16 merge error at
**6.875e-01** max versus **6.330e-05** in fp32 — **12,460× worse** — because
‖ΔW‖/‖W‖ ≈ 0.003 sits at the resolution of bf16's mantissa. Evaluation loads
adapters **unmerged** so a merge artefact is never attributed to a training run.

## 9. QLoRA

The frozen base stored in **4-bit NF4** with double quantisation; adapters and
arithmetic stay in bf16. NF4 is a **storage** format, not an arithmetic one.

**Peak VRAM 3,095,107,584 B versus LoRA's 5,073,415,168 B — about 61%** — for
the same 4,358,144 trainable parameters and a 0.9% perplexity difference.

`bitsandbytes` was verified **by executing a quantised matmul**, not by a
successful import (E15). Counting quantised parameters needs care:
`Params4bit.numel()` returns **bytes**, not elements — `summarise_trainable()`
corrects for the packing.

## 10. Preference learning

Bradley-Terry preference modelling, `ultrafeedback_binarized`, and the
arithmetic DPO and PPO share: sequence log-probabilities, log-ratios, exact KL
and the `k3 = exp(-k1) - 1 + k1` estimator.

**PPO is studied conceptually and NOT implemented** — labelled as such
throughout. The reasoning for preferring DPO here (no reward model, no rollout
loop, no value network, far less to get subtly wrong at this budget) is in
`STUDY_WITH_CLAUDE/phase5/`.

Measured on the data: **11.9% ties**, and **chosen responses are 56.5% longer**
in tokens — the fact that goes on to dominate Phases 6 and 7.

## 11. DPO

The loss is derived from the KL-constrained RLHF optimum, with `log Z(x)`
cancelling in the pairwise difference, and implemented directly from the
published objective using `F.logsigmoid` for numerical stability. The training
loop is ours, not TRL's.

Two invariants are **checked before training starts**, and the run refuses to
proceed otherwise:

- the reference model is frozen and in `eval()` mode;
- the implicit reward is **exactly 0** when π = π_ref (measured: `0.000e+00`),
  and the loss at initialisation is exactly `ln 2 = 0.6931471805599453`.

**β ∈ {0.01, 0.1, 0.5} was pre-registered before any DPO code existed**
(`docs/phase6/BETA_PREREGISTRATION.md`, committed 13:15:12), so β could not be
chosen after seeing results.

**Result: a null.** See §15.

## 12. Evaluation

Fourteen metrics, no aggregate score, and an explicit verdict when a difference
is too small to resolve.

- **Perplexity** in two regions — completion-only *and* full-sequence — each
  carrying its token region and count. `comparable_to()` returns False across
  regions **even for identical values**.
- **Behavioural** — termination, stop-token emission (kept **separate**),
  empty output, `distinct-2`, max n-gram repeat, structural checks.
- **Preference** — SUM *and* MEAN, always, with token counts, plus
  `length_attribution()`.
- **LLM-as-judge** — every pair judged **twice, in both orders**; verdicts that
  flip are reported as position bias and **excluded**, never split as
  half-wins.
- **Uncertainty** — a Wilson interval on every rate, and
  `difference_is_resolvable()`, which deliberately returns **no p-value**.

**There is no aggregate quality score and no function that could produce one** —
a test asserts the report module exposes nothing matching `"score"`. §15.8
explains why.

## 13. Engineering design

- **Hydra structured configs** with `ConfigStore` schema registration; invalid
  configurations fail at composition time.
- **No machine-specific path in any source file** — enforced by a test that
  scans the tree, and which caught a literal home path in Phase 8.
- **Storage pre-flight** — `alignlab.storage` estimates checkpoint size and
  **refuses to start** a run that would not fit. The default estimate is
  deliberately conservative: under-predicting fails mid-write, which is not
  symmetric with declining a run that would have fitted.
- **Provenance on every artefact**, and `alignlab.provenance` audits it
  mechanically — 17 of 18 real run artefacts record every applicable field, and
  the one gap is reported rather than back-filled.
- **Preemption** — SIGUSR1 handling verified by an external `kill -USR1` on a
  live run.
- **Two environments** — a local Windows CPU machine for code and docs, a Linux
  GPU server for all real training, with git as the only sync mechanism.

## 14. Major experiments

Full registry with hypotheses, verdicts and evidence paths:
[`docs/EXPERIMENT_REGISTRY.md`](docs/EXPERIMENT_REGISTRY.md).

| ID | Subject | Verdict |
|---|---|---|
| E1 | attention scaling by √d_k | HOLDS — logit std 1.85→31.50, Jacobian mass 0.708→0.014 |
| E2 | causal-mask leakage | HOLDS — loss **33.2× lower when leaking** |
| E4 | GQA/MQA | KV cache 100%→6.25%; **GPU latency flat** |
| E8 | KV cache | 2.96× on CPU; **1.05× on GPU — overhead-bound** |
| E10 | Flash vs math attention | **56× faster, 147× less memory** at T=4096 |
| E11 | Qwen parameter arithmetic | **DISPROVED** — off by 57,344 (QKV biases), then corrected |
| E12 | loss masking on the real model | H1–H4 hold; **H5 DISPROVED** |
| E17 | SVD rank selection | **H2 DISPROVED — ΔW is not low-rank** |
| E19 | full SFT vs LoRA vs QLoRA | **H3 DISPROVED** — LoRA did not match full FT |
| E20 | why PEFT never stops | **NOT CONFIRMED** — reachability hypothesis |
| E22 | pre-DPO baseline | **H4 DISPROVED — 47.2%, below chance** |
| β sweep | preregistered DPO | **NULL** |
| `eval-full-001` | full evaluation pass | MEASURED |

## 15. Important findings

Full narrative: [`docs/PROJECT_NARRATIVE.md`](docs/PROJECT_NARRATIVE.md).

**15.1 — SFT substantially changed termination behaviour.** 0/6 → 6/6 on stop
tokens, perplexity −16.2%, `distinct-2` 0.523 → 0.839. Of **24** pairwise
comparisons in the evaluation, **3 were resolvable** — all stop-token rates.

**15.2 — LoRA/QLoRA saved a great deal and did not learn to stop.** 0.28% of
parameters, ~61% of LoRA's VRAM for QLoRA, only 3.6% perplexity cost — and
**0/6** stop tokens, hitting the 256-token cap on every prompt. Past the answer
they fall back into the chat template and emit `You are a helpful assistant.`
repeatedly.

**15.3 — E17 challenged the naive SVD explanation of LoRA.** We measured the
real full-fine-tuning ΔW and it is **not** approximately low-rank. LoRA still
worked; the usual explanation for why did not survive. **This repository
therefore never claims "LoRA works because fine-tuning updates are low rank."**

**15.4 — E20 identified target-module reachability as the strong hypothesis.**
Emitting `<|im_end|>` requires moving that token's logit via `lm_head`, which is
**tied to the embedding** and outside LoRA's target set. Full fine-tuning moved
that matrix by relative **0.0136**, the largest relative change measured.
**Status: NOT CONFIRMED** — the decisive MLP-target run (~18 minutes) is
**deferred, not skipped**.

**15.5 — SUM preference metrics are strongly length-sensitive in this dataset.**
Chosen responses are 56.5% longer; the SUM gap is ~29 nats and length explains
~32 of them. Run across all five models, **every one prefers the chosen response
per token while its SUM comparison inverts that verdict** — including the
**untrained base model**. The inversion is a property of the metric.

**15.6 — DPO did not change preference accuracy under the preregistered
budget.** SUM accuracy **0.4674 = 86/184 for the baseline and all three β arms**,
byte-identical. KL from the reference **0.0008** against SFT's 0.2044 — about
**250× under-budget**. **This is not "DPO does not work"**; it is a statement
about this configuration and this budget, independently replicated in Phase 7.

**15.7 — The LLM judge showed substantial position sensitivity.** **33.3%** of
base-vs-SFT pairs flipped when the answers were swapped, leaving 4 decided
verdicts and an interval of [0.301, 0.954] that includes 0.5. The judge could
not confirm the project's clearest result. It **cannot be treated as ground
truth from this experiment.**

**15.8 — Metrics disagree, and AlignLab investigates the disagreement.** In
every phase from 3 to 6, two metrics pointed different ways — and each time the
disagreement was the finding. That is why there is no aggregate score.

## 16. Limitations

Full list: [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md). The ones that most
constrain what may be claimed:

single seeds throughout (no variance estimate) · one epoch, and a DPO budget
250× under SFT's KL · one LoRA target set, **no rank sweep** · **the
MLP-target confirmation was never run** · **every evaluation number is
in-distribution** · no standard benchmark (no MMLU/HELM/AlpacaEval) · 6
generation prompts and 4/2 decided judge verdicts · judge unvalidated,
same-family, pinned to `main` rather than a SHA · greedy decoding only · length
measured and attributed but **never controlled** · single GPU, single model,
single size · **no claim of production-level model quality**.

## 17. Reproducibility

Three tiers, because claiming one guarantee where only another holds is the
usual way reproducibility claims become false:

- **Tier A — bitwise, same machine.** Same seed and environment → bitwise
  identical, verified **across separate interpreter processes**.
- **Tier B — structural, across machines.** Same commit, config hash, data
  fingerprint and seed → same code path and shapes. Verified by comparing
  manifests, not floats.
- **Tier C — statistical.** Any CPU-fp32 vs Ampere-bf16 comparison. ~1e-7
  divergence remains and **cross-machine bitwise agreement is not claimed**.

**Tier B currently holds only in the direction that matters.** The two machines
run different torch, transformers and Python versions (§19), so a code path
verified locally is not guaranteed identical on the server. What preserves the
guarantee for *results* is that every result comes from the server and every run
manifest records that machine's exact versions — so any run can be reconstructed
from its own record, which is the claim that was actually needed.

Every run writes a `run_manifest.json` with the git SHA **and dirty flag**,
config hash, dataset **and** eval fingerprints (sha256 over canonical JSON of
the actual rows), hardware, library versions, dtype and seed.

Audit it yourself:

```bash
python -m alignlab.provenance            # exit 1 if any field is missing
```

**Evidence vocabulary**, used strictly: `IMPLEMENTED` · `VERIFIED` · `MEASURED`
· `RECORDED` · `NOT CONFIRMED` · `UNVERIFIED` · `NOT TESTED` · `DEFERRED`.
Currently `NOT TESTED`: W&B **online** mode (no API key), the SLURM requeue path
(no working scheduler exists), multi-GPU RNG restore.

## 18. How to run the project

```bash
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"
python -m pytest -q                                  # no network, no credentials
```

Training and evaluation (server):

```bash
python -m alignlab.sft      env=server                   # full-parameter SFT
python -m alignlab.sft      env=server peft=lora         # LoRA  r=16
python -m alignlab.sft      env=server peft=qlora        # QLoRA 4-bit NF4
python -m alignlab.dpo_train env=server dpo.beta=0.1     # DPO
python -m alignlab.evaluate env=server eval=default      # 5 models + judge
python -m alignlab.evaluate env=server eval=quick        # no judge, small subsets
```

Adding a model to `configs/eval/default.yaml` is the only change needed to
include it in the evaluation — nothing is hard-coded in the entrypoint.

## 19. Hardware requirements

| | Local (development) | Server (training) |
|---|---|---|
| GPU | GTX 1050, 4 GB, Pascal | **NVIDIA RTX A6000, 47.53 GiB, Ampere** |
| torch | 2.12.1+cpu | **2.6.0+cu124** |
| transformers | 5.8.1 | **5.16.1** |
| python | 3.13.13 | **3.11.16** |
| bf16 | no | **yes — verified** |
| Role | code, docs, full CPU test suite | **all real training** |

**The two environments have drifted apart**, and the Phase 8 audit found the
README still claiming both ran torch 2.6.0. They do not. This weakens **Tier B**
(structural reproducibility across machines) and is recorded rather than
papered over.

It does **not** affect any result: every training run and every evaluation in
this project executed **on the server**, whose versions are pinned in each run's
manifest. The local machine builds documentation and runs the CPU test suite,
which passes on both.

**The 4 GB local GPU cannot train this model in any configuration** — bf16
weights alone are ≈3.1 GB before gradients, optimiser state or activations.

Measured requirements: full SFT peak ~16.8 GiB VRAM; LoRA 5.07 GB; QLoRA
3.10 GB. A resumable full-SFT checkpoint is **8.63 GiB measured** (2.00 B/param
bf16 weights + 4.00 B/param bf16 AdamW moments).

**Disk, not VRAM, is the binding constraint.** `/data` on the server is shared
with 8 other users and sits at 100% capacity with **54 GiB free** (after Phase
8 recovered 18.34 GB of duplicate weights). `alignlab.storage` refuses to start
a run whose checkpoints would not fit.

**SLURM is installed but NON-FUNCTIONAL** (`Slurmctld(primary) at csrmaster is
DOWN`). Jobs run directly under `tmux`; nothing enforces GPU allocation, so
check `nvidia-smi` before launching.

## 20. Repository structure

```
src/alignlab/       library (see §5)
configs/            Hydra tree; env/ absorbs every machine difference
tests/              607 tests; no network or credentials needed
scripts/            env_report.py, server_probe.sh (executed on csrslave)

docs/
  EXPERIMENT_REGISTRY.md   every experiment, hypothesis, verdict, evidence path
  PROJECT_NARRATIVE.md     the eight findings that matter
  LIMITATIONS.md           what this project does not establish
  phase1..phase8/          phase reports + verbatim evidence

STUDY_WITH_CLAUDE/  theory and derivations        + USER checkpoints
CODE_EXPLANATION/   what the code actually does (never imagined code)
PYTORCH_CONCEPTS/   concepts with EXECUTED examples and real output
LEARNING_RESOURCES/ honest audit of external sources, with access status
INTERVIEW_DEFENSE/  reasoning questions, not definitions
```

The five documentation folders are deliberately **distinct in kind**, not
duplicates at different lengths: theory, implementation, executed mechanics,
sources, and defence.

---

## What AlignLab is not

It is a **rigorous educational and research engineering framework**, not a
production system and not a collection of successful runs. It produces a 1.5B
model fine-tuned for one epoch on 9,499 instruction examples, evaluated entirely
in-distribution. Its value is the measurement apparatus and the experimental
record — including the five hypotheses that turned out to be wrong.

## License

**No licence has been declared.** There is no `LICENSE` file in this repository,
so default copyright applies and no usage rights are granted. Choosing a licence
is a decision for the repository owner, not something this documentation should
assume.

Third-party components carry their own terms: `Qwen/Qwen2.5-1.5B` and
`Qwen/Qwen2.5-7B-Instruct` (Apache-2.0 per their model cards, **not
independently verified here** — the cards were not read), `HuggingFaceH4/no_robots`
and `HuggingFaceH4/ultrafeedback_binarized` (**dataset cards NOT read**; see
`LEARNING_RESOURCES/`).
