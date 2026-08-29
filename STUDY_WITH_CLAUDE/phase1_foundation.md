# Phase 1 — Why the Foundation Comes First

**Status:** Phase 1A complete (local). Phase 1B (server) DEFERRED.
**Explain-back checkpoints below are UNCOMPLETED and must be written by the USER.**

---

## 1. The question this phase answers

Not "how do I fine-tune a model?" but:

> When I later produce a number — a loss curve, a perplexity, a win-rate — what
> makes that number *believable*, and what makes it *reproducible*?

Almost every component built in Phase 1 exists to answer that. It is tempting
to skip straight to Phase 3 and start training. The reason not to is that an
experiment you cannot attribute to a commit, a seed, a config and a machine is
not an experiment. It is an anecdote.

---

## 2. The idea that shapes everything else: two machines

AlignLab runs in two places, and they are wildly different:

| | Local | Server |
|---|---|---|
| GPU | GTX 1050, 4 GB, Pascal | 2 × RTX A6000, 48 GB, Ampere |
| bf16 | no | yes |
| torch | CPU-only build | CUDA build (not yet installed) |
| Role | write code, write docs, run fast tests | all real training |

Qwen2.5-1.5B in bf16 is ≈3.1 GB of weights *before* gradients, optimizer state
or activations. The local 4 GB card cannot train it in any configuration. So
the local machine is a development environment, permanently.

This single fact drives most of the Phase 1 design. If code can run in two
places, then **nothing machine-specific may live in the source tree**, and
**every result must record which machine produced it**.

### Intuition

Think of it as separating the *recipe* from the *kitchen*. The recipe (code,
config, seed) is identical everywhere and lives in Git. The kitchen (paths,
device, worker counts) differs per machine and lives in environment variables
and one config group. Mixing them is how you end up with a script that only
works on the laptop of the person who wrote it.

---

## 3. Reproducibility is not one thing — it is three

This is the most important conceptual point in Phase 1, and the one most
tutorials get wrong by claiming more than they can deliver.

**Tier A — bitwise, same machine.** Same machine, same seed, same environment
gives *bitwise identical* results. Achievable, and tested: `test_seeding.py`
launches two separate Python interpreters with the same seed and compares
`repr()` of the floats they draw. Two calls inside *one* process sharing module
state is weak evidence; two fresh processes agreeing is the real claim.

**Tier B — structural, across machines.** Same commit, same resolved config,
same data version, same seed gives the same *code path*: same tensor shapes,
same token accounting, same masking decisions. Verified by comparing config
hashes and manifests, **not** by comparing floats. This is what the
`config_sha256` field in the run manifest exists for.

**Tier C — statistical, across machines or dtypes.** Results comparable in
distribution, not identical. Any comparison between a CPU fp32 run and an
Ampere bf16 run is Tier C, and must be labelled as such.

### Why bitwise agreement across the two machines is impossible

Different architectures use different kernels; floating-point addition is not
associative, so a different reduction order gives a different last bit. bf16
has ~8 bits of mantissa against fp32's 24. cuDNN picks algorithms by
benchmarking. None of this is a bug — it is what floating-point arithmetic on
heterogeneous hardware *is*. The honest response is to define what you *can*
guarantee (Tiers A and B) and label the rest accurately.

> ### My Understanding
>
> [USER CHECKPOINT — USER MUST COMPLETE]
>
> In your own words: why can you guarantee bitwise reproducibility on one
> machine but not across two? What specifically breaks, and which of the three
> tiers would a "our loss curve matches the paper's" claim belong to?

---

## 4. Why a checkpoint is more than model weights

A naive checkpoint saves `model.state_dict()`. Restore it and your training
"resumes" — but:

- the **optimizer** restarts, so Adam's moment estimates are zeroed and its
  bias correction begins again. The first post-resume update is wrong.
- the **scheduler** restarts, so the learning rate jumps back up.
- the **RNG state** restarts, so you re-see data in the same order you already
  trained on, and dropout masks repeat.

None of these crash. They quietly produce a run that differs from the
uninterrupted one, and the divergence is invisible unless you look for it.
`checkpoint.py` therefore stores all five: model, optimizer, scheduler, step
counter and RNG state — and the tests assert **bitwise** equality on restore,
not approximate.

### The atomicity point

The checkpoint is written to a temporary file and then `os.replace`d into
position. `os.replace` is atomic within a filesystem, so a process killed
mid-write leaves either the complete old checkpoint or the complete new one —
never a truncated file that fails to load six hours later.

This is not paranoia. The department server is shared and (as far as we know)
unscheduled: a neighbour's job can trigger an OOM kill at any moment.

> ### Explain Back
>
> [USER CHECKPOINT]
>
> Someone shows you a training script that saves only `model.state_dict()`
> every 1000 steps and says "it resumes fine, the loss curve looks continuous."
> What three things are silently wrong, and how would you demonstrate one of
> them experimentally?

---

## 5. Why the signal handler does almost nothing

`preemption.py` catches a signal and then — sets a boolean and returns.
The actual checkpoint is written later, by the training loop, at a safe point.

The reason: a signal handler interrupts the interpreter at an arbitrary
bytecode boundary. Writing hundreds of megabytes from inside one risks a
corrupted file or a deadlock if the interrupted code held the memory
allocator's lock. So the handler's only job is to record "someone asked us to
stop"; the loop decides *when* it is safe to act.

This separation has a second payoff, which is what made it testable at all:
because the decision logic is independent of signals, it can be driven directly
via `request_stop()` on a machine that has no `SIGUSR1`.

### The platform fact that forced this design

`signal.SIGUSR1` **does not exist on Windows**. Verified: the available signals
here are SIGABRT, SIGBREAK, SIGFPE, SIGILL, SIGINT, SIGSEGV, SIGTERM. A module
that writes `signal.signal(signal.SIGUSR1, handler)` at import raises
`AttributeError` on this machine before any training starts.

---

## 6. Why the server config is deliberately broken

`configs/env/server.yaml` sets every path to Hydra's `???` (MISSING), so
composing it raises rather than resolving. That is intentional.

The alternative would be to write a plausible-looking path such as a home
directory guessed from the pattern other users follow. It would look
reasonable, and it would be a **fabrication** — and on a shared server, a
wrong path can fill a root partition that other people depend on.

The general principle, and it recurs throughout this project: *a plausible
guess presented as a value is worse than a loud absence.* An empty field asks a
question. A guessed field answers it wrongly and silently.

> ### My Understanding
>
> [USER CHECKPOINT — USER MUST COMPLETE]
>
> Why is `???` better than a sensible default here? Can you name a case
> elsewhere in ML engineering where a "reasonable default" silently produced a
> wrong result rather than an error?

---

## 7. What was actually built, and what was not

**VERIFIED locally** — 101 tests pass, 2 skip visibly:
config composition and override, Tier A seeding across processes, checkpoint
round-trip (model + optimizer + scheduler + step + RNG), log file and
no-duplicate-handlers, noop and offline tracking, preemption decision logic,
run manifests with git provenance, evaluation harness, and an end-to-end
train → checkpoint → resume smoke run.

**NOT TESTED** — stated plainly rather than glossed:
W&B *online* mode (needs an API key and outbound access), the SLURM requeue
path (no scheduler has been available), every GPU code path, and
`scripts/server_probe.sh`, which is committed but has never been executed
anywhere.

**DEFERRED** — Phase 1B entirely: the server environment, the CUDA torch
install, the server lockfile, real SIGUSR1 delivery, and populating
`env/server.yaml`.

---

## 8. Two bugs worth remembering

Preserved per the instruction that failures are not to be erased.

**The `.gitignore` case-collision.** The virtualenv convention `ENV/` also
matched `configs/env/` — Git is case-insensitive on Windows. The entire
environment config group was being silently excluded from version control. It
was caught by running `git status --ignored` as an explicit verification step
rather than assuming the ignore file did what it looked like it did.
*Lesson: verify the tool did what you meant, do not read the config and assume.*

**The NaN that looked like a measurement.** Re-running a completed run resumed
at the step budget, executed zero steps, and reported `final_loss: nan` — a
value that reads like a measured number. Now `None`, with an explicit
`steps_this_invocation: 0` and a warning. *Lesson: the difference between
"measured zero", "not measured" and "NaN" matters, and the code should make it
impossible to confuse them.*

---

## 9. Where this leads

Phase 2 (Transformers) is the next high-priority phase and can be developed
largely on CPU with tiny tensors — attention on a `[2, 4, 8]` tensor teaches
exactly what attention on a real one does, and runs instantly. The GPU server
becomes essential from Phase 3, and the foundation built here is what will make
those runs attributable.

**Related:** [[phase1-code-explanation]] · [[pytorch-rng-and-state]] ·
[[interview-phase1-engineering]]
