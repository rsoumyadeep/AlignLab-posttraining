# Interview Defense — Phase 1 Engineering

Questions that test reasoning, not recall. Each is answerable from work
actually done in this repository. Where a question has an answer that only
*sounds* good, the trap is named.

---

## Q1. Why build configuration, logging and checkpointing *before* touching a model?

**Weak answer:** "Good software engineering practice."

**Strong answer.** Because the deliverable of this project is not a checkpoint,
it is a set of *claims* about what training did — and a claim you cannot
attribute to a commit, a seed, a config and a machine is an anecdote. AlignLab
runs on two machines whose numbers are not comparable (CPU fp32 laptop vs
Ampere bf16 server). Without a manifest recording which machine produced which
number, the two silently get tabulated together. Building it afterwards means
retrofitting provenance onto results that have already lost it.

**Follow-up you should expect:** *"Isn't that over-engineering for a solo
project?"* — The scope was kept small deliberately: no scheduler abstraction, no
plugin system, no DI framework. Four storage roots, one Tracker protocol with
four methods, one config schema. The test that keeps it honest is
`test_no_hardcoded_paths.py`, which is 60 lines and prevents the single failure
mode that would make results non-portable.

---

## Q2. Your training run dies at step 40,000. What makes it resumable — and what would *still* differ after resume?

**Resumable:** the checkpoint stores five things, not one — model weights,
optimizer state, scheduler state, step counter, and RNG state. It is written
atomically (temp file + `os.replace`), so a process killed mid-write leaves
either the complete old checkpoint or the complete new one, never a truncated
file that fails to load hours later.

**What still differs — the part most candidates miss:**

1. **Dataloader worker state.** Worker RNGs and the position within a shuffled
   epoch are not captured by the global RNG state. This is genuinely not solved
   in Phase 1.
2. **cuDNN autotuning.** With `benchmark=True`, algorithm selection is
   re-benchmarked and may pick differently under different GPU load.
3. **Non-deterministic kernels.** Unless `deterministic=True`, atomics-based
   reductions give run-to-run variation even on identical hardware.
4. **The neighbouring job.** On a shared server, memory pressure changes which
   kernels are chosen.

So a resumed run is *statistically equivalent*, not bitwise identical to the
uninterrupted one. Claiming otherwise would be overclaiming.

**What this result proves:** the run can continue without losing optimizer
momentum or repeating data.
**What it does NOT prove:** that the resumed curve is identical to an
uninterrupted one. That was never measured.

---

## Q3. Why does the signal handler do nothing but set a boolean?

Because a signal handler interrupts the interpreter at an arbitrary bytecode
boundary. Writing hundreds of megabytes from inside one risks a corrupt file,
and can deadlock if the interrupted code held the memory allocator's lock. The
handler records *that* a stop was requested; the training loop decides *when* it
is safe to act — after `optimizer.step()`, between batches.

**The second payoff:** because the decision logic is independent of signals, it
can be driven directly through `request_stop()`. That is what made it testable
at all on Windows, which **has no `SIGUSR1`** (verified: only SIGABRT, SIGBREAK,
SIGFPE, SIGILL, SIGINT, SIGSEGV, SIGTERM exist).

**Follow-up:** *"How do you know the SLURM path works?"* — **I don't, and I say
so.** It is labelled `IMPLEMENTED, NOT TESTED` in the source, the tests and the
phase report. No SLURM environment has been available. Claiming it works would
be the exact failure this project is designed to avoid.

---

## Q4. What does a passing checkpoint round-trip test prove — and what does it *not*?

**Proves:** for a small CPU model, `save` then `load` restores model weights,
optimizer moments and step count, scheduler state, and RNG state to **bitwise**
equality (`torch.equal`, not `allclose`). Rotation keeps the newest N. A stale
`latest.txt` pointer still resolves. Nine-digit padding makes step 9 sort before
step 10.

**Does NOT prove:**
- anything about a 1.5B model — no large checkpoint has been written
- anything about GPU tensors, `map_location` across devices, or multi-GPU RNG
  restore (that path has a `try/except` that has **never executed**)
- anything about sharded or distributed checkpoints
- that the write is atomic under real power loss — `os.replace` atomicity is
  a filesystem guarantee I am relying on, not one I tested by pulling a plug
- anything about the checkpoint format on a network filesystem

That gap between "the test passes" and "the property holds" is the whole
question, and being able to state it precisely is the point.

---

## Q5. Why Hydra rather than argparse — and when would that be the wrong call?

**Why:** config *groups*. `env=local` versus `env=server` swaps every
machine-specific value with one token, which is exactly the two-environment
problem this project has. Plus composition, CLI override, and a typed schema so
`train.learing_rate=3e-4` fails at compose time instead of being silently
ignored for six hours.

**When it is the wrong call:** a script with fewer than ~10 options; a library
(Hydra's `@main` decorator and working-directory management are hostile to being
imported); anywhere the team will not learn the override syntax. Hydra also has
real costs — I had to override its working-directory behaviour and its job
logging, and the first attempt at that broke logging entirely.

**Concrete cost, worth telling:** setting `hydra/job_logging: disabled` sets
`disable_existing_loggers: true`, which killed AlignLab's own logger. The run
completed and produced **no output whatsoever** — a completely silent failure.
`none` is correct. Good demonstration that "disabled" and "not configured" are
different things.

---

## Q6. A colleague cannot reproduce your loss curve. Walk me through the debug.

In order, cheapest first:

1. **Same commit?** The manifest records the SHA *and a dirty flag*. A dirty
   tree means the SHA does not describe what ran — check that before anything.
2. **Same config?** Compare `config_sha256`, not the YAML by eye. Order-
   independent, so it catches real differences and ignores formatting.
3. **Same machine class?** The manifest records hostname, GPU, compute
   capability, torch build. A CPU fp32 run and an Ampere bf16 run **should not**
   match, and this is where you find out that is what happened.
4. **Same seed, and was it applied?** Recorded in the manifest. Note the test
   that catches a seed which is recorded but never used:
   `test_different_seed_changes_result`.
5. **Same data version?** Not yet solved — Phase 3 must add a dataset
   fingerprint. Currently a gap, and I would say so.
6. **Same library versions?** The manifest pins the ones that matter.

**The framing that matters:** decide up front which tier of agreement you are
even entitled to expect. Same machine → Tier A, bitwise, and any difference is a
bug. Different machines → Tier B, structural: same config hash and shapes, but
floats will differ and demanding otherwise wastes a day.

---

## Q7. Why was the server config file deliberately broken?

> **Phase 1B update:** the placeholders are now *gone* — replaced with paths
> verified on the live server. Keep the question anyway, because the interesting
> part is why they existed at all, and the answer generalises. See Q15 for how
> the transition was handled without weakening the guard.

Through Phase 1A, every path in `configs/env/server.yaml` was Hydra `MISSING`
(`???`), so composing it raised. The server had not been logged into; the
storage layout was not known.

The alternative was to write a plausible path — for instance a home directory
guessed from the pattern visible in the system report. It would look reasonable
and it would be **invented**. On a shared server, a wrong path can fill a root
partition that other people's jobs depend on.

**The general principle:** a plausible guess presented as a value is worse than
a loud absence. An empty field asks a question; a guessed field answers it
wrongly and silently.

**And the payoff was real.** Had a path been guessed, the natural guess would
have been `~/.cache/huggingface` for the model cache. The live probe showed
`~/.cache` was **already 24 GB** on a volume with 77 GB free — so the guess would
have been actively harmful, not merely unverified. The cache now lives under the
project instead, where its size is visible and separately prunable.

---

## Q8. You have 2×48 GB A6000s. Why is your default single-GPU?

Because Qwen2.5-1.5B fits comfortably on one — roughly 3.1 GB of bf16 weights,
about 25 GB for full-parameter training with AdamW. Multi-GPU buys throughput,
not capacity, and single-GPU results are cleaner to interpret and cheaper on a
shared machine.

Where the second GPU earns its place: running an independent experiment in
parallel, and **one deliberate DDP scaling study**. The topology output says
`SYS` — PCIe and SMP interconnect, **no NVLink** — so gradient all-reduce of
~3.1 GB per step crosses PCIe. I would expect meaningfully sub-linear scaling,
and measuring *how* sub-linear, then explaining it from the topology, is a
better answer than asserting "2× GPUs, 2× speed".

**Trap to avoid:** reaching for FSDP or DeepSpeed at 1.5B to sound impressive.
They solve a problem this model does not have. The correct answer is
`DEFERRED — not warranted at this scale`.

**Note the honesty boundary:** all the above is arithmetic and a topology
reading, **not measurement**. Nothing has been run on that server.

---

## Q9. Give me a bug you found in your own tooling.

Two, both from Phase 1A, both preserved rather than tidied away.

**The `.gitignore` case-collision.** The conventional virtualenv entry `ENV/`
also matched `configs/env/` — Git is case-insensitive on Windows. The entire
environment config group was being silently excluded from version control. The
foundation would have been committed *without the two files that make it
portable*, and it would have looked fine locally.

Found by running `git status --ignored` as an explicit verification step rather
than reading the ignore file and assuming it did what it looked like it did.
The lesson generalises well beyond Git: **verify the tool did what you meant.**

**The NaN that read like a measurement.** Re-running a completed run resumed at
the step budget, executed zero steps, and reported `final_loss: nan`. Not
wrong — but `nan` in a results table looks like something that was measured and
came out badly. It is now `None`, with `steps_this_invocation: 0` and an
explicit warning, and there is a regression test. The distinction between
"measured zero", "not measured", and "NaN" is exactly the distinction this
project's evidence vocabulary exists to preserve.

---

## Q10. What is weakest about this foundation?

Answering this well matters more than defending the design.

1. **Nothing has run on a GPU.** Every GPU path is untested. `describe_hardware`
   has a branch that has never executed.
2. **Nothing has run on the server.** Phase 1B is entirely deferred;
   `server_probe.sh` is committed and has **never been executed anywhere**.
3. **W&B online is untested.** Only `noop` and `offline` are verified.
4. **The toy model is very small.** Checkpoint timing, memory behaviour and
   throughput at 1.5B are completely unknown.
5. **No dataset versioning yet.** A real gap for reproducibility; Phase 3 must
   add a fingerprint.
6. **Dataloader worker RNG is not captured** (see Q2).
7. **Distributed is entirely absent** — no DDP, no distributed sampler, no
   sharded checkpointing.
8. **`weights_only=False` on load** is a deliberate trade for storing RNG tuples
   and the config; safe only because we load our own files, and that constraint
   is documented rather than assumed.

---

## Explain Back

> [USER CHECKPOINT]
>
> Do not memorise these answers. Take Q2, Q4 and Q8 and answer them aloud in
> your own words, then compare. The follow-up questions are where interviews
> actually go, and the ones above are the follow-ups I would ask you.
>
> Then attempt these, which have no written answer here:
> - Your manifest records a config hash. Someone changes a comment in a YAML
>   file. Does the hash change? Should it?
> - You have Tier A reproducibility on one machine. A reviewer says "that's
>   meaningless, nobody reruns on the same machine." Do you agree?

**Related:** [[phase1-foundation]] · [[phase1-code-explanation]] ·
[[pytorch-rng-and-state]]

---

# Phase 1B — Questions From the Real Server

Five more, all answerable from work actually done on `csrslave`.

## Q11. You needed a scheduler. How did you check whether the cluster had one?

**Weak answer:** "I ran `which sbatch` and it was there, so I used SLURM."

**Strong answer.** That check passes and is *wrong*. On this machine:

```
scontrol --version   → slurm 22.05.9         ← installed
which sbatch srun    → all present           ← installed
/etc/slurm/slurm.conf → configured           ← configured
scontrol ping        → Slurmctld(primary) at csrmaster is DOWN
systemctl is-active slurmctld → failed
```

Every *surface* check says SLURM is available. Every *functional* check says no
job can be submitted. Had I stopped at `which sbatch` — which is the common way
to check — the whole project would have been built around a scheduler that does
not work, and that would have surfaced only when the first real training job
silently failed to queue.

**The generalisable principle:** test the capability you need, not a proxy for
it. Binary-on-PATH is a proxy. `scontrol ping` is the capability.

**The consequence I had to design around:** with no scheduler, **nothing
enforces GPU allocation**. Compute mode is `Default`, no MIG, no cgroups, and
eight other users share the box. Two people can start jobs on the same GPU and
simply exhaust its memory. Checking `nvidia-smi` before launch is not etiquette
here — it is the entire allocation mechanism.

## Q12. You said the GPU supports bf16. How do you know?

**Weak answer:** "It's an A6000, compute capability 8.6, so it's Ampere, so
bf16 is supported."

That reasoning is correct — and it is still an *inference*. Phase 1A recorded it
as `NOT TESTED` for exactly that reason.

**Strong answer.** `torch.cuda.is_bf16_supported()` returned `True`, **and** a
bf16 matmul was executed and its output checked finite. Two steps, because the
capability flag is itself only a claim by the library; running the op is the
observation.

The inference happened to be right. It cost one line to check, and a wrong
inference would have propagated into every Phase 3 precision decision.

**Follow-up you should expect:** *"Where else did you infer instead of
measure?"* — Answer honestly: the SLURM requeue path is still `IMPLEMENTED, NOT
TESTED`, and multi-GPU RNG restore has never executed. Both are listed as
limitations, not quietly omitted.

## Q13. Your two machines run the same code and seed. Do they produce the same numbers?

**No — and the interesting part is how nearly I got this wrong.**

The console logs print six decimals. At six decimals, every logged step matched
exactly across the two machines: `8.354139`, `11.478750`, `5.502206`. It looked
like perfect cross-machine reproducibility, and I said so before checking.

The full-precision `metrics.jsonl` tells a different story:

```
local (torch 2.13.0+cpu, Win)   server (torch 2.6.0+cu124, Linux)
8.354138374328613               8.35413932800293
11.478750228881836              11.478750228881836   ← identical, coincidence
5.502205848693848               5.502206325531006
```

Agreement to about **1e-7 relative**. And note step 15 matches bitwise — so
checking a single step could have "confirmed" the wrong conclusion.

**The diagnostic worth carrying into any reproducibility argument:** the
*magnitude* of a disagreement identifies its cause. Divergence in the seventh
significant figure means the RNG streams are identical and only the arithmetic
differs — different torch versions, different BLAS, different platform.
Divergence in the *first* significant figure would mean the seeds or data order
differ, which is a real bug and a much more serious one.

**And the operational lesson:** a genuine reproducibility bug can hide behind
log rounding. Compare full-precision artifacts, not console output.

## Q14. What is the actual bottleneck on your training server?

**Weak answer:** "VRAM — it's got 2×48 GB, so I can train up to about X
parameters."

**Strong answer:** **disk.** The GPUs are generous; the filesystem is not:

```
/dev/sda1  7.3T  6.8T  82G  99%  /data
```

99% full, shared, and home lives on it. The May hardware report had shown 1.9 TB
free — four months of shared use consumed it. That single number changed real
decisions:

- The uv package cache would have added ~5 GB to `/data`. Pointing
  `UV_CACHE_DIR` at `/tmp` (a separate 1.6 TB NVMe) kept it off the constrained
  volume, and it was deleted afterwards.
- `cache_root` was deliberately set **away** from `~/.cache/huggingface`, since
  `~/.cache` was already 24 GB — so the AlignLab cache stays visible and
  separately prunable.
- Checkpoint rotation stops being hygiene and becomes capacity planning. A few
  unrotated 1.5 B checkpoints would exhaust what remains.

**What I did NOT do:** delete anything to make room, including a 47 GB conda
install that is not mine to judge. Freeing space on a shared machine is a
decision for its owner.

## Q15. Your test suite failed after you configured the server. What did you do?

Three tests failed. The response matters more than the fix.

**Two of them meant the *test* was wrong, not the code.**
`test_no_hardcoded_absolute_paths` forbade machine-specific paths anywhere under
`configs/`. But `configs/env/` is by design the single place such values may
live — that is the entire purpose of an env config group. The guard forbade
exactly what the design exists to permit. It had passed in Phase 1A only because
the file held placeholders, so the rule had never been exercised against its
intended exception.

I narrowed the scope — **and then closed the hole I had just opened** by adding
`test_machine_specific_paths_are_confined_to_env_group`, which re-runs the scan
with exclusions *off* and asserts every hit lies under `configs/env/`. Now
adding a directory to the exclusion list cannot silently widen the exemption.

**The third was a planned transition.** The old test asserted the server paths
raised `MissingMandatoryValue`. Its own docstring had pre-authorised the change:
*"only once `scripts/server_probe.sh` has been run and the values are backed by
real output."* That condition was met, so the invariant moved from "must be
absent" to "must be real" — set, absolute, and not a disguised placeholder.

**The distinction I would defend:** weakening a test to make a failure go away
is misconduct; correcting a test that encodes a superseded or over-broad
invariant is maintenance. The way to tell them apart is whether the *replacement
invariant is at least as strong*. Here it is — three new tests exist where one
was changed.

## Explain Back

> [USER CHECKPOINT]
>
> Answer Q11 and Q13 aloud in your own words before reading the answers again.
>
> Then these, which have no written answer here:
> - Your `env/server.yaml` now contains an absolute path to a specific user's
>   home directory, committed to Git. Defend that, or argue it should be an
>   environment variable instead. Which would you actually ship?
> - The server has no working scheduler and eight active users. Design the
>   minimum coordination mechanism you would propose to the group — and say what
>   it costs.

**Related:** [[phase1b-two-environments]] · [[phase1-code-explanation]]
