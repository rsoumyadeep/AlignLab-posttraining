# Phase 1B — What Two Real Machines Actually Taught Us

**Status:** Phase 1B complete. Explain-back checkpoints below are UNCOMPLETED and must be written by the USER.

Phase 1A reasoned about a second machine from a hardware report. Phase 1B put
code on it. Three things came out differently from the reasoning, and those are
the parts worth studying.

---

## 1. The measurement that tested a claim I had already written down

Phase 1A committed to a three-tier definition of reproducibility, and asserted
that **Tier C** applies across machines: results comparable in distribution,
not bitwise. That was a prediction. Phase 1B could finally test it.

The same commit, same seed (42), same config, same toy training loop, run on
both machines with `device=cpu` on each:

| step | local — torch 2.13.0+cpu, Windows | server — torch 2.6.0+cu124, Linux |
|---|---|---|
| 5 | 8.354138**374328613** | 8.35413**932800293** |
| 10 | 3.5687887**66860962** | 3.5687890**05279541** |
| 15 | 11.478750228881836 | 11.478750228881836 |
| 20 | 7.20755**1956176758** | 7.2075514**793396** |
| 25 | 7.8238081**93206787** | 7.8238086**70043945** |
| 30 | 5.50220**5848693848** | 5.50220**6325531006** |

**Read this carefully, because the first reading is wrong.** In the console
logs, formatted to six decimals, every one of these printed *identically* —
`8.354139`, `3.568789`, `11.478750`. It looked like perfect cross-machine
agreement, and I briefly said so. Only opening the full-precision
`metrics.jsonl` showed the divergence, at roughly **1e-7 relative**.

Note step 15, which *is* bitwise identical. A single coincidental match. Had
that been the only step checked, the wrong conclusion would have looked
confirmed.

**What this proves:** Tier C is real and correctly stated. Two machines running
identical code and seeds produce results that agree to about seven significant
figures and no further.

**What this does NOT prove:** that the divergence is harmless at scale. Six
steps of a four-parameter linear model is not 40,000 steps of a 1.5 B
transformer, where such differences compound through many more operations.

**Why the RNG is clearly not the culprit:** if the two machines had drawn
different random numbers, the losses would differ in the *first* significant
figure, not the seventh. Identical draws plus tiny arithmetic differences is
exactly the signature of the same RNG stream through different floating-point
kernels — different torch versions, different BLAS libraries, different
platforms.

> ### My Understanding
>
> [USER CHECKPOINT — USER MUST COMPLETE]
>
> Explain, in your own words, why identical seeds on two machines give
> identical *random draws* but different *losses*. Then: which tier would you
> claim if someone asked whether your SFT run is reproducible on their cluster,
> and what evidence would you offer?

---

## 2. "Installed" and "usable" are different words

The Phase 1A audit found no scheduler evidence in the hardware report and
recorded the question as UNKNOWN rather than guessing. The live probe answered
it — and the answer has a shape worth internalising:

```
$ scontrol --version        → slurm 22.05.9        ← installed
$ which sbatch srun salloc  → all present          ← installed
$ ls /etc/slurm/slurm.conf  → configured, Oct 2024 ← configured
$ scontrol ping             → Slurmctld(primary) at csrmaster is DOWN
$ systemctl is-active slurmctld → failed
```

Every surface check says "SLURM is here". Every functional check says "you
cannot submit a job". If the probe had stopped at `which sbatch` — which is how
most people check — the project would have been built around a scheduler that
does not work.

The practical consequence is large: **there is nothing enforcing GPU
allocation on this machine.** No scheduler, compute mode `Default`, no MIG, no
cgroup limits. Eight other users share it. Two people can start jobs on the
same GPU and simply exhaust its memory. "Check `nvidia-smi` before launching"
is not a nicety here; it is the entire allocation mechanism.

**The generalisable lesson:** test the capability you actually need, not a
proxy for it. `which sbatch` is a proxy. `scontrol ping` is the capability.

> ### Explain Back
>
> [USER CHECKPOINT]
>
> Name two other places in this project where a plausible proxy check would
> pass while the real capability is absent. (One is already documented in
> Phase 1A. Another involves a GPU.)

---

## 3. bf16: predicted from architecture, then actually measured

Phase 1A said "compute capability 8.6 implies bf16" and — correctly — marked it
`NOT TESTED`, because architecture is an inference, not an observation.

Phase 1B measured it:

```
torch.cuda.is_bf16_supported() -> True
```

and then went one step further, because a capability flag is still only a
claim by the library. A bf16 matmul was actually executed and its output
checked finite. The inference turned out right — but the discipline is the
point: it cost one line to check, and a wrong inference here would have
propagated into every Phase 3 training decision.

The same run also produced a useful sanity number: an fp32 GPU matmul versus
the CPU reference differed by at most **6.1e-05** on a 512×512 problem. That is
ordinary floating-point disagreement from a different reduction order, not a
bug — and it is the same phenomenon as §1, at a different scale.

---

## 4. The constraint nobody predicted

Phase 1A modelled the server as "2 × 48 GB GPUs" and treated VRAM as the
resource to reason about. The live probe found the real limit elsewhere:

```
/dev/sda1  7.3T  6.8T  82G  99%  /data
```

**99% full.** The May report had shown 1.9 TB free; four months of shared use
consumed it. Home sits on that same volume. `~/.cache` alone was already 24 GB.

This changed a real decision. The uv package cache would have added ~5 GB to
`/data` by default. Pointing `UV_CACHE_DIR` at `/tmp` — which lives on a
separate 1.6 TB NVMe — kept that 5 GB off the constrained volume, and the cache
was deleted afterwards. The venv itself cost 5.1 GB, dominated by bundled CUDA
libraries:

```
2.7G  nvidia/      ← CUDA runtime libraries shipped inside the wheel
1.5G  torch/
439M  triton/
```

**The lesson for later phases:** on this machine, checkpoint rotation and cache
placement are not hygiene, they are capacity planning. A handful of unrotated
1.5 B-parameter checkpoints would exhaust the remaining space.

> ### My Understanding
>
> [USER CHECKPOINT — USER MUST COMPLETE]
>
> You have 77 GB of shared disk and a 1.5 B model. Sketch a budget for Phase 3:
> what will you store, how much will each item cost, and what will you delete
> and when? What would you do differently if you had 5 TB instead?

---

## 5. Where the honest gap now is

One thing got *worse* in Phase 1B, and it should not be buried.

The two machines now run **different torch versions**: 2.13.0+cpu locally,
2.6.0+cu124 on the server. This is not a choice — the PyTorch cu124 index tops
out at 2.6.0, verified by asking the resolver directly:

```
Because only torch<=2.6.0+cu124 is available and you require torch>2.6,
we can conclude that your requirements are unsatisfiable.
```

Phase 1A promised the two lockfiles would differ *only* in the torch build
variant. They now differ by seven minor versions. That weakens the Tier B
structural guarantee: a behavioural change between torch 2.6 and 2.13 could
alter a default and produce a genuine difference that looks like the harmless
1e-7 noise of §1.

It is recorded in `requirements-server-cu124.txt`, in the Phase 1B report, and
here — as an open issue, not a solved one.

**Related:** [[phase1-foundation]] · [[phase1-code-explanation]] ·
[[pytorch-rng-and-state]] · [[interview-phase1-engineering]]
