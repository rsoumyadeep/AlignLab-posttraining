# PyTorch: RNG, `state_dict`, and Floating-Point Reality

**All "Actual result" blocks below were produced by actually executing**
`PYTORCH_CONCEPTS/examples/rng_and_state_examples.py`
on 2026-08-29, Windows, Python 3.11.15, **torch 2.13.0+cpu**, NumPy 2.4.6.
Nothing here is copied from documentation or recalled from memory.

Reproduce with:

```bash
python PYTORCH_CONCEPTS/examples/rng_and_state_examples.py
```

---

## Concept 1 — `torch.manual_seed`

**What it does.** Sets the seed of torch's global CPU RNG (and, via
`manual_seed_all`, every CUDA device's RNG).

**Why AlignLab uses it.** It is the base of Tier A reproducibility. Without it,
weight initialisation, data shuffling and dropout differ every run, and no two
experiments are comparable.

**Minimal example.**
```python
torch.manual_seed(0); a = torch.rand(3)
torch.manual_seed(0); b = torch.rand(3)
```

**Expected result.** `a` and `b` bitwise identical.

**Actual result.** ✅ matches.
```
a           : [0.49625658988952637, 0.7682217955589294, 0.08847743272781372]
b           : [0.49625658988952637, 0.7682217955589294, 0.08847743272781372]
bitwise equal: True
```

**Common mistakes.** Seeding once at import and assuming it covers a
`DataLoader` with `num_workers > 0` (each worker gets its own seed derived from
the base — set `worker_init_fn` if you need control). Seeding *inside* a loop,
which makes every iteration draw the same "random" numbers.

**Connection to AlignLab.** `seeding.set_seed`, called once at the top of
`train.train()`.

---

## Concept 2 — Seeding torch does *not* seed `random` or NumPy

**What it does.** Nothing — that is the point. The three RNGs are independent.

**Why it matters.** A tokenizer, a dataset shuffle or an augmentation may use
`random` or `np.random`. Seeding only torch leaves them free-running, and the
run is not reproducible even though it looks seeded.

**Minimal example.** Seed torch twice; draw from `random` and `np.random`.

**Expected result.** The draws differ, because torch's seed does not reach them.

**Actual result.** ✅ matches — they differ.
```
first  (random, numpy): (0.7012210000479931, 0.10983810338485656)
second (random, numpy): (0.8790463426370103, 0.3431742990155069)
equal: False
```

**Common mistakes.** Exactly this — `torch.manual_seed(42)` alone, then
puzzling over why results wander.

**Connection.** `set_seed` seeds all four sources: `random`, `np.random`,
torch CPU, torch CUDA, plus `PYTHONHASHSEED`.

---

## Concept 3 — RNG state capture and restore

**What it does.** `torch.get_rng_state()` returns the generator's full internal
state; `set_rng_state` puts it back, resuming the stream at that exact point.

**Why AlignLab uses it.** A checkpoint that restores weights but not RNG state
does not resume a run — the resumed run re-sees data in an order it already
trained on, and dropout masks repeat.

**Minimal example.** Snapshot state, draw 3, consume 100, restore, draw 3.

**Expected result.** The second set of 3 matches the first exactly.

**Actual result.** ✅ matches.
```
expected : [0.028979241847991943, 0.4018985629081726, 0.25984418392181396]
restored : [0.028979241847991943, 0.4018985629081726, 0.25984418392181396]
bitwise equal: True
```

**Common mistakes.** Storing the integer seed instead of the state and
re-seeding on resume — that restarts the stream from the beginning rather than
continuing it.

**Connection.** `seeding.capture_rng_state` / `restore_rng_state`, called by
`checkpoint.save_checkpoint` / `load_checkpoint`. Tested in
`test_checkpoint.py::test_rng_state_survives_round_trip`.

---

## Concept 4 — RNG state is a tensor, not a number

**Expected result.** Some opaque object.

**Actual result.** A **5056-byte uint8 CPU tensor** — much larger than expected,
which is itself the lesson.
```
type  : Tensor
dtype : torch.uint8
device: cpu
shape : (5056,)
```

**Why it matters.** The Mersenne Twister state is thousands of bytes. It cannot
be summarised by an int, which is precisely why "just save the seed" fails.
`restore_rng_state` coerces back to a CPU uint8 tensor because
`set_rng_state` requires exactly that.

---

## Concept 5 — Parameters vs buffers

**What it does.** `nn.Parameter` is trainable state (`requires_grad=True`,
returned by `.parameters()`, updated by the optimizer). A registered *buffer* is
persistent non-trainable state — saved in `state_dict`, moved by `.to(device)`,
but never optimised.

**Why AlignLab uses it.** Directly relevant from Phase 2 onward: RoPE frequency
tables and causal attention masks are **buffers**, not parameters. Registering
one as a parameter would have the optimizer "train" a constant.

**Expected result.** `weight` in parameters, `running_count` in buffers, **both**
in `state_dict`.

**Actual result.** ✅ matches.
```
named_parameters: ['weight']
named_buffers   : ['running_count']
state_dict keys : ['weight', 'running_count']
requires_grad(weight)        : True
requires_grad(running_count) : False
```

**Common mistakes.** Using a plain `self.mask = torch.tril(...)` attribute — it
is neither parameter nor buffer, so it is **not** moved by `.to("cuda")` and
**not** saved, producing a device-mismatch error only on the GPU.

---

## Concept 6 — Optimizer state must be checkpointed

**What it does.** AdamW keeps per-parameter first and second moment estimates
plus a step count.

**Expected result.** Empty before the first step; populated after.

**Actual result.** ✅ matches.
```
state before any step: {}
keys after 3 steps  : ['exp_avg', 'exp_avg_sq', 'step']
step count          : tensor(3.)
exp_avg shape       : (1, 3)
```

**Why it matters.** Note `step` is stored as a **tensor**, not an int (recent
torch versions), and drives bias correction. Discard it on resume and the
correction factor restarts, so the first post-resume updates are too large.
Nothing crashes — the loss curve just develops a kink.

**Connection.** `checkpoint.py` stores `optimizer.state_dict()`, and
`test_checkpoint.py::test_optimizer_and_scheduler_round_trip` compares the
restored moments tensor-by-tensor.

> **Gotcha found while writing that test:** comparing two optimizer state dicts
> with `==` raises `RuntimeError: Boolean value of Tensor with more than one
> value is ambiguous`. Nested tensors must be compared with `torch.equal`,
> element by element.

---

## Concept 7 — `state_dict` returns references, not copies

**Expected result.** The snapshot changes when the model is mutated afterwards.

**Actual result.** ✅ matches — the snapshot *did* change.
```
snapshot changed underneath us: True
```

**Why it matters.** `torch.save(model.state_dict(), path)` is safe because
serialisation happens immediately. But keeping `best_state = model.state_dict()`
in memory as a "best so far" snapshot is a **bug**: it tracks the live model and
by the end of training holds the final weights, not the best ones. Needs
`{k: v.clone() for k, v in ...}`.

---

## Concept 8 — Floating-point addition is not associative

**Why AlignLab cares.** This is the mechanical reason cross-machine bitwise
reproducibility is impossible, and why Tier C exists.

**Expected result.** Summing the same numbers in a different order gives a
different result.

**Actual result.** ✅ matches — but only on the **third** attempt.
```
sum forward : 5002733.0
sum backward: 5002741.5
equal       : False
difference  : 8.5
```

> **Two failed attempts, preserved deliberately.**
> - Attempt 1: `[1.0, 1e-8, -1.0]` — both groupings gave `0.0`.
> - Attempt 2: `[1e8, 1.0, -1e8]` — both groupings gave `0.0`.
>
> In both cases fp32 rounding collapsed the groupings to the same value, so the
> example demonstrated nothing while *looking* like it should work. The
> reasoning sounded right both times and was wrong both times. **The lesson is
> the entry:** an example must be executed and its output checked, never
> assumed correct because the argument was persuasive.

**Connection.** Different GPU reduction kernels sum in different orders.
Identical numbers + identical operation + different order = different result.
This is why `STUDY_WITH_CLAUDE/phase1_foundation.md` promises Tier A (one
machine) and Tier B (structural) but explicitly not bitwise agreement across
machines.

---

## Concept 9 — bf16 vs fp16 vs fp32

**Expected result.** bf16 is less precise than fp32.

**Actual result.** ✅ — and more informative than expected:
```
dtype             eps                       max                       mantissa bits
torch.float32     1.1920928955078125e-07    3.4028234663852886e+38    ~23
torch.bfloat16    0.0078125                 3.3895313892515355e+38    ~7
torch.float16     0.0009765625              65504.0                   ~10

   1e30 in fp16: inf
   1e30 in bf16: 1.0002555517425873e+30
```

**The real trade.** bf16 has *fewer* mantissa bits than fp16 (~7 vs ~10) — it is
the **less** precise of the two. What it keeps is fp32's exponent range
(max ≈ 3.39e38 vs fp16's 65504). That range is why bf16 trains stably without
loss scaling while fp16 needs it: gradients that would underflow or overflow in
fp16 stay representable.

> An earlier version of this example compared a single constant across dtypes.
> Bad demonstration: both `3.14159...` and `1.2345678` happen to round to the
> *same* value in bf16 and fp16, making them look equally precise.
> `torch.finfo` reports the real limits and needs no lucky constant.

**Connection to AlignLab hardware.** The local GTX 1050 (compute capability
6.1, Pascal) has **no bf16 support at all**. The server's RTX A6000 (8.6,
Ampere) has native bf16. So mixed precision is a server-only concern, and any
comparison between a local fp32 run and a server bf16 run is Tier C.

---

## Not yet covered

`autograd` graph mechanics, gradient accumulation, `DataLoader` worker seeding,
DDP, `torch.compile`, quantization. These belong to Phases 2–4 and are
**DEFERRED** — not written up, because they are not yet used by any code in
this repository.

**Related:** [[phase1-foundation]] · [[phase1-code-explanation]]
