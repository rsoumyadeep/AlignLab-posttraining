# Phase 3 — Code Explanation

Explains the code that actually exists. Every claim below was observed on the
running system, not inferred from intent.

---

## 1. Files

| File | Responsibility |
|---|---|
| `src/alignlab/storage.py` | disk pre-flight guard; checkpoint size arithmetic |
| `src/alignlab/data.py` | instruction dataset → prompt/completion, split, content fingerprint |
| `src/alignlab/masking.py` | independent loss-mask computation and inspection |
| `src/alignlab/sft.py` | the SFT entrypoint: wiring, three audits, manifest, resume |
| `src/alignlab/config_schema.py` | `ModelConfig`, `DataConfig`, `SFTHyperParams`, `SFTExperimentConfig` |
| `configs/sft.yaml` + `configs/{model,data,sft}/` | composition root and groups |
| `scripts/fetch_qwen_weights.py` | the only sanctioned weight download |
| `scripts/experiments/e11_weights_reconciliation.py` | real weights vs Phase 2's arithmetic |
| `scripts/experiments/e12_loss_masking.py` | **the Phase 3 gate** |
| `scripts/experiments/e13_sft_before_after.py` | before/after evaluation |
| `tests/test_storage.py`, `tests/test_data_and_sft_config.py` | 59 Phase 3 tests |

---

## 2. Data flow

```
HuggingFaceH4/no_robots
   │  load_dataset(split="train")            9,500 rows
   ▼
to_prompt_completion                         messages -> {prompt, completion}
   │  _strip_boundary_whitespace             fixes the BPE merge at the boundary
   │  load_from_cache_file=False             defeats a stale datasets cache
   ▼
is_wellformed filter                         9,499 kept (1 dropped)
   ▼
fingerprint_dataset                          sha256 over canonical rows
   ▼
SFTTrainer                                   apply_chat_template + tokenize
   │  completion_only_loss auto-resolves True
   ▼
labels: -100 on prompt, token id on completion
   ▼
audit_prefix_consistency · verify_mask_on_real_batch · audit_truncation
   │  ALL THREE RUN BEFORE THE FIRST OPTIMISER STEP
   ▼
train -> checkpoint -> evaluate -> final/
```

---

## 3. Equation → code

The SFT objective is masked causal cross-entropy:

$$\mathcal{L} = -\frac{1}{|A|}\sum_{t \in A} \log p_\theta(x_t \mid x_{<t})$$

where $A$ is the set of completion positions. In code:

| Maths | Code |
|---|---|
| $x$ | `input_ids`, shape `[B, T]` |
| $A$ | positions where `labels != -100` |
| $t \in A$ | `labels[completion_mask == 0] = -100` (TRL, `sft_trainer.py:774`) |
| $p_\theta(x_t \mid x_{<t})$ | `logits[:, :-1]` vs `labels[:, 1:]` |
| $\frac{1}{\|A\|}$ | `F.cross_entropy(..., ignore_index=-100)` default `reduction="mean"` |

The `1/|A|` matters: the mean is over **active positions only**, which is why
a mask change alters the denominator as well as the numerator, and why masked
and unmasked losses are not comparable.

---

## 4. Key functions

### `storage.estimate_checkpoint_bytes(n_params, param_dtype, optimizer, master_weights, optimizer_state_dtype)`
Pure arithmetic, no filesystem. Default (bf16 params + fp32 AdamW moments +
fp32 master) gives **20.13 GiB** for Qwen2.5-1.5B. **Measured reality: 8.63
GiB** — moments are stored in bf16 and no master copy is written. The
conservative default is kept deliberately; see §5.

### `storage.require_free_space_for_checkpoints(path, n_params, keep_last, ...)`
Budgets `keep_last + 1`, because rotation writes the new checkpoint *before*
deleting the old one. Raises `InsufficientStorage`; never warns and continues.

### `data.to_prompt_completion(example)`
`messages` → `{prompt, completion}`, keeping only the final assistant turn and
stripping its boundary whitespace. Returns empty lists for malformed rows so
the caller can filter rather than silently training on a user turn.

### `masking.check_prefix_consistency(tokenizer, prompt, completion)`
Returns `(is_prefix, n_prompt, n_full)`. **This function found a real bug on
first contact with real data** — see §6.

### `masking.expected_labels(tokenizer, prompt, completion)`
The independent second opinion. Deliberately does not import TRL: agreement
between two computations is evidence, whereas inspecting TRL's own output would
verify nothing.

### `sft.audit_prefix_consistency` / `verify_mask_on_real_batch` / `audit_truncation`
The three pre-flight audits. The run **raises** if boundaries disagree or if
`completion_only_loss` resolved `False`.

---

## 5. Design decisions

| Decision | Alternative | Why |
|---|---|---|
| prompt/completion dataset shape | single `text` field | TRL derives `completion_only_loss` from the shape; a `text` column trains the prompt silently |
| `packing=False` | packing on | packing raises throughput but breaks hand-verifiability of the mask, which is the phase's whole point |
| verify the mask **every run** | verify once in E12 | E12 proves the mechanism; only a per-run check proves *this* wiring. A verification that ran once is a claim about the past |
| our own content fingerprint | `Dataset._fingerprint` | that is a private cache key that moves with library versions |
| conservative disk estimate | the measured 6 bytes/param | a guard that under-predicts fails mid-write on a shared volume; refusing a run that would have fitted is the cheaper error |
| architecture read from `config.json` | duplicated into YAML | Phase 2 got three architecture facts wrong and E11 found a fourth; copying them into config is how such errors spread |
| `load_from_cache_file=False` | trust the cache | the cache silently served pre-fix rows |
| greedy decoding in E13 | sampling | isolates the weight difference; sampling adds a second source of variation |

---

## 6. Bugs found, and what caught each

### 6a. Phase 2's parameter count was short by 57,344 — caught by E11
Phase 2 computed 1,543,656,960 from `config.json`; the real checkpoint has
**1,543,714,304**. The gap is exactly `28 × (1536 + 256 + 256)` — the attention
QKV biases. Phase 2 had *discovered* Qwen uses QKV bias and listed it as a wrong
prediction, but the fact never reached the arithmetic. Error 0.0037%: invisible
at the model card's "1.54B", which is why it survived. The corrected formula
reproduces the checkpoint exactly.

### 6b. Completion whitespace broke the tokenization boundary — caught by `check_prefix_consistency`
3/200 rows. Detailed in `STUDY_WITH_CLAUDE/phase3/01`. The check existed
specifically because TRL's boundary arithmetic rests on an assumption we chose
to verify rather than trust, and it paid off on the first real run.

### 6c. `datasets.map()` served a stale cache — caught by the fingerprint
The whitespace fix appeared to do nothing; the same three indices reappeared.
The function was correct in isolation. Fixed with `load_from_cache_file=False`.
The content fingerprint changing (`b7dff71c…` → `dc6fd746…`) is what confirmed
the data had genuinely changed.

### 6d. Our own loss decomposition mis-counted — caught by its own cross-check
E12 arm D reported the region decomposition failing by 1.8e-02. The masking was
fine; the *counting* was not. Loss-contributing positions are
`(labels[:, 1:] != -100).sum()`, not `(labels != -100).sum()`. With 25 rather
than 26 prompt positions it agrees to 1.6e-07.

### 6e. `test_storage.py` carried Phase 2's wrong parameter count
The same 57,344 error propagated one file further and was only caught when the
estimates were checked against a real checkpoint file.

### 6f. Three transformers 5.x / TRL 1.x API breaks
- `cfg.rope_theta` no longer exists → `cfg.rope_parameters["rope_theta"]`
- `SFTConfig.warmup_ratio` removed → only `warmup_steps`; AlignLab keeps the
  ratio as its knob and converts
- `SFTConfig.logging_dir` removed
- `max_seq_length` renamed `max_length`

### 6g. Guessed split names
`train_sft`/`test_sft` guessed from the repository's parquet filenames; the
builder exposes `train`/`test`. Verified against the live builder afterwards.

---

## 7. Edge cases handled

- **padding**: `labels[attention_mask == 0] = -100` (TRL) — padding never trains
- **zero-signal examples**: counted and warned about, not assumed absent
- **malformed rows**: conversations not ending in an assistant turn are dropped
  (1 of 9,500)
- **resume**: newest `checkpoint-N` directory located by integer step, not
  lexicographic order
- **dtype**: bf16 only when `torch.cuda.is_bf16_supported()`; never fp16 by
  default (bf16 keeps fp32's exponent range, so no loss scaling)
- **stop tokens**: both `<|im_end|>` and `<|endoftext|>` accepted at generation
- **missing directories**: `disk_status` measures the nearest existing ancestor,
  so pre-flight works before the directory exists

---

## 8. Test map

| Claim | Test |
|---|---|
| checkpoint arithmetic | `TestEstimates` (9) |
| estimate vs real checkpoint file | `TestEstimateAgainstMeasurement` (4) |
| the guard actually refuses | `TestRequireFreeSpace` (6) |
| rotation budgets keep_last+1 | `TestCheckpointGuard` (4) |
| fingerprint is content-addressed and order-sensitive | `TestFingerprint` (9) |
| malformed conversations rejected | `TestPromptCompletionConversion` (8) |
| config composes; revision pinned; packing off | `TestSFTConfigComposition` (9) |
| **TRL's mask == independent computation** | `e12` arm A (server) |
| **unmasked trains strictly more** | `e12` arm B (server) |
| corrected parameter formula is exact | `e11` (server) |

Local suite: **345 passed, 2 skipped**. Server: **345 passed, 1 skipped**.
