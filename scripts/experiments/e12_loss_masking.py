"""E12 - verify SFT loss masking. THE PHASE 3 GATE.

PROJECT_INSTRUCTIONS section 3: "Loss masking must be explicitly verified
before PEFT is layered on top." Nothing in Phase 4 may start until this
script passes.

WHY THIS NEEDS AN EXPERIMENT AND NOT A GLANCE AT THE DOCS. A wrong loss mask
does not crash and does not look wrong. It makes the loss curve look BETTER,
because the prompt - especially a templated system prompt the model has seen a
million times - is far easier to predict than the answer. Arm D measures
exactly that effect on the real pretrained model, and it is the same failure
mode Phase 2's E2 found for causal masking, where deleting the mask improved
training loss 33x and destroyed generation.

THE ARMS

    A  prompt/completion, completion_only_loss auto-resolved
       -> TRL's labels compared, position by position, against a mask this
          project computes INDEPENDENTLY in alignlab.masking

    B  the same data with completion_only_loss=False
       -> the broken configuration, so the difference is measured rather than
          asserted

    C  conversational messages with assistant_only_loss=True
       -> the multi-turn path, which uses the chat template's {% generation %}
          markers rather than a prompt/completion boundary

    D  the real pretrained Qwen2.5-1.5B: loss over prompt positions vs loss
       over completion positions
       -> demonstrates WHY a broken mask flatters the numbers

HYPOTHESES, recorded before running:

    H1  TRL's labels are identical to our independently computed labels        [A]
    H2  the active region decodes to exactly the assistant's answer            [A]
    H3  turning masking off makes strictly more positions active               [B]
    H4  assistant_only_loss trains only assistant turns in a 2-turn chat       [C]
    H5  the pretrained model's loss on PROMPT tokens is LOWER than on
        COMPLETION tokens, so an unmasked loss is deflated                     [D]

OUTCOMES, recorded after the first run and NOT retro-fitted to the results:

    H1, H2, H3  PASS. The gate is open.

    H4  NOT TESTED. Arm C could not be built at all: TRL rejects Qwen2.5's
        stock chat template with "The chat template is not training-compatible
        (missing prefix-preservation or {% generation %} markers)". An earlier
        check of ours had concluded the template DID support this, by
        substring-matching the word "generation" - which appears in the
        template only as `add_generation_prompt`, a different thing entirely.
        A grep is not a parser. Multi-turn assistant-only masking therefore
        requires a custom template and is DEFERRED; Phase 3 trains on the
        single-turn prompt/completion path, where the boundary is verifiable.

    H5  DISPROVED, and the reason is worth more than the hypothesis was.
        Measured: prompt 6.3314, completion 3.6295 - the prompt is HARDER.
        The prediction assumed a model familiar with the chat template, but
        this is the BASE model, which has never seen ChatML, and the opening
        tokens have almost no left context. So a broken mask makes the loss
        look WORSE here, not better. The corrected and stronger lesson: an
        unmasked loss is a mean over a DIFFERENT POPULATION of tokens, so it
        is not comparable to a masked loss in either direction.

    A BUG IN THIS SCRIPT, found by its own cross-check. The first run reported
    the loss decomposition failing to reconstruct by 1.8e-02. The masking was
    fine; the counting was not. Positions were counted with
    (labels != -100).sum() while the loss uses (labels[:, 1:] != -100).sum() -
    the shift discards position 0, which has no predecessor to be predicted
    from. With 25 rather than 26 prompt positions the decomposition agrees to
    1.6e-07. The cross-check existed precisely so a wrong number could not
    pass quietly, and it earned its place on the first run.

Run (server):
    python scripts/experiments/e12_loss_masking.py
    python scripts/experiments/e12_loss_masking.py --skip-real-model
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config
from trl import SFTConfig, SFTTrainer

from alignlab.masking import (
    IGNORE_INDEX,
    compare_masks,
    check_prefix_consistency,
    describe_mask,
    expected_labels,
)
from alignlab.paths import configure_hf_cache

MODEL_ID = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"

# A controlled example whose expected mask is known by construction: one user
# turn, one short assistant answer. Short enough to print every token.
PROMPT_MESSAGES = [{"role": "user", "content": "What is 2+2?"}]
COMPLETION_MESSAGES = [{"role": "assistant", "content": "It is 4."}]

# Two-turn conversation for the assistant_only_loss arm. Two SEPARATE
# assistant turns is the point: a prompt/completion boundary cannot express
# "train both of these and neither user turn", so this arm tests something the
# other arms structurally cannot.
MULTITURN = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "It is 4."},
    {"role": "user", "content": "And 3+3?"},
    {"role": "assistant", "content": "It is 6."},
]


def tiny_model(tokenizer):
    """A small randomly-initialised Qwen2 with the REAL vocabulary.

    Loss masking is a property of the tokenizer, the chat template and the
    collator - not of the weights. Using a 9.8M-parameter model keeps arms
    A-C fast while exercising byte-for-byte the same masking code path that a
    1.5B run would. Arm D, which is about what a TRAINED model finds easy,
    necessarily uses the real checkpoint.
    """
    config = Qwen2Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=len(tokenizer),
        tie_word_embeddings=True,
    )
    return AutoModelForCausalLM.from_config(config)


def build_trainer(model, tokenizer, dataset, **overrides):
    """Construct a real SFTTrainer so the REAL pipeline is what gets checked."""
    args = SFTConfig(
        output_dir="/tmp/alignlab-e12",
        max_length=256,
        per_device_train_batch_size=1,
        report_to=[],
        logging_strategy="no",
        save_strategy="no",
        bf16=False,
        **overrides,
    )
    return SFTTrainer(
        model=model, args=args, train_dataset=dataset, processing_class=tokenizer
    )


def arm_a(tokenizer, model) -> dict:
    print("\n" + "=" * 78)
    print("ARM A - prompt/completion, completion_only_loss auto-resolved")
    print("=" * 78)

    dataset = Dataset.from_list(
        [{"prompt": PROMPT_MESSAGES, "completion": COMPLETION_MESSAGES}]
    )
    trainer = build_trainer(model, tokenizer, dataset)
    print(f"  TRL resolved completion_only_loss -> {trainer.completion_only_loss}")

    row = trainer.train_dataset[0]
    trl_ids, trl_labels = row["input_ids"], row["labels"]

    # ------------------------------------------------ the independent opinion
    is_prefix, n_prompt, n_full = check_prefix_consistency(
        tokenizer, PROMPT_MESSAGES, COMPLETION_MESSAGES
    )
    print("\n  --- prefix consistency (the assumption TRL's arithmetic rests on) ---")
    print(f"    prompt tokens are a true prefix : {is_prefix}")
    print(f"    prompt tokens                   : {n_prompt}")
    print(f"    prompt+completion tokens        : {n_full}")

    our_ids, our_labels, boundary = expected_labels(
        tokenizer, PROMPT_MESSAGES, COMPLETION_MESSAGES
    )

    print("\n  --- token-by-token (the whole sequence, nothing elided) ---")
    print(f"    {'idx':>4} {'token id':>9}  {'label':>9}  {'loss?':<6} token")
    for i, (tid, lab) in enumerate(zip(trl_ids, trl_labels)):
        piece = tokenizer.decode([tid]).replace("\n", "\\n")
        flag = "TRAIN" if lab != IGNORE_INDEX else "  -  "
        print(f"    {i:>4} {tid:>9}  {lab:>9}  {flag:<6} {piece!r}")

    comparison = compare_masks(our_labels, trl_labels)
    report = describe_mask(tokenizer, trl_ids, trl_labels, prefix_consistent=is_prefix)

    print("\n  --- H1: TRL's labels == our independently computed labels ---")
    print(f"    identical              : {comparison['identical']}")
    print(f"    active in ours / theirs: {comparison['n_active_ours']} / {comparison['n_active_theirs']}")
    if not comparison["identical"]:
        print(f"    disagreements at       : {comparison['first_disagreements']}")

    print("\n  --- H2: the active region is exactly the assistant's answer ---")
    print(f"    boundary index   : {report.boundary}  (ours computed {boundary})")
    print(f"    ignored text     : {report.ignored_text!r}")
    print(f"    ACTIVE text      : {report.active_text!r}")
    print(f"    active fraction  : {report.active_fraction:.1%}")
    print(f"    contiguous suffix: {report.contiguous_active_suffix}")
    for note in report.notes:
        print(f"    NOTE: {note}")

    # The answer text must appear in the active region and NOT in the ignored one.
    answer = COMPLETION_MESSAGES[0]["content"]
    h2 = answer in report.active_text and answer not in report.ignored_text

    print(f"\n    H1 holds: {comparison['identical']}")
    print(f"    H2 holds: {h2}")

    return {
        "completion_only_loss": trainer.completion_only_loss,
        "prefix_consistent": is_prefix,
        "n_tokens": report.n_tokens,
        "n_active": report.n_active,
        "boundary": report.boundary,
        "our_boundary": boundary,
        "labels_identical_to_ours": comparison["identical"],
        "active_text": report.active_text,
        "ignored_text": report.ignored_text,
        "H1": bool(comparison["identical"]),
        "H2": bool(h2),
    }


def arm_b(tokenizer, model, arm_a_result: dict) -> dict:
    print("\n" + "=" * 78)
    print("ARM B - the SAME data with completion_only_loss=False (the bug)")
    print("=" * 78)

    dataset = Dataset.from_list(
        [{"prompt": PROMPT_MESSAGES, "completion": COMPLETION_MESSAGES}]
    )
    trainer = build_trainer(model, tokenizer, dataset, completion_only_loss=False)
    row = trainer.train_dataset[0]
    report = describe_mask(tokenizer, row["input_ids"], row["labels"])

    print(f"  active positions : {report.n_active} / {report.n_tokens}")
    print(f"  active fraction  : {report.active_fraction:.1%}")
    print(f"  ACTIVE text      : {report.active_text!r}")

    masked_active = arm_a_result["n_active"]
    h3 = report.n_active > masked_active
    print(f"\n  masked run trained {masked_active} positions, unmasked {report.n_active}")
    print(f"  H3 holds (unmasked trains strictly more): {h3}")
    print(
        "\n  INTERPRETATION: with the mask off, the model is trained to emit the "
        "\n  system prompt and the user's question. That is not instruction "
        "\n  following - it is memorising the template."
    )

    return {
        "n_active": report.n_active,
        "n_tokens": report.n_tokens,
        "active_fraction": report.active_fraction,
        "H3": bool(h3),
    }


def arm_c(tokenizer, model) -> dict:
    print("\n" + "=" * 78)
    print("ARM C - multi-turn messages with assistant_only_loss=True")
    print("=" * 78)

    dataset = Dataset.from_list([{"messages": MULTITURN}])
    try:
        trainer = build_trainer(
            model, tokenizer, dataset, assistant_only_loss=True
        )
    except Exception as exc:  # noqa: BLE001 - we want to record the failure mode
        print(f"  FAILED to build: {type(exc).__name__}: {exc}")
        print("  (assistant_only_loss needs a chat template with {% generation %})")
        return {"supported": False, "error": f"{type(exc).__name__}: {exc}"}

    row = trainer.train_dataset[0]
    report = describe_mask(tokenizer, row["input_ids"], row["labels"])

    print(f"  active positions : {report.n_active} / {report.n_tokens}")
    print(f"  ACTIVE text      : {report.active_text!r}")
    print(f"  ignored text     : {report.ignored_text!r}")
    print(f"  contiguous suffix: {report.contiguous_active_suffix} (expected False - two separate turns)")

    active = report.active_text
    both_answers = "It is 4." in active and "It is 6." in active
    no_questions = "What is 2+2?" not in active and "And 3+3?" not in active
    h4 = both_answers and no_questions

    print(f"\n  both assistant answers present in active region : {both_answers}")
    print(f"  neither user question present in active region  : {no_questions}")
    print(f"  H4 holds: {h4}")

    return {
        "supported": True,
        "n_active": report.n_active,
        "n_tokens": report.n_tokens,
        "contiguous": report.contiguous_active_suffix,
        "H4": bool(h4),
    }


def _masked_mean_ce(logits, labels) -> tuple[float, int]:
    """Cross-entropy over non-ignored positions, plus the number that counted.

    Written out rather than delegated so the shift is visible: position t's
    logits predict token t+1, so logits lose their last position and labels
    lose their first. Getting this backwards is the other classic SFT bug.

    RETURNING THE COUNT IS NOT COSMETIC. The number of positions that actually
    contribute to the mean is ``(labels[:, 1:] != IGNORE_INDEX).sum()``, which
    is NOT the same as ``(labels != IGNORE_INDEX).sum()`` whenever position 0
    is unmasked - the shift discards it, because no token precedes it to
    predict it from. Counting the wrong one is what made this script's first
    run report a decomposition mismatch of 1.8e-02; see the header note.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    n_contributing = int((shift_labels != IGNORE_INDEX).sum())
    loss = float(
        F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)).float(),
            shift_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
    )
    return loss, n_contributing


def arm_d(tokenizer) -> dict:
    print("\n" + "=" * 78)
    print("ARM D - the real pretrained model: why a broken mask looks better")
    print("=" * 78)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=dtype
    ).to(device).eval()
    print(f"  model {MODEL_ID} @ {REVISION[:12]} on {device}, dtype {dtype}")

    ids, labels_masked, boundary = expected_labels(
        tokenizer, PROMPT_MESSAGES, COMPLETION_MESSAGES
    )
    input_ids = torch.tensor([ids], device=device)

    # Three label sets over the SAME sequence - only the mask differs.
    completion_only = torch.tensor([labels_masked], device=device)
    everything = input_ids.clone()
    prompt_only = input_ids.clone()
    prompt_only[:, boundary:] = IGNORE_INDEX

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits

    loss_completion, n_completion = _masked_mean_ce(logits, completion_only)
    loss_prompt, n_prompt = _masked_mean_ce(logits, prompt_only)
    loss_all, n_all = _masked_mean_ce(logits, everything)

    # The label-active count differs from the loss-contributing count exactly
    # when position 0 is unmasked. Printing both makes the shift visible.
    labelled_prompt = int((prompt_only != IGNORE_INDEX).sum())

    print(f"\n  {'region':<26} {'labelled':>9} {'counted':>8} {'mean CE':>10}")
    print(f"  {'PROMPT only (masked out)':<26} {labelled_prompt:>9} {n_prompt:>8} {loss_prompt:>10.4f}")
    print(f"  {'COMPLETION only (trained)':<26} {n_completion:>9} {n_completion:>8} {loss_completion:>10.4f}")
    print(f"  {'everything (broken mask)':<26} {labelled_prompt + n_completion:>9} {n_all:>8} {loss_all:>10.4f}")
    print(
        f"\n  NOTE: the prompt region is labelled at {labelled_prompt} positions but only "
        f"{n_prompt} contribute\n        to the loss - the shift drops position 0, which has "
        f"no predecessor."
    )

    h5 = loss_prompt < loss_completion
    ratio = loss_completion / loss_prompt if loss_prompt > 0 else float("inf")

    print(f"\n  H5 (prompt is EASIER than completion) : {h5}")
    print(f"  completion loss / prompt loss         : {ratio:.2f}x")
    print(f"  loss if masked correctly              : {loss_completion:.4f}")
    print(f"  loss with the mask broken             : {loss_all:.4f}")

    if not h5:
        print(
            "\n  H5 IS DISPROVED, and the reason matters more than the hypothesis."
            "\n  This is the BASE model, not the Instruct model: it was never trained"
            "\n  on ChatML, so <|im_start|>system ... is UNFAMILIAR to it, and the"
            "\n  opening tokens have almost no left context to be predicted from. The"
            "\n  prompt is therefore HARDER than the answer, and a broken mask makes"
            "\n  the reported loss look WORSE here, not better."
            "\n"
            "\n  The corrected lesson is stronger than the one predicted. An unmasked"
            "\n  loss is not reliably lower OR higher - it is a mean over a DIFFERENT"
            "\n  POPULATION of tokens, so it is not comparable to a masked loss at all."
            "\n  Whether the contamination flatters or penalises depends on the model"
            "\n  and the template. E2's rule survives in a sharper form: never compare"
            "\n  losses computed over different token populations."
        )
    else:
        print(
            "\n  The pretrained model predicts the templated prompt more easily than"
            "\n  the answer, so including those positions drags the mean down and a"
            "\n  broken mask reports a better number while learning less."
        )

    # Sanity: the three losses must be a weighted decomposition of each other,
    # using the COUNTED positions.
    reconstructed = (loss_prompt * n_prompt + loss_completion * n_completion) / (
        n_prompt + n_completion
    )
    agrees = abs(reconstructed - loss_all) < 1e-3
    print(f"\n  cross-check: weighted mean of the regions = {reconstructed:.6f}")
    print(f"               loss over everything          = {loss_all:.6f}")
    print(f"               agree to 1e-3                 : {agrees}")

    return {
        "device": str(device),
        "dtype": str(dtype),
        "n_prompt_labelled": labelled_prompt,
        "n_prompt_counted": n_prompt,
        "n_completion_tokens": n_completion,
        "loss_prompt_only": loss_prompt,
        "loss_completion_only": loss_completion,
        "loss_everything": loss_all,
        "decomposition_reconstructed": reconstructed,
        "decomposition_agrees": bool(agrees),
        "H5": bool(h5),
        "H5_note": (
            "DISPROVED - base model has not seen ChatML, so the prompt is harder "
            "than the completion; contamination direction is model-dependent"
            if not h5
            else "held"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-real-model", action="store_true")
    args = parser.parse_args()

    configure_hf_cache()
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)

    print("=" * 78)
    print("E12 - SFT LOSS MASKING VERIFICATION (Phase 3 gate)")
    print("=" * 78)
    print(f"  tokenizer      : {MODEL_ID} @ {REVISION[:12]}")
    print(f"  vocab size     : {len(tokenizer)}")
    print(f"  eos_token      : {tokenizer.eos_token!r} (id {tokenizer.eos_token_id})")
    print(f"  pad_token      : {tokenizer.pad_token!r} (id {tokenizer.pad_token_id})")
    print(f"  IGNORE_INDEX   : {IGNORE_INDEX}")

    model = tiny_model(tokenizer)
    print(f"  tiny model     : {sum(p.numel() for p in model.parameters()):,} params")

    results = {}
    results["arm_a"] = arm_a(tokenizer, model)
    results["arm_b"] = arm_b(tokenizer, model, results["arm_a"])
    results["arm_c"] = arm_c(tokenizer, model)
    if args.skip_real_model:
        print("\n(--skip-real-model: arm D not run)")
        results["arm_d"] = {"skipped": True}
    else:
        results["arm_d"] = arm_d(tokenizer)

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    checks = {
        "H1 TRL labels == independent computation": results["arm_a"].get("H1"),
        "H2 active region is exactly the answer  ": results["arm_a"].get("H2"),
        "H3 unmasked trains strictly more tokens ": results["arm_b"].get("H3"),
        "H4 assistant_only trains only assistant ": results["arm_c"].get("H4"),
        "H5 prompt easier -> unmasked loss lower ": results["arm_d"].get("H5"),
    }
    for name, ok in checks.items():
        status = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
        print(f"  [{status}] {name}")

    required = [checks["H1 TRL labels == independent computation"],
                checks["H2 active region is exactly the answer  "],
                checks["H3 unmasked trains strictly more tokens "]]
    gate_open = all(bool(c) for c in required)
    print(f"\n  PHASE 3 GATE (H1, H2, H3 required): {'OPEN' if gate_open else 'CLOSED'}")
    if not gate_open:
        print("  Phase 4 (PEFT) MUST NOT start.")

    print("\n" + json.dumps(results, indent=2, sort_keys=True, default=str))
    return 0 if gate_open else 1


if __name__ == "__main__":
    raise SystemExit(main())
