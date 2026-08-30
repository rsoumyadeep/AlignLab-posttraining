"""E11 - reconcile the REAL Qwen2.5-1.5B weights against Phase 2's arithmetic.

Phase 2 (WP9) read Qwen's config.json and computed a parameter count from
component formulas, WITHOUT ever downloading weights. The Phase 2 report
claimed:

    "our component formulas give 1,543,656,960 / 1,310,281,728 - matching the
     card's 1.54B / 1.31B independently"

Phase 3 downloads the weights, so that claim is now falsifiable. This script
falsifies it.

HYPOTHESIS (recorded before running, and WRONG):
    sum(p.numel() for p in model.parameters()) == 1,543,656,960

ACTUAL: 1,543,714,304. The formula was short by 57,344 parameters.

WHY THIS IS THE INTERESTING KIND OF WRONG. Phase 2 *discovered* that Qwen
uses attention QKV bias - it is listed in the Phase 2 report as one of three
predictions we got wrong, and the model card says so explicitly. But the
discovery never propagated into the parameter arithmetic. The bias terms were
noted in prose and omitted from the sum. 57,344 is exactly

    28 layers x (1536 q_bias + 256 k_bias + 256 v_bias) = 28 x 2048

so the error is precisely the thing Phase 2 had already learned about. Knowing
a fact and having your arithmetic reflect it are different states, and only
contact with the real checkpoint distinguished them.

The error is 0.0037% - invisible at the "1.54B" precision the model card
quotes, which is exactly why it survived Phase 2. A claim agreeing to three
significant figures is not the same as a claim being exact, and the Phase 2
report used the word "exactly".

This script recomputes with the bias terms included and shows the corrected
formula reproduces the checkpoint to the parameter.

Run (server, weights required):
    python scripts/experiments/e11_weights_reconciliation.py
"""

from __future__ import annotations

import json

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from alignlab.paths import configure_hf_cache

MODEL_ID = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"

# The number Phase 2 predicted from config.json alone, quoted from
# docs/phase2/PHASE_2_REPORT.md section 8. Preserved here as the original
# prediction - PROJECT_INSTRUCTIONS section 17 forbids rewriting history to
# make an earlier phase look better.
PHASE2_PREDICTED_TOTAL = 1_543_656_960
PHASE2_PREDICTED_NON_EMBEDDING = 1_310_281_728


def rope_theta_from(config) -> float:
    """Read rope_theta without assuming where transformers keeps it.

    IMPORTANT COMPATIBILITY FINDING. Under transformers 4.x this was
    ``config.rope_theta``. Under transformers 5.x - which is what AlignLab
    actually has installed - that attribute DOES NOT EXIST; the value lives in
    ``config.rope_parameters["rope_theta"]`` (with ``config.rope_scaling`` as
    an alias). Code written against the 4.x layout raises AttributeError here.

    PROJECT_INSTRUCTIONS requires Phase 3 to "explicitly read rope_theta from
    the config, never from our defaults". Doing that correctly means not
    assuming the attribute path either.
    """
    params = getattr(config, "rope_parameters", None)
    if isinstance(params, dict) and "rope_theta" in params:
        return float(params["rope_theta"])
    if hasattr(config, "rope_theta"):
        return float(config.rope_theta)
    raise AttributeError("rope_theta not found under any known config layout")


def predicted_parameters(config, include_qkv_bias: bool) -> dict[str, int]:
    """Compute the parameter count from component formulas.

    ``include_qkv_bias=False`` reproduces Phase 2's arithmetic exactly, so the
    two arms differ in one term and the discrepancy is attributable.
    """
    d = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv = config.num_key_value_heads
    head_dim = d // n_heads
    kv_dim = n_kv * head_dim
    ffn = config.intermediate_size
    layers = config.num_hidden_layers
    vocab = config.vocab_size

    q = d * d + (d if include_qkv_bias else 0)
    k = d * kv_dim + (kv_dim if include_qkv_bias else 0)
    v = d * kv_dim + (kv_dim if include_qkv_bias else 0)
    o = d * d  # Qwen gives o_proj NO bias - verified against the checkpoint
    attention = q + k + v + o

    # SwiGLU: three matrices, not two. gate and up both project d -> ffn,
    # down projects back ffn -> d.
    mlp = 3 * d * ffn

    # Two RMSNorm gains per block, one scalar per channel, no bias.
    norms = 2 * d

    per_layer = attention + mlp + norms
    embedding = vocab * d
    final_norm = d
    # tie_word_embeddings=True means lm_head SHARES the embedding matrix and
    # contributes nothing further.
    lm_head = 0 if config.tie_word_embeddings else vocab * d

    non_embedding = per_layer * layers + final_norm
    return {
        "embedding": embedding,
        "per_layer": per_layer,
        "attention_per_layer": attention,
        "mlp_per_layer": mlp,
        "norms_per_layer": norms,
        "final_norm": final_norm,
        "lm_head": lm_head,
        "non_embedding": non_embedding,
        "total": embedding + non_embedding + lm_head,
    }


def main() -> int:
    configure_hf_cache()

    config = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=torch.bfloat16
    )

    actual_total = sum(p.numel() for p in model.parameters())
    embed = model.get_input_embeddings().weight
    actual_embedding = embed.numel()
    actual_non_embedding = actual_total - actual_embedding

    print("=" * 78)
    print("E11 - real weights vs Phase 2's config-only arithmetic")
    print("=" * 78)
    print(f"model    : {MODEL_ID}")
    print(f"revision : {REVISION}  (PINNED)")
    print(f"dtype    : {next(model.parameters()).dtype}")

    print("\n--- HYPOTHESIS (from the Phase 2 report, recorded before running) ---")
    print(f"  total parameters == {PHASE2_PREDICTED_TOTAL:,}")

    print("\n--- ACTUAL ---")
    print(f"  total          {actual_total:,}")
    print(f"  embedding      {actual_embedding:,}")
    print(f"  non-embedding  {actual_non_embedding:,}")

    delta = actual_total - PHASE2_PREDICTED_TOTAL
    print("\n--- VERDICT ---")
    print(f"  hypothesis holds : {actual_total == PHASE2_PREDICTED_TOTAL}")
    print(f"  discrepancy      : {delta:+,} parameters")
    print(f"  relative error   : {abs(delta) / actual_total * 100:.4f}%")

    # ------------------------------------------------ attribute the difference
    without = predicted_parameters(config, include_qkv_bias=False)
    with_bias = predicted_parameters(config, include_qkv_bias=True)

    d = config.hidden_size
    kv_dim = config.num_key_value_heads * (d // config.num_attention_heads)
    bias_per_layer = d + 2 * kv_dim
    bias_total = bias_per_layer * config.num_hidden_layers

    print("\n--- ATTRIBUTION ---")
    print(f"  QKV bias per layer : {d} + {kv_dim} + {kv_dim} = {bias_per_layer:,}")
    print(f"  x {config.num_hidden_layers} layers        : {bias_total:,}")
    print(f"  explains the gap   : {bias_total == delta}")

    print("\n--- FORMULAS ---")
    print(f"  Phase 2 formula (no QKV bias) : {without['total']:,}")
    print(f"  reproduces Phase 2's number   : {without['total'] == PHASE2_PREDICTED_TOTAL}")
    print(f"  corrected formula (with bias) : {with_bias['total']:,}")
    print(f"  matches the checkpoint EXACTLY: {with_bias['total'] == actual_total}")
    print(f"  non-embedding, corrected      : {with_bias['non_embedding']:,}")
    print(f"  matches actual non-embedding  : {with_bias['non_embedding'] == actual_non_embedding}")

    # ------------------------------------------------- structural verification
    print("\n--- STRUCTURE (verified against the loaded module tree) ---")
    attn = model.model.layers[0].self_attn
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        module = getattr(attn, name)
        bias = tuple(module.bias.shape) if module.bias is not None else None
        print(f"  {name:<7} weight {str(tuple(module.weight.shape)):<16} bias {bias}")

    tied = model.get_input_embeddings().weight.data_ptr() == model.lm_head.weight.data_ptr()
    print(f"  embeddings tied to lm_head : {tied}")
    print(f"  embedding == vocab * d     : {actual_embedding == config.vocab_size * d}")

    print("\n--- CONFIG VALUES PHASE 3 MUST READ, NOT ASSUME ---")
    print(f"  hidden_size             {config.hidden_size}")
    print(f"  num_attention_heads     {config.num_attention_heads}")
    print(f"  num_key_value_heads     {config.num_key_value_heads}")
    print(f"  head_dim (derived)      {d // config.num_attention_heads}")
    print(f"  num_hidden_layers       {config.num_hidden_layers}")
    print(f"  intermediate_size       {config.intermediate_size}")
    print(f"  d_ff / d_model          {config.intermediate_size / d:.4f}")
    print(f"  rope_theta              {rope_theta_from(config):,.0f}")
    print(f"  max_position_embeddings {config.max_position_embeddings}")
    print(f"  vocab_size              {config.vocab_size}")
    print(f"  tie_word_embeddings     {config.tie_word_embeddings}")
    print(f"  use_sliding_window      {config.use_sliding_window}")

    print("\n--- WHAT THIS PROVES / DOES NOT PROVE ---")
    print("  PROVES     : the corrected component formula reproduces the real")
    print("               checkpoint exactly, and the Phase 2 shortfall is")
    print("               entirely the omitted QKV bias terms.")
    print("  DOES NOT   : say anything about the model's behaviour or quality.")
    print("               Counting parameters is not evaluating a model.")

    summary = {
        "phase2_predicted_total": PHASE2_PREDICTED_TOTAL,
        "actual_total": actual_total,
        "discrepancy": delta,
        "discrepancy_explained_by_qkv_bias": bool(bias_total == delta),
        "corrected_formula_total": with_bias["total"],
        "corrected_formula_exact": bool(with_bias["total"] == actual_total),
        "actual_embedding": actual_embedding,
        "actual_non_embedding": actual_non_embedding,
        "rope_theta": rope_theta_from(config),
    }
    print("\n" + json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
