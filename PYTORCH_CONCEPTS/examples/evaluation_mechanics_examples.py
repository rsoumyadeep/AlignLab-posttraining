"""Executable examples for PYTORCH_CONCEPTS/pytorch-evaluation-mechanics.md.

Every number quoted in that note comes from running this file:

    PYTHONPATH=src python PYTORCH_CONCEPTS/examples/evaluation_mechanics_examples.py

CPU, seeded, tiny models. The constructs demonstrated here are the ones that
silently corrupt an evaluation rather than crashing it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from alignlab.evals.metrics import (
    perplexity_from_totals,
    termination_stats,
    wilson_interval,
)


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def example_eval_mode() -> None:
    section("1. .eval() vs .train() - dropout makes evaluation non-deterministic")

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.Dropout(p=0.5), nn.Linear(8, 4))
    x = torch.randn(1, 8)

    model.train()
    torch.manual_seed(1)
    a = model(x)
    torch.manual_seed(2)
    b = model(x)
    print(f"  train() mode, two passes on the SAME input:")
    print(f"    max|a - b| = {float((a - b).abs().max()):.6f}   identical: {torch.equal(a, b)}")

    model.eval()
    c = model(x)
    d = model(x)
    print(f"  eval() mode, two passes on the SAME input:")
    print(f"    max|c - d| = {float((c - d).abs().max()):.6f}   identical: {torch.equal(c, d)}")

    print("\n  In train() mode dropout resamples every forward, so the SAME input")
    print("  gives different outputs. An evaluation run in train() mode reports")
    print("  noise. It does not crash, and the numbers look plausible.")
    print("\n  This is why alignlab.dpo.verify_reference_is_frozen() rejects a")
    print("  reference model left in train() mode: its log-probabilities would")
    print("  be stochastic and every implicit reward noisy.")


def example_no_grad() -> None:
    section("2. torch.no_grad - what it does and does not do")

    model = nn.Linear(100, 100)
    x = torch.randn(32, 100)

    y = model(x)
    print(f"  without no_grad: output.requires_grad = {y.requires_grad}, "
          f"grad_fn = {type(y.grad_fn).__name__}")

    with torch.no_grad():
        y2 = model(x)
    print(f"  with    no_grad: output.requires_grad = {y2.requires_grad}, "
          f"grad_fn = {y2.grad_fn}")

    print(f"\n  outputs identical: {torch.allclose(y, y2)}")
    print("\n  no_grad does NOT change the numbers - it stops the graph being")
    print("  built, which frees the activation memory that backward would need.")
    print("  It also does NOT put the model in eval() mode: dropout and")
    print("  batch-norm still behave as they were set. Those are two separate")
    print("  switches and forgetting the second is the quiet bug.")


def example_generation_stopping() -> None:
    section("3. eos_token_id as a LIST - two valid terminators")

    print("  Qwen2.5-1.5B base:")
    print("    tokenizer.eos_token = '<|endoftext|>'  (id 151643)")
    print("    ChatML terminator   = '<|im_end|>'     (id 151645)")
    print()
    print("  The base model's configured EOS is NOT the template's turn")
    print("  terminator. Generating with only tokenizer.eos_token_id would")
    print("  never stop on <|im_end|>, and a model that HAD learned to emit it")
    print("  would be scored as failing to terminate.")
    print()
    print("  So AlignLab passes BOTH:")
    print("      eos_token_id=[im_end_id, tokenizer.eos_token_id]")
    print()
    print("  Phase 7 measured what this distinguishes, on 6 fixed prompts:")
    print("      SFT 6/6   DPO 6/6   base 0/6   LoRA 0/6   QLoRA 0/6")
    print("  With a single-id stop criterion that difference is invisible.")


def example_termination_vs_stop_token() -> None:
    section("4. 'terminated' and 'emitted a stop token' are different")

    cap = 100
    cases = [
        ("stopped early with the token", [{"n_generated": 40, "emitted_stop_token": True, "text": "ok"}]),
        ("ran out of budget, no token", [{"n_generated": 100, "emitted_stop_token": False, "text": "ok"}]),
        ("stopped early, no token", [{"n_generated": 40, "emitted_stop_token": False, "text": "ok"}]),
    ]
    print(f"  {'case':<32} {'terminated':>11} {'stop token':>11} {'hit cap':>8}")
    for name, gens in cases:
        stats = termination_stats(gens, cap)
        print(f"  {name:<32} {stats.terminated:>11} "
              f"{stats.emitted_stop_token:>11} {stats.hit_length_cap:>8}")

    print("\n  A model can stop without emitting the stop token - it hit a")
    print("  DIFFERENT eos id. Phase 7 measured exactly that on the base model:")
    print()
    print("      base:   terminated 2/6   emitted <|im_end|> 0/6   hit cap 4/6")
    print("      SFT:    terminated 6/6   emitted <|im_end|> 6/6   hit cap 0/6")
    print("      LoRA:   terminated 0/6   emitted <|im_end|> 0/6   hit cap 6/6")
    print()
    print("  The base model stopped twice - on <|endoftext|>, never on the")
    print("  ChatML terminator it had never been trained to produce. A single")
    print("  'did it finish' number would score base at 33% and hide the fact")
    print("  that it emitted the turn terminator zero times.")


def example_perplexity_regions() -> None:
    section("5. the same forward pass, two perplexities")

    # Phase 3's measured numbers on the base model.
    completion = perplexity_from_totals(total_nll=3.6295 * 7, n_tokens=7, region="completion")
    prompt = perplexity_from_totals(total_nll=6.3314 * 25, n_tokens=25, region="prompt")

    print(f"  {'region':<14} {'tokens':>7} {'mean NLL':>10} {'perplexity':>12}")
    for r in (completion, prompt):
        print(f"  {r.region:<14} {r.n_tokens:>7} {r.mean_nll:>10.4f} {r.perplexity:>12.3f}")

    print(f"\n  comparable_to each other: {completion.comparable_to(prompt)}")
    print(f"  ratio of perplexities   : {prompt.perplexity / completion.perplexity:.2f}x")
    print("\n  Same model, same forward pass, one number 'better' than the other")
    print("  by 14x - and the comparison is meaningless, because they are means")
    print("  over different token populations. PerplexityResult carries its")
    print("  region so this cannot be reported as an improvement.")


def example_small_sample_intervals() -> None:
    section("6. what a small sample can and cannot resolve")

    print(f"  {'case':<34} {'rate':>7} {'95% interval':>20} {'excludes 0.5':>13}")
    for name, successes, n in (
        ("Phase 6 preference 86/184", 86, 184),
        ("Phase 7 stop, PEFT 0/6", 0, 6),
        ("Phase 7 stop, SFT 6/6", 6, 6),
        ("Phase 7 judge, SFT wins 3/4", 3, 4),
        ("Phase 7 judge, DPO wins 1/2", 1, 2),
        ("a 60/100 result", 60, 100),
    ):
        interval = wilson_interval(successes, n)
        print(f"  {name:<34} {interval.point:>7.3f} "
              f"[{interval.low:.3f}, {interval.high:.3f}]{'':>5} "
              f"{str(interval.excludes(0.5)):>13}")

    print("\n  The two judge rows are the real Phase 7 result. SFT beat base on")
    print("  3 of 4 DECIDED pairs and the interval STILL includes 0.5 - so the")
    print("  judge did not establish that SFT is better, even though every other")
    print("  metric in the run says it is. Reporting '75% win rate' would be")
    print("  dishonest precision at n=4.")
    print("\n  The stop-token rows behave differently: 0/6 and 6/6 each exclude")
    print("  0.5 on their own, and their intervals do not overlap, so THAT")
    print("  comparison is resolvable at n=6. A tiny sample is not automatically")
    print("  uninformative - it depends on how extreme the split is.")


def main() -> None:
    print("evaluation mechanics - executed examples")
    print(f"torch {torch.__version__}, cpu")
    example_eval_mode()
    example_no_grad()
    example_generation_stopping()
    example_termination_vs_stop_token()
    example_perplexity_regions()
    example_small_sample_intervals()
    print()


if __name__ == "__main__":
    main()
