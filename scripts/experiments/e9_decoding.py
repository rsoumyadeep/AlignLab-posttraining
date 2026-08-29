"""E9 - how decoding strategies actually behave.

OBJECTIVE   Compare greedy, temperature, top-k, top-p and beam search on a
            model whose distribution MEANS something, rather than on random
            logits.

HYPOTHESIS  greedy      deterministic and repetitive - the highest-probability
                        continuation of a repeated phrase is more of the same
            low temp    close to greedy
            temp = 1    the model's own distribution; more varied
            high temp   flattens toward uniform; grammar should degrade
            top-k       removes the tail; keeps variety while cutting nonsense
            top-p       adaptive; similar to top-k but sized per context
            beam        higher sequence log-probability than greedy, but blander

METRICS     distinct-4      unique 4-grams / total 4-grams (higher = more varied)
            repeat rate     fraction of positions equal to the token 8 back
            grammatical     fraction of complete sentences matching the
                            DET NOUN VERB DET NOUN . template the corpus uses
            logprob         total log-probability of the continuation

CONFIG      tiny educational LM, seed 20260829, greedy prompt "the cat ",
            96 new tokens, 8 samples per stochastic setting.

Run: python scripts/experiments/e9_decoding.py
"""
from __future__ import annotations

import re

import torch

from tiny_lm import DETERMINERS, NOUNS, VERBS, decode, encode, train_tiny_lm
from alignlab.models.generation import beam_search, generate

SEED, N_NEW, N_SAMPLES = 20260829, 96, 8
PROMPT = "the cat "

SENTENCE_RE = re.compile(
    rf"\b({'|'.join(DETERMINERS)}) ({'|'.join(NOUNS)}) ({'|'.join(VERBS)}) "
    rf"({'|'.join(DETERMINERS)}) ({'|'.join(NOUNS)}) \."
)


def distinct_n(text: str, n: int = 4) -> float:
    grams = [text[i : i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def repeat_rate(ids: list[int], lag: int = 8) -> float:
    if len(ids) <= lag:
        return 0.0
    return sum(ids[i] == ids[i - lag] for i in range(lag, len(ids))) / (len(ids) - lag)


def grammatical_fraction(text: str) -> float:
    """Fraction of period-terminated chunks that match the grammar template."""
    chunks = [c.strip() for c in text.split(".") if c.strip()]
    if not chunks:
        return 0.0
    ok = sum(bool(SENTENCE_RE.fullmatch(c + " .")) for c in chunks[:-1])
    return ok / max(len(chunks) - 1, 1)


def sequence_logprob(model, seq: torch.Tensor, prompt_len: int) -> float:
    with torch.no_grad():
        logits, _ = model(seq[:, :-1])
    lp = torch.log_softmax(logits, dim=-1)
    picked = lp.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)
    return float(picked[:, prompt_len - 1 :].sum())


def run(model, label: str, **kwargs) -> dict:
    prompt_ids = torch.tensor([encode(PROMPT)], dtype=torch.long)
    n = 1 if kwargs.get("greedy") else N_SAMPLES

    d4, rep, gram, lps, first = [], [], [], [], None
    for i in range(n):
        gen = torch.Generator().manual_seed(SEED + i)
        out = generate(model, prompt_ids, max_new_tokens=N_NEW, generator=gen, **kwargs)
        text = decode(out[0].tolist())
        new_ids = out[0, prompt_ids.shape[1] :].tolist()

        d4.append(distinct_n(text))
        rep.append(repeat_rate(new_ids))
        gram.append(grammatical_fraction(text))
        lps.append(sequence_logprob(model, out, prompt_ids.shape[1]))
        if first is None:
            first = text
    return {
        "label": label,
        "distinct4": sum(d4) / len(d4),
        "repeat": sum(rep) / len(rep),
        "gram": sum(gram) / len(gram),
        "logprob": sum(lps) / len(lps),
        "sample": first,
    }


def main() -> None:
    print("=" * 92)
    print("E9 - decoding strategy behaviour")
    print("=" * 92)
    model, res = train_tiny_lm(seed=SEED, verbose=False)
    print(f"tiny LM: {res.n_params:,} params | train loss {res.final_train_loss:.4f} "
          f"| uniform {res.uniform_loss:.4f}")
    print(f"prompt {PROMPT!r} | {N_NEW} new tokens | {N_SAMPLES} samples per "
          f"stochastic setting | seed {SEED}\n")

    rows = [
        run(model, "greedy", greedy=True),
        run(model, "temp=0.5", temperature=0.5),
        run(model, "temp=1.0", temperature=1.0),
        run(model, "temp=2.0", temperature=2.0),
        run(model, "top-k=5", temperature=1.0, top_k=5),
        run(model, "top-p=0.9", temperature=1.0, top_p=0.9),
    ]

    print(f"{'strategy':<12}{'distinct-4':>11}{'repeat':>9}{'grammatical':>13}{'logprob':>10}")
    print("-" * 92)
    for r in rows:
        print(f"{r['label']:<12}{r['distinct4']:>11.3f}{r['repeat']:>9.3f}"
              f"{r['gram']:>13.3f}{r['logprob']:>10.2f}")

    # beam search, scored the same way
    prompt_ids = torch.tensor([encode(PROMPT)], dtype=torch.long)
    beam = beam_search(model, prompt_ids, max_new_tokens=N_NEW, beam_width=4)
    beam_text = decode(beam[0].tolist())
    print(f"{'beam=4':<12}{distinct_n(beam_text):>11.3f}"
          f"{repeat_rate(beam[0, prompt_ids.shape[1]:].tolist()):>9.3f}"
          f"{grammatical_fraction(beam_text):>13.3f}"
          f"{sequence_logprob(model, beam, prompt_ids.shape[1]):>10.2f}")

    print("\n--- one sample from each ---")
    for r in rows:
        print(f"  {r['label']:<11} {r['sample']!r}")
    print(f"  {'beam=4':<11} {beam_text!r}")
    print("\n" + "=" * 92)


if __name__ == "__main__":
    main()
