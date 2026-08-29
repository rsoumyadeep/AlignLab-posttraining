"""Autoregressive generation and decoding strategies.

AUTOREGRESSIVE GENERATION, token by token:

    prompt = [t1, t2, t3]
    step 1: forward(prompt)      -> logits [B, 3, V]; take logits[:, -1] -> t4
    step 2: forward([...t4])     -> take the last position           -> t5
    step 3: ...                                                       -> t6

TWO POINTS THAT ARE EASY TO MISS.

1. Only the LAST position's logits matter. The forward pass produces a
   prediction at every position, but positions 0..T-2 are predicting tokens we
   already have. During training all of them contribute to the loss (which is
   what makes training parallel over positions); during generation all but the
   last are discarded.

2. Training and inference differ fundamentally. Training is ONE parallel pass
   over the whole sequence with teacher forcing - the model always conditions
   on the TRUE prefix. Generation is T sequential passes conditioning on the
   model's OWN outputs. This gap (exposure bias) is why a model with excellent
   teacher-forced loss can still generate poorly.

DECODING STRATEGIES turn logits into a token. They are pure functions of the
logits, which makes them unit-testable against hand-computed cases without a
model at all.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from alignlab.models.kv_cache import KVCache

NEG_INF = float("-inf")


# ==========================================================================
# logit -> token strategies
# ==========================================================================


def greedy_select(logits: torch.Tensor) -> torch.Tensor:
    """Always take the highest-probability token. argmax over the vocab.

    Deterministic. Tends to repeat, because the highest-probability
    continuation of a repeated phrase is usually more of the same - there is
    no mechanism to escape a loop.
    """
    return torch.argmax(logits, dim=-1, keepdim=True)


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by T before softmax.

        T -> 0    approaches argmax (greedy)
        T = 1     the model's own distribution, unmodified
        T -> inf  approaches uniform over the vocabulary

    Dividing by T < 1 spreads the logits apart, so softmax sharpens; dividing
    by T > 1 compresses them, so softmax flattens. It rescales CONFIDENCE
    without changing the RANKING - the argmax is the same for every T > 0.
    """
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if temperature == 0:
        # The limit T -> 0 is argmax. Handled explicitly to avoid dividing by 0.
        out = torch.full_like(logits, NEG_INF)
        return out.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 0.0)
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k highest-scoring tokens, mask the rest to -inf.

    Removes the long tail of implausible tokens, which is where most
    degenerate samples come from: individually each tail token is unlikely,
    but collectively they can hold substantial probability mass.

    LIMITATION, and the reason top-p exists: k is FIXED. When the model is
    confident (one obvious continuation) k=50 admits 49 bad options; when it is
    uncertain (many valid continuations) k=50 may cut off good ones.
    """
    if k <= 0:
        raise ValueError(f"top_k must be >= 1, got {k}")
    k = min(k, logits.shape[-1])

    kth_value = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth_value, NEG_INF)


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set whose probability mass >= p.

    The ADAPTIVE answer to top-k's fixed cutoff. Sort by probability, take the
    running cumulative sum, and keep tokens up to and including the one that
    crosses p. Confident distributions keep few tokens; flat ones keep many.

    The token that CROSSES the threshold is kept (not dropped), so the retained
    mass is always >= p and the set is never empty - important, because an
    empty set would make sampling undefined.
    """
    if not 0.0 < p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {p}")

    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

    # Remove tokens once the cumulative mass has already reached p. The shift
    # keeps the crossing token itself.
    remove = cumulative > p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(remove, NEG_INF)

    # Unsort. torch.sort gives sorted_logits[i] == logits[sorted_idx[i]], so
    # the inverse is out[sorted_idx[i]] = sorted_logits[i] - a scatter with
    # sorted_idx ITSELF as the index.
    #
    # BUG FIXED HERE (found by experiment E9). This previously used
    # sorted_idx.argsort(dim=-1) as the scatter index, which is the inverse
    # permutation - correct for a gather, wrong for a scatter. The surviving
    # logits were written to the WRONG vocabulary positions, so the nucleus
    # ended up containing arbitrary tokens rather than the most probable ones.
    #
    # It survived the unit tests because those used already-descending logits,
    # where sorted_idx is the identity and argsort(identity) is also the
    # identity - so the bug was invisible. See
    # tests/test_generation.py::test_top_p_unsorts_correctly.
    return torch.full_like(logits, NEG_INF).scatter(-1, sorted_idx, sorted_logits)


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Full sampling pipeline: temperature -> top-k -> top-p -> multinomial.

    ORDER MATTERS. Temperature is applied first because it changes the
    probabilities that top-k and top-p then threshold. Applying temperature
    afterwards would filter on one distribution and sample from another.

    Args:
        logits: [B, vocab_size]
    Returns:
        [B, 1] sampled token ids.
    """
    logits = apply_temperature(logits, temperature)

    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


# ==========================================================================
# generation loops
# ==========================================================================


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
    use_cache: bool = True,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate tokens autoregressively.

    Args:
        model: a DecoderOnlyTransformer.
        input_ids: [B, T_prompt]
        use_cache: when True, only the newest token is fed each step and K/V
            are reused. Must produce the same logits as use_cache=False - that
            is the correctness claim, verified in tests/test_kv_cache.py.

    Returns:
        [B, T_prompt + n_generated]
    """
    model.eval()
    generated = input_ids
    cache = KVCache(model.cfg.n_layers) if use_cache else None

    for step in range(max_new_tokens):
        if cache is not None:
            # First step processes the whole prompt (filling the cache);
            # afterwards only the single newest token.
            step_input = generated if step == 0 else generated[:, -1:]
        else:
            step_input = generated

        logits, _ = model(step_input, cache=cache)
        next_logits = logits[:, -1, :]  # only the LAST position matters

        if greedy:
            next_token = greedy_select(next_logits)
        else:
            next_token = sample_from_logits(
                next_logits, temperature, top_k, top_p, generator=generator
            )

        generated = torch.cat([generated, next_token], dim=1)

        if eos_token_id is not None and bool((next_token == eos_token_id).all()):
            break

    return generated


@torch.no_grad()
def beam_search(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    beam_width: int = 4,
    length_penalty: float = 1.0,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Beam search, batch size 1, written for clarity.

    Greedy decoding commits to the best token at each step and can never
    recover from an early mistake. Beam search keeps the ``beam_width`` best
    partial sequences and expands all of them, so a locally worse token that
    leads somewhere better can survive.

    Scores are summed LOG-probabilities. Summing logs (rather than multiplying
    probabilities) keeps the arithmetic in a numerically sane range - a product
    of hundreds of probabilities underflows float32 almost immediately.

    LENGTH PENALTY. Every extra token adds a negative log-probability, so
    longer sequences always score worse and raw beam search is biased toward
    short output. Dividing by ``length ** length_penalty`` counteracts it;
    1.0 is plain mean log-probability.

    NOTE: beam search is standard for translation and summarisation, where one
    high-probability answer is wanted. It is a poor fit for open-ended text
    generation - it produces bland, repetitive output, because the
    highest-probability continuation is usually the least interesting one.
    """
    if input_ids.shape[0] != 1:
        raise ValueError("this educational beam search handles batch size 1 only")
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")

    model.eval()
    device = input_ids.device

    beams = input_ids  # [beam, T], starts as [1, T]
    scores = torch.zeros(1, device=device)  # summed log-probs
    finished: list[tuple[torch.Tensor, float]] = []

    for _ in range(max_new_tokens):
        logits, _ = model(beams)
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # [beam, V]

        # every (beam, token) continuation, scored
        candidate = scores.unsqueeze(1) + log_probs  # [beam, V]
        flat = candidate.view(-1)  # [beam*V]

        top_scores, top_idx = torch.topk(flat, min(beam_width, flat.numel()))
        beam_idx = top_idx // log_probs.shape[-1]
        token_idx = top_idx % log_probs.shape[-1]

        beams = torch.cat([beams[beam_idx], token_idx.unsqueeze(1)], dim=1)
        scores = top_scores

        if eos_token_id is not None:
            done = token_idx == eos_token_id
            for i in torch.nonzero(done).flatten().tolist():
                length = beams.shape[1] - input_ids.shape[1]
                finished.append(
                    (beams[i].clone(), float(scores[i]) / (length**length_penalty))
                )
            if bool(done.all()):
                break

    if finished:
        best = max(finished, key=lambda pair: pair[1])[0]
        return best.unsqueeze(0)

    length = max(beams.shape[1] - input_ids.shape[1], 1)
    normalised = scores / (length**length_penalty)
    return beams[int(normalised.argmax())].unsqueeze(0)
