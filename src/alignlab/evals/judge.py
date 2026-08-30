"""LLM-as-judge — pairwise comparison with position randomisation.

PROJECT_INSTRUCTIONS: *"Do not treat the judge as ground truth. Do not report a
tiny score difference as meaningful without discussing uncertainty/variance."*

Both are load-bearing here, so the design enforces them rather than hoping.

--------------------------------------------------------------- POSITION BIAS

An LLM judge shown two answers has a measurable tendency to prefer whichever it
sees first (or second). If you only ask once per pair, that bias is baked into
your win rate and you cannot see it.

So every pair is judged **twice**: once as (A, B) and once as (B, A). Then:

    consistent   the judge picked the SAME response both times -> a real vote
    inconsistent it picked whichever came first (or second) both times ->
                 the verdict is position, not preference

`position_bias_rate` is reported as a first-class number. A judge with a high
inconsistency rate is not measuring quality, and the report says so instead of
averaging the noise away.

TIES ARE ALLOWED AND COUNTED. Forcing a binary choice on genuinely comparable
answers manufactures signal.

------------------------------------------------------------ NOT GROUND TRUTH

The judge here is Qwen2.5-7B-Instruct: roughly 4.5x the size of the models it
judges, but small by frontier standards, AND FROM THE SAME MODEL FAMILY as the
models under test. Same-family judging is a known bias risk - a judge may
favour text that resembles its own training distribution. That limitation is
recorded in every result this module produces, not just in prose.

The output is a win rate with a Wilson interval. If the interval includes 0.5,
the honest statement is "not resolvable at this sample size".
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass, field
from typing import Sequence

import torch

from alignlab.evals.metrics import wilson_interval
from alignlab.logging_utils import get_logger

logger = get_logger(__name__)

# The judge prompt. Recorded verbatim in every result, because a win rate
# without its prompt is not reproducible.
JUDGE_SYSTEM = (
    "You are an impartial evaluator. You compare two candidate answers to the "
    "same question and decide which is better. Judge only on the quality of "
    "the answer: correctness, helpfulness, relevance, and clarity. Ignore "
    "which answer is longer - length is not quality. Ignore the order in "
    "which the answers are presented."
)

JUDGE_TEMPLATE = """Question:
{prompt}

Answer A:
{answer_a}

Answer B:
{answer_b}

Which answer is better? Reply with exactly one of:
A
B
TIE

Reply with only that token and nothing else."""

_VERDICT = re.compile(r"\b(A|B|TIE)\b")


@dataclass
class JudgeVerdict:
    """One pair, judged in both orders."""

    prompt: str
    answer_a: str
    answer_b: str
    raw_first: str
    raw_second: str
    # Verdicts normalised to the ORIGINAL a/b labelling, not presentation order
    verdict_first: str
    verdict_second: str
    consistent: bool
    resolved: str  # "a" | "b" | "tie" | "inconsistent"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class JudgeResult:
    """Aggregated judgement over a prompt set."""

    model_a: str
    model_b: str
    judge_model: str
    n_pairs: int
    a_wins: int
    b_wins: int
    ties: int
    inconsistent: int
    verdicts: list[dict] = field(default_factory=list)
    judge_prompt: str = JUDGE_TEMPLATE
    judge_system: str = JUDGE_SYSTEM
    settings: dict = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    @property
    def decided(self) -> int:
        """Pairs with a consistent, non-tie verdict."""
        return self.a_wins + self.b_wins

    @property
    def position_bias_rate(self) -> float:
        return self.inconsistent / self.n_pairs if self.n_pairs else 0.0

    def win_rate_b(self) -> dict | None:
        """B's win rate among DECIDED pairs, with a Wilson interval.

        Computed over decided pairs only: ties and position-flips carry no
        preference information, and folding them in as half-wins would
        manufacture precision.
        """
        if self.decided == 0:
            return None
        interval = wilson_interval(self.b_wins, self.decided)
        return {
            **interval.to_dict(),
            "excludes_chance": interval.excludes(0.5),
            "interpretation": (
                "resolvable: the interval excludes 0.5"
                if interval.excludes(0.5)
                else f"NOT resolvable at n={self.decided}: the interval includes 0.5"
            ),
        }

    def to_dict(self) -> dict:
        data = asdict(self)
        data["decided"] = self.decided
        data["position_bias_rate"] = round(self.position_bias_rate, 6)
        data["win_rate_b"] = self.win_rate_b()
        return data


def parse_verdict(text: str) -> str:
    """Extract A / B / TIE from the judge's reply.

    Returns "unparsed" rather than guessing. An unparsed reply is counted and
    reported; silently mapping it to a tie would invent data.
    """
    match = _VERDICT.search(text.strip().upper())
    if not match:
        return "unparsed"
    return match.group(1).lower()


@torch.no_grad()
def _ask(judge_model, judge_tokenizer, prompt: str, device, max_new_tokens: int = 8) -> str:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    text = judge_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = judge_tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    n_in = inputs["input_ids"].shape[1]
    output = judge_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy: the judge must be deterministic
        pad_token_id=judge_tokenizer.pad_token_id or judge_tokenizer.eos_token_id,
    )
    return judge_tokenizer.decode(output[0][n_in:], skip_special_tokens=True).strip()


def judge_pairs(
    judge_model,
    judge_tokenizer,
    prompts: Sequence[str],
    answers_a: Sequence[str],
    answers_b: Sequence[str],
    device,
    model_a: str,
    model_b: str,
    judge_model_id: str,
    seed: int = 42,
) -> JudgeResult:
    """Judge every pair TWICE, in both presentation orders.

    The randomisation is seeded so the run is reproducible, and each pair is
    presented once in each order regardless - the seed controls only which
    order comes first, which removes any systematic coupling between a pair's
    content and its first presentation.
    """
    if not (len(prompts) == len(answers_a) == len(answers_b)):
        raise ValueError(
            f"prompts ({len(prompts)}), answers_a ({len(answers_a)}) and "
            f"answers_b ({len(answers_b)}) must be the same length"
        )

    rng = random.Random(seed)
    verdicts: list[JudgeVerdict] = []
    a_wins = b_wins = ties = inconsistent = unparsed = 0

    for prompt, ans_a, ans_b in zip(prompts, answers_a, answers_b):
        a_first = rng.random() < 0.5

        # Presentation 1
        if a_first:
            raw1 = _ask(judge_model, judge_tokenizer,
                        JUDGE_TEMPLATE.format(prompt=prompt, answer_a=ans_a, answer_b=ans_b),
                        device)
            v1 = {"a": "a", "b": "b"}.get(parse_verdict(raw1), parse_verdict(raw1))
        else:
            raw1 = _ask(judge_model, judge_tokenizer,
                        JUDGE_TEMPLATE.format(prompt=prompt, answer_a=ans_b, answer_b=ans_a),
                        device)
            # positions were swapped, so map back to the original labels
            v1 = {"a": "b", "b": "a"}.get(parse_verdict(raw1), parse_verdict(raw1))

        # Presentation 2 - the opposite order
        if a_first:
            raw2 = _ask(judge_model, judge_tokenizer,
                        JUDGE_TEMPLATE.format(prompt=prompt, answer_a=ans_b, answer_b=ans_a),
                        device)
            v2 = {"a": "b", "b": "a"}.get(parse_verdict(raw2), parse_verdict(raw2))
        else:
            raw2 = _ask(judge_model, judge_tokenizer,
                        JUDGE_TEMPLATE.format(prompt=prompt, answer_a=ans_a, answer_b=ans_b),
                        device)
            v2 = {"a": "a", "b": "b"}.get(parse_verdict(raw2), parse_verdict(raw2))

        if "unparsed" in (v1, v2):
            unparsed += 1
            resolved = "inconsistent"
            consistent = False
        elif v1 == v2:
            consistent = True
            resolved = v1
        else:
            consistent = False
            resolved = "inconsistent"

        if resolved == "a":
            a_wins += 1
        elif resolved == "b":
            b_wins += 1
        elif resolved == "tie":
            ties += 1
        else:
            inconsistent += 1

        verdicts.append(
            JudgeVerdict(
                prompt=prompt, answer_a=ans_a, answer_b=ans_b,
                raw_first=raw1, raw_second=raw2,
                verdict_first=v1, verdict_second=v2,
                consistent=consistent, resolved=resolved,
            )
        )

    result = JudgeResult(
        model_a=model_a,
        model_b=model_b,
        judge_model=judge_model_id,
        n_pairs=len(prompts),
        a_wins=a_wins,
        b_wins=b_wins,
        ties=ties,
        inconsistent=inconsistent,
        verdicts=[v.to_dict() for v in verdicts],
        settings={
            "decoding": "greedy",
            "max_new_tokens": 8,
            "seed": seed,
            "both_orders": True,
            "unparsed_replies": unparsed,
        },
        limitations=[
            "The judge is NOT ground truth.",
            f"Judge ({judge_model_id}) is from the SAME model family as the "
            f"models under test - same-family judging may favour text "
            f"resembling the judge's own training distribution.",
            "The judge is small by frontier standards; its agreement with "
            "human preference has NOT been validated in this project.",
            "Position-inconsistent pairs are excluded from the win rate, not "
            "split as half-wins - they carry no preference information.",
            "No human agreement study was run.",
        ],
    )
    logger.info(
        "judge %s vs %s: a=%d b=%d tie=%d inconsistent=%d (position bias %.1f%%)",
        model_a, model_b, a_wins, b_wins, ties, inconsistent,
        100 * result.position_bias_rate,
    )
    return result
