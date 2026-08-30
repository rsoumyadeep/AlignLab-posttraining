"""LoRA from first principles.

PROJECT_INSTRUCTIONS section 4: "The core LoRA mechanism should be implemented
manually rather than simply relying on a PEFT library implementation." This
module is that implementation. `peft` is used later for the realistic training
run, and the two are compared - but this file must stand on its own and be
understandable without reading `peft`.

THE IDEA IN ONE LINE. A full fine-tune learns an update to every weight:

    W' = W + dW           dW has the same shape as W: d_out x d_in

LoRA asserts that dW, for a *specific downstream task*, has low intrinsic rank,
and factorises it:

    dW = B A              B: d_out x r,  A: r x d_in,  r << min(d_out, d_in)

so only B and A are trained and W is frozen. Parameters drop from
`d_out * d_in` to `r * (d_out + d_in)`.

    W' x = W x + (alpha / r) * B A x

THE SCALING FACTOR, and why it is alpha/r rather than alpha. The intent is that
changing `r` should not force you to re-tune the learning rate. Since A is
initialised from a zero-mean distribution and B from zeros, the variance of
`BAx` grows roughly linearly with r, so dividing by r keeps the update's
magnitude roughly r-independent. `alpha` is then a single knob for "how much
adaptation", decoupled from capacity. Note the common convention alpha = 2r,
which makes the scale 2 regardless of r.

WHY B = 0 AND A ~ N(0, sigma), NOT BOTH RANDOM AND NOT BOTH ZERO.

  * both zero  -> dW = 0 forever. The gradient of the loss w.r.t. A is
    proportional to B (which is 0) and w.r.t. B is proportional to A (0), so
    both gradients vanish and the adapter never leaves the origin. This is a
    saddle point, and it is the reason a "just initialise everything to zero"
    instinct fails here specifically.
  * both random -> dW != 0 at step 0, so the adapted model does NOT equal the
    pretrained model before any training. You have silently perturbed a model
    you were trying to preserve.
  * B = 0, A random -> dW = 0 at step 0 (exact equality with the base model),
    but dL/dA = B^T(...) = 0 while dL/dB = (...)A^T != 0. So B moves first,
    then A follows. Training starts exactly from the pretrained model AND is
    not stuck.

Both properties are asserted in tests rather than assumed.

WHAT THIS MODULE DELIBERATELY DOES NOT DO. No quantization (that is QLoRA, and
it uses bitsandbytes), no dropout on the adapter path, no rank-stabilised
scaling (rsLoRA), no DoRA. Those are variations; the point here is that the
core mechanism is legible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

from alignlab.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class LoRAConfig:
    """Hyperparameters for a LoRA adaptation."""

    r: int = 16
    alpha: float = 32.0
    # Substrings matched against module names. Qwen2's attention projections are
    # q_proj/k_proj/v_proj/o_proj and its MLP is gate_proj/up_proj/down_proj -
    # VERIFIED against the loaded module tree, not assumed from a textbook.
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError(f"rank must be positive, got {self.r}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")

    @property
    def scaling(self) -> float:
        return self.alpha / self.r


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` with a trainable low-rank update.

    Wraps rather than replaces the base layer: ``self.base`` is the original
    module, kept intact so that merging and un-merging are exact and so the
    original weights can always be recovered.

    Shapes, following the project's convention of stating them explicitly::

        x        : [..., d_in]
        base(x)  : [..., d_out]
        A        : [r, d_in]        lora_A.weight
        A x      : [..., r]
        B        : [d_out, r]       lora_B.weight
        B A x    : [..., d_out]
        output   : base(x) + scaling * B A x
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear wraps nn.Linear, got {type(base).__name__}")
        if r <= 0:
            raise ValueError(f"rank must be positive, got {r}")

        self.base = base
        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.merged = False

        d_in = base.in_features
        d_out = base.out_features

        # FREEZE THE BASE. This is the whole point of PEFT and is done here, at
        # construction, rather than left to the caller to remember.
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        # bias=False on both: a bias on the adapter path would add a term that
        # is not part of B A x, so the module would no longer implement the
        # equation it claims to.
        self.lora_A = nn.Linear(d_in, r, bias=False)
        self.lora_B = nn.Linear(r, d_out, bias=False)

        # Match the base layer's dtype/device so the adapter can be added to it.
        self.lora_A.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B.to(device=base.weight.device, dtype=base.weight.dtype)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """A ~ Kaiming-uniform, B = 0. See the module docstring for why."""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def delta_weight(self) -> torch.Tensor:
        """``scaling * B A``, with the same shape as the base weight."""
        return self.scaling * (self.lora_B.weight @ self.lora_A.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            # Already folded into base.weight; adding it again would double it.
            return out
        return out + self.scaling * self.lora_B(self.lora_A(x))

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the adapter into the base weight, in place.

        After merging the module is an ordinary linear layer with no adapter
        overhead at inference - which is LoRA's practical selling point over
        adapter architectures that add depth.

        Idempotent by guard: merging twice would add ``delta_weight`` twice.
        """
        if self.merged:
            logger.debug("merge() called on an already-merged layer; ignoring")
            return
        self.base.weight.data += self.delta_weight.to(self.base.weight.dtype)
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Undo :meth:`merge`.

        Exact only up to floating-point round-off: subtracting recovers the
        original weight to within the precision of the dtype, not bitwise. The
        tests assert a tolerance rather than equality, and say why.
        """
        if not self.merged:
            logger.debug("unmerge() called on a non-merged layer; ignoring")
            return
        self.base.weight.data -= self.delta_weight.to(self.base.weight.dtype)
        self.merged = False

    def extra_repr(self) -> str:
        return (
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4f}, "
            f"merged={self.merged}"
        )


def _iter_target_linears(
    model: nn.Module, target_modules: Iterable[str]
) -> list[tuple[str, nn.Linear]]:
    """Find ``nn.Linear`` modules whose qualified name matches a target.

    Matches on the LAST path component, not on the whole qualified name.
    Substring-matching the full name would make a target like "q_proj" also
    match a hypothetical "not_q_proj_thing", and would match container names.
    """
    targets = tuple(target_modules)
    found: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in targets:
            found.append((name, module))
    return found


def apply_lora(model: nn.Module, config: LoRAConfig) -> dict:
    """Replace matching ``nn.Linear`` layers with :class:`LoRALinear`, in place.

    Every non-adapter parameter in the model is frozen, so the returned
    trainable count is the honest one rather than "the adapters plus whatever
    else happened to be trainable".

    Returns a summary dict suitable for a run manifest.
    """
    targets = _iter_target_linears(model, config.target_modules)
    if not targets:
        raise ValueError(
            f"no nn.Linear modules matched {config.target_modules!r}. "
            f"Inspect the real module names before guessing - see "
            f"scripts/experiments/e16_lora_targets.py."
        )

    total_before = sum(p.numel() for p in model.parameters())

    # Freeze EVERYTHING first, then let the new adapters be trainable. Doing it
    # in this order means a parameter that was already trainable for some other
    # reason cannot silently ride along.
    for param in model.parameters():
        param.requires_grad_(False)

    replaced = []
    for name, module in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapper = LoRALinear(module, r=config.r, alpha=config.alpha)
        setattr(parent, attr, wrapper)
        replaced.append(
            {
                "name": name,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "base_params": module.weight.numel()
                + (module.bias.numel() if module.bias is not None else 0),
                "adapter_params": wrapper.lora_A.weight.numel()
                + wrapper.lora_B.weight.numel(),
            }
        )

    total_after = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    summary = {
        "rank": config.r,
        "alpha": config.alpha,
        "scaling": config.scaling,
        "target_modules": list(config.target_modules),
        "matched_modules": len(replaced),
        "total_params_before": total_before,
        "total_params_after": total_after,
        "trainable_params": trainable,
        "trainable_fraction": trainable / total_after if total_after else 0.0,
        "adapter_params": sum(m["adapter_params"] for m in replaced),
        "modules": replaced,
    }
    logger.info(
        "LoRA r=%d alpha=%s applied to %d modules: %s trainable of %s (%.4f%%)",
        config.r, config.alpha, len(replaced),
        f"{trainable:,}", f"{total_after:,}", 100 * summary["trainable_fraction"],
    )
    return summary


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract ONLY the adapter tensors.

    This is what makes a LoRA checkpoint small: the frozen base is not saved,
    because it is already on disk as the pretrained model. Saving
    ``model.state_dict()`` instead would produce a full-size checkpoint and
    quietly discard LoRA's storage advantage.
    """
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load adapter tensors produced by :func:`lora_state_dict`.

    Strict by construction: every key must exist in the model and every
    adapter parameter in the model must be present in the state. A silently
    partial load would leave some adapters at their initialisation - which,
    because B starts at zero, means those layers contribute nothing and the
    model looks "almost right" rather than broken.
    """
    own = dict(model.named_parameters())
    missing = [k for k in state if k not in own]
    if missing:
        raise KeyError(f"state contains keys not in the model: {missing[:5]}")

    expected = {n for n in own if "lora_A" in n or "lora_B" in n}
    absent = expected - set(state)
    if absent:
        raise KeyError(f"state is missing adapter keys: {sorted(absent)[:5]}")

    with torch.no_grad():
        for name, tensor in state.items():
            own[name].copy_(tensor.to(own[name].device, own[name].dtype))


def merge_all(model: nn.Module) -> int:
    """Merge every :class:`LoRALinear` in the model. Returns how many."""
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
            count += 1
    return count


def unmerge_all(model: nn.Module) -> int:
    """Un-merge every :class:`LoRALinear` in the model. Returns how many."""
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()
            count += 1
    return count


def count_parameters(model: nn.Module) -> dict[str, int | float]:
    """Total / trainable / frozen counts, for manifests and comparisons."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_fraction": trainable / total if total else 0.0,
    }
