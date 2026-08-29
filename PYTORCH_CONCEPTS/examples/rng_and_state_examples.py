"""Runnable examples for PYTORCH_CONCEPTS/pytorch-rng-and-state.md.

Every "Actual result" in that document is produced by running this file. Run it
yourself to confirm - the numbers should match on any machine with the same
torch version, and the whole point of Tier A is that they do.

    python PYTORCH_CONCEPTS/examples/rng_and_state_examples.py
"""

from __future__ import annotations

import random

import numpy as np
import torch


def banner(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


# ---------------------------------------------------------------------------
banner("1. torch.manual_seed makes draws repeatable")

torch.manual_seed(0)
a = torch.rand(3)
torch.manual_seed(0)
b = torch.rand(3)

print("a           :", a.tolist())
print("b           :", b.tolist())
print("bitwise equal:", torch.equal(a, b))


# ---------------------------------------------------------------------------
banner("2. Seeding torch alone does NOT seed random or numpy")

torch.manual_seed(0)
first = (random.random(), float(np.random.rand()))
torch.manual_seed(0)
second = (random.random(), float(np.random.rand()))

print("first  (random, numpy):", first)
print("second (random, numpy):", second)
print("equal:", first == second)
print("-> this is why set_seed() seeds all three, not just torch")


# ---------------------------------------------------------------------------
banner("3. RNG state can be captured and restored mid-stream")

torch.manual_seed(1234)
state = torch.get_rng_state()
expected = torch.rand(3)

_ = torch.rand(100)  # consume the stream

torch.set_rng_state(state)
restored = torch.rand(3)

print("expected :", expected.tolist())
print("restored :", restored.tolist())
print("bitwise equal:", torch.equal(expected, restored))
print("-> this is what makes a resumed run continue the same data order")


# ---------------------------------------------------------------------------
banner("4. RNG state is a uint8 CPU tensor, not an int")

state = torch.get_rng_state()
print("type  :", type(state).__name__)
print("dtype :", state.dtype)
print("device:", state.device)
print("shape :", tuple(state.shape))
print("-> it must be saved as a tensor; int(seed) is not enough to resume")


# ---------------------------------------------------------------------------
banner("5. parameters vs buffers - both are in state_dict, one is trained")


class WithBuffer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.register_buffer("running_count", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.running_count += 1
        return x * self.weight


module = WithBuffer()
print("named_parameters:", [n for n, _ in module.named_parameters()])
print("named_buffers   :", [n for n, _ in module.named_buffers()])
print("state_dict keys :", list(module.state_dict().keys()))
print("requires_grad(weight)        :", module.weight.requires_grad)
print("requires_grad(running_count) :", module.running_count.requires_grad)
print("-> buffers are saved and moved with .to(device) but never optimised")


# ---------------------------------------------------------------------------
banner("6. Optimizer state is NOT empty - it must be checkpointed")

model = torch.nn.Linear(3, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

print("state before any step:", optimizer.state_dict()["state"])

for _ in range(3):
    loss = model(torch.randn(4, 3)).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

state_after = optimizer.state_dict()["state"]
first_param = state_after[0]
print("keys after 3 steps  :", sorted(first_param.keys()))
print("step count          :", first_param["step"])
print("exp_avg shape       :", tuple(first_param["exp_avg"].shape))
print("-> discarding this restarts Adam bias correction after a resume")


# ---------------------------------------------------------------------------
banner("7. state_dict holds references, not copies")

model = torch.nn.Linear(2, 1)
snapshot = model.state_dict()
before = snapshot["weight"].clone()

with torch.no_grad():
    model.weight.add_(1.0)

print("snapshot changed underneath us:", not torch.equal(snapshot["weight"], before))
print("-> torch.save serialises immediately so this is safe there,")
print("   but an in-memory 'backup' needs an explicit .clone()")


# ---------------------------------------------------------------------------
banner("8. Float addition is not associative - why cross-machine differs")

# NOTE: TWO earlier attempts at this example FAILED to show anything.
#   attempt 1: [1.0, 1e-8, -1.0]   -> both orderings gave 0.0
#   attempt 2: [1e8, 1.0, -1e8]    -> both orderings gave 0.0
# In both cases fp32 rounding collapsed the two groupings to the same value.
# Preserved because it is the actual lesson: a demonstration must be RUN and
# checked, never assumed to work because the reasoning sounded right.
#
# What reliably works: sum many values in two different orders. This is
# exactly what different GPU reduction kernels do.
torch.manual_seed(0)
vals = torch.rand(10000, dtype=torch.float32) * 1000.0

forward = torch.zeros((), dtype=torch.float32)
for v in vals:
    forward = forward + v

backward = torch.zeros((), dtype=torch.float32)
for v in vals.flip(0):
    backward = backward + v

print("sum forward :", forward.item())
print("sum backward:", backward.item())
print("equal       :", forward.item() == backward.item())
print("difference  :", abs(forward.item() - backward.item()))
print("-> identical numbers, identical operation, different ORDER.")
print("   Different GPUs reduce in different orders, so a different machine")
print("   gives a different result. This is Tier C, not a bug.")


# ---------------------------------------------------------------------------
banner("9. bf16 has far less mantissa precision than fp32")

# An earlier version compared a single constant across dtypes. That was a bad
# demonstration: 3.14159... and 1.2345678 both happen to round to the SAME
# value in bf16 and fp16, making the two look equally precise. torch.finfo
# reports the real limits directly and needs no lucky constant.
import math

print(f"{'dtype':<18}{'eps':<26}{'max':<26}mantissa bits")
for dtype in (torch.float32, torch.bfloat16, torch.float16):
    info = torch.finfo(dtype)
    bits = abs(round(math.log2(info.eps)))
    print(f"{str(dtype):<18}{info.eps:<26}{info.max:<26}~{bits}")

print()
print("-> bf16 has ~7 mantissa bits vs fp32's ~23: far COARSER precision,")
print("   but nearly fp32's exponent range. fp16 has more mantissa than bf16")
print("   yet overflows at 65504:")
big = 1e30
print("   1e30 in fp16:", torch.tensor(big, dtype=torch.float16).item())
print("   1e30 in bf16:", torch.tensor(big, dtype=torch.bfloat16).item())
print("-> that range is why Ampere training uses bf16, and why a CPU fp32 run")
print("   and an Ampere bf16 run can never be compared bitwise.")


banner("DONE")
print("All output above was produced by executing this file.")
