"""Runnable examples for PYTORCH_CONCEPTS/pytorch-tensor-manipulation.md.

Every "Actual result" in that document comes from running this file.
    python PYTORCH_CONCEPTS/examples/tensor_reshaping_examples.py
"""
from __future__ import annotations

import torch
import torch.nn as nn


def banner(t): print("\n" + "=" * 70); print(t); print("=" * 70)


banner("1. view vs reshape vs transpose vs permute")
x = torch.arange(24).reshape(2, 3, 4)
print("x                 :", tuple(x.shape))
print("x.view(2,12)      :", tuple(x.view(2, 12).shape), "- reinterprets, no copy")
print("x.transpose(1,2)  :", tuple(x.transpose(1, 2).shape), "- swaps TWO axes")
print("x.permute(2,0,1)  :", tuple(x.permute(2, 0, 1).shape), "- arbitrary reorder")
print("is_contiguous after transpose:", x.transpose(1, 2).is_contiguous())
print("-> transpose returns a VIEW with different strides, not a copy")


banner("2. Why .contiguous() is required before view")
x = torch.arange(24).reshape(2, 3, 4)
t = x.transpose(1, 2)
try:
    t.view(2, 12)
    print("view succeeded (unexpected)")
except RuntimeError as e:
    print("t.view(2,12) raised:", str(e).split(".")[0])
print("t.contiguous().view(2,12):", tuple(t.contiguous().view(2, 12).shape), "OK")
print("t.reshape(2,12)          :", tuple(t.reshape(2, 12).shape),
      "OK - reshape copies when it must")
print("-> this is the exact bug in _merge_heads if .contiguous() is omitted")


banner("3. The head split IS a reshape - proven against explicit slicing")
B, T, H, d_k = 1, 2, 3, 4
x = torch.arange(B * T * H * d_k, dtype=torch.float32).reshape(B, T, H * d_k)
via_view = x.view(B, T, H, d_k).transpose(1, 2)
via_slice = torch.stack([x[..., h * d_k:(h + 1) * d_k] for h in range(H)], dim=1)
print("view+transpose:", tuple(via_view.shape), " slicing:", tuple(via_slice.shape))
print("identical:", torch.equal(via_view, via_slice))
print("-> the 'head split' is just a reinterpretation of the feature axis")


banner("4. Broadcasting a [T,T] mask over [B,H,T,T] scores")
scores = torch.zeros(2, 4, 5, 5)
mask = torch.ones(5, 5, dtype=torch.bool).tril()
out = scores.masked_fill(~mask, float("-inf"))
print("scores:", tuple(scores.shape), " mask:", tuple(mask.shape),
      " result:", tuple(out.shape))
print("broadcast rule: trailing dims align, missing leading dims are added")
print("row 0 of head 0:", out[0, 0, 0].tolist())


banner("5. masked_fill with -inf, and why it must precede softmax")
scores = torch.tensor([[1.0, 2.0, 3.0]])
mask = torch.tensor([[True, True, False]])
before = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
after = torch.softmax(scores, dim=-1) * mask
print("mask BEFORE softmax:", [round(v, 4) for v in before[0].tolist()],
      "sum =", round(float(before.sum()), 6))
print("mask AFTER  softmax:", [round(v, 4) for v in after[0].tolist()],
      "sum =", round(float(after.sum()), 6))
print("-> masking after leaves the row summing to < 1, shrinking the output")


banner("6. softmax axis: dim=-1 vs dim=-2")
s = torch.randn(1, 1, 3, 3)
over_keys = torch.softmax(s, dim=-1)
over_queries = torch.softmax(s, dim=-2)
print("softmax(dim=-1) row sums   :", [round(v, 4) for v in over_keys[0, 0].sum(-1).tolist()])
print("softmax(dim=-2) row sums   :", [round(v, 4) for v in over_queries[0, 0].sum(-1).tolist()])
print("softmax(dim=-2) COLUMN sums:", [round(v, 4) for v in over_queries[0, 0].sum(-2).tolist()])
print("-> dim=-1 gives each QUERY a distribution over keys. That is what we want.")


banner("7. register_buffer vs nn.Parameter")
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(3))
        self.register_buffer("persistent_mask", torch.ones(3))
        self.register_buffer("temp_table", torch.ones(3), persistent=False)
        self.plain = torch.ones(3)          # neither - a trap

m = M()
print("parameters      :", [n for n, _ in m.named_parameters()])
print("buffers         :", [n for n, _ in m.named_buffers()])
print("state_dict keys :", list(m.state_dict().keys()))
print("-> 'temp_table' is a buffer but NOT in state_dict (persistent=False)")
print("-> 'plain' is in NEITHER: not saved, and NOT moved by .to(device)")
if torch.cuda.is_available():
    m2 = M().cuda()
    print("after .cuda(): buffer on", m2.persistent_mask.device,
          "| plain attr on", m2.plain.device, "<- STILL CPU")
else:
    print("(CUDA unavailable here; the .to(device) trap is demonstrated on the server)")


banner("8. torch.multinomial respects zero probabilities")
probs = torch.tensor([[0.0, 0.5, 0.0, 0.5]])
gen = torch.Generator().manual_seed(0)
draws = [int(torch.multinomial(probs, 1, generator=gen)) for _ in range(500)]
print("500 draws from [0, .5, 0, .5] gave indices:", sorted(set(draws)))
print("-> tokens masked to -inf get softmax prob exactly 0 and are never drawn")


banner("9. cross_entropy takes LOGITS, never softmax output")
logits = torch.tensor([[2.0, 1.0, 0.1]])
target = torch.tensor([0])
import torch.nn.functional as F
correct = F.cross_entropy(logits, target)
wrong = F.cross_entropy(torch.softmax(logits, -1), target)
print("cross_entropy(logits)          :", round(float(correct), 6), "  CORRECT")
print("cross_entropy(softmax(logits)) :", round(float(wrong), 6), "  WRONG (double softmax)")
print("-> cross_entropy applies log_softmax internally")


banner("DONE")
print("All output above was produced by executing this file.")
