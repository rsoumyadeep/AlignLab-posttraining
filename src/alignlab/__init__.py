"""AlignLab - an LLM post-training framework built from first principles.

Scope note (see PROJECT_INSTRUCTIONS.md section 1): AlignLab does NOT pretrain
a language model. It begins from a pretrained base checkpoint and covers what
happens after pretraining - supervised fine-tuning, parameter-efficient
fine-tuning, preference optimisation and evaluation.

Phase 1 provides the engineering foundation only: configuration, logging,
seeding, checkpointing, experiment tracking, preemption handling and the
environment abstraction that lets the same code run on a local CPU machine and
on a GPU server without machine-specific paths in the source tree.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
