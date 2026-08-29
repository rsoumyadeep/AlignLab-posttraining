"""Checkpoint save, load, rotation and resume-fidelity.

The central claim: a checkpoint restores model weights, optimizer state,
scheduler state, step counter and RNG state exactly. Anything weaker is not a
resume - it is a restart that happens to begin from similar weights.

Equality is asserted with torch.equal (bitwise), not allclose. A resume that
is merely close is a bug, not a rounding artefact.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from alignlab.checkpoint import (
    LATEST_LINK_NAME,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from alignlab.seeding import set_seed


def _states_equal(a: dict, b: dict) -> bool:
    """Bitwise comparison of two state_dicts."""
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[key], b[key]) for key in a)


def test_save_creates_file_and_pointer(tmp_path: Path, tiny_model) -> None:
    path = save_checkpoint(tmp_path, step=1, model=tiny_model)
    assert path.is_file()
    assert path.name == "step-000000001.pt"
    assert (tmp_path / LATEST_LINK_NAME).read_text(encoding="utf-8") == path.name


def test_no_temp_files_left_behind(tmp_path: Path, tiny_model) -> None:
    """The atomic write must not leave .tmp debris."""
    save_checkpoint(tmp_path, step=1, model=tiny_model)
    assert list(tmp_path.glob("*.tmp")) == []


def test_model_round_trip_is_bitwise_exact(tmp_path: Path, tiny_model) -> None:
    original = {k: v.clone() for k, v in tiny_model.state_dict().items()}
    path = save_checkpoint(tmp_path, step=5, model=tiny_model)

    # Perturb the live model so a no-op load cannot pass.
    with torch.no_grad():
        for param in tiny_model.parameters():
            param.add_(1.0)
    assert not _states_equal(original, tiny_model.state_dict())

    load_checkpoint(path, model=tiny_model)
    assert _states_equal(original, tiny_model.state_dict())


def test_optimizer_and_scheduler_round_trip(tmp_path: Path, tiny_model) -> None:
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # Take real steps so the optimizer accumulates non-trivial state
    # (Adam moment estimates) rather than saving a freshly-initialised one.
    for _ in range(4):
        loss = tiny_model(torch.randn(3, 4)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    expected_lr = scheduler.get_last_lr()
    path = save_checkpoint(
        tmp_path, step=4, model=tiny_model, optimizer=optimizer, scheduler=scheduler
    )

    fresh_optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=999.0)
    fresh_scheduler = torch.optim.lr_scheduler.StepLR(
        fresh_optimizer, step_size=2, gamma=0.1
    )
    payload = load_checkpoint(
        path, optimizer=fresh_optimizer, scheduler=fresh_scheduler
    )

    assert payload.step == 4
    assert fresh_optimizer.param_groups[0]["lr"] == pytest.approx(
        optimizer.param_groups[0]["lr"]
    )
    assert fresh_scheduler.get_last_lr() == expected_lr

    # Adam moment estimates and step counts must survive, otherwise bias
    # correction restarts and the first post-resume update is wrong.
    # Compared tensor-by-tensor: a plain dict == on tensor values raises.
    restored_state = fresh_optimizer.state_dict()["state"]
    original_state = optimizer.state_dict()["state"]
    assert restored_state.keys() == original_state.keys()
    for param_id, entry in original_state.items():
        for key, value in entry.items():
            restored = restored_state[param_id][key]
            if isinstance(value, torch.Tensor):
                assert torch.equal(restored, value), f"param {param_id} field {key}"
            else:
                assert restored == value, f"param {param_id} field {key}"


def test_rng_state_survives_round_trip(tmp_path: Path, tiny_model) -> None:
    """A resumed run must continue the same random stream."""
    set_seed(31337)
    path = save_checkpoint(tmp_path, step=1, model=tiny_model, include_rng=True)
    expected = [random.random() for _ in range(3)]

    for _ in range(50):
        random.random()

    load_checkpoint(path, restore_rng=True)
    assert [random.random() for _ in range(3)] == expected


def test_metadata_round_trip(tmp_path: Path, tiny_model) -> None:
    config = {"train": {"learning_rate": 3e-4}, "seed": 42}
    path = save_checkpoint(
        tmp_path,
        step=7,
        epoch=2,
        model=tiny_model,
        config=config,
        extra={"note": "phase-1 smoke"},
    )
    payload = load_checkpoint(path)
    assert payload.step == 7
    assert payload.epoch == 2
    assert payload.config == config
    assert payload.extra["note"] == "phase-1 smoke"


def test_rotation_keeps_only_newest(tmp_path: Path, tiny_model) -> None:
    for step in range(1, 6):
        save_checkpoint(tmp_path, step=step, model=tiny_model, keep_last=2)

    remaining = [p.name for p in list_checkpoints(tmp_path)]
    assert remaining == ["step-000000004.pt", "step-000000005.pt"]


def test_rotation_disabled_keeps_everything(tmp_path: Path, tiny_model) -> None:
    for step in range(1, 4):
        save_checkpoint(tmp_path, step=step, model=tiny_model, keep_last=None)
    assert len(list_checkpoints(tmp_path)) == 3


def test_latest_checkpoint_uses_pointer(tmp_path: Path, tiny_model) -> None:
    save_checkpoint(tmp_path, step=1, model=tiny_model)
    newest = save_checkpoint(tmp_path, step=2, model=tiny_model)
    assert latest_checkpoint(tmp_path) == newest


def test_latest_checkpoint_recovers_from_stale_pointer(
    tmp_path: Path, tiny_model
) -> None:
    """A run killed between the rename and the pointer write stays resumable."""
    save_checkpoint(tmp_path, step=1, model=tiny_model)
    newest = save_checkpoint(tmp_path, step=2, model=tiny_model)
    (tmp_path / LATEST_LINK_NAME).write_text("step-000000999.pt", encoding="utf-8")
    assert latest_checkpoint(tmp_path) == newest


def test_latest_checkpoint_none_when_empty(tmp_path: Path) -> None:
    assert latest_checkpoint(tmp_path) is None
    assert list_checkpoints(tmp_path) == []


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "does-not-exist.pt")


def test_steps_sort_correctly_past_ten(tmp_path: Path, tiny_model) -> None:
    """Zero-padded step numbers must sort numerically, not lexicographically.

    Without padding, step-9 would sort after step-10 and rotation would delete
    the wrong file.
    """
    for step in (2, 9, 10, 100):
        save_checkpoint(tmp_path, step=step, model=tiny_model, keep_last=None)
    names = [p.name for p in list_checkpoints(tmp_path)]
    assert names == [
        "step-000000002.pt",
        "step-000000009.pt",
        "step-000000010.pt",
        "step-000000100.pt",
    ]
