#!/usr/bin/env python3
"""Step 5b (HAWQ v1.1 DPO plan) - verify the hand-rolled DPO loss math
BEFORE any GPU spend. Needs only torch, no GPU, no model.

Imports dpo_loss from scripts/dpo_common.py - the SAME function
scripts/train_dpo.py uses - so these tests validate what actually runs on
the A100, not a disconnected duplicate.

    logits = (pol_chosen_lp - ref_chosen_lp) - (pol_rejected_lp - ref_rejected_lp)
    loss   = -logsigmoid(beta * logits)

Run with plain python3:
    python3 scripts/test_dpo_loss.py
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dpo_common import dpo_loss  # noqa: E402

BETA = 0.1  # must match train_dpo.py's beta - passed explicitly below so a
            # local change here can never silently diverge from training.


def test_equal_logprobs_gives_log_half():
    """No preference signal at all (policy == reference on both sides, or
    policy's advantage over reference is identical for chosen and rejected)
    -> logits=0 -> loss = -log(sigmoid(0)) = -log(0.5) = log(2) ~= 0.6931."""
    lp = torch.tensor([-50.0])
    loss = dpo_loss(lp, lp, lp, lp, beta=BETA)
    expected = math.log(2)
    assert abs(loss.item() - expected) < 1e-5, (
        f"expected {expected:.6f}, got {loss.item():.6f}")
    print(f"[test] equal_logprobs -> loss={loss.item():.6f} "
          f"(expected {expected:.6f}) PASS")


def test_strongly_preferred_chosen_drives_loss_toward_zero():
    """Policy strongly prefers chosen over reference's baseline (large
    positive logits) -> sigmoid(beta*logits) -> 1 -> loss -> 0."""
    pol_chosen = torch.tensor([0.0])
    pol_rejected = torch.tensor([-100.0])
    ref_chosen = torch.tensor([-50.0])
    ref_rejected = torch.tensor([-50.0])
    loss = dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta=BETA)
    assert loss.item() < 1e-3, f"expected loss near 0, got {loss.item():.6f}"
    print(f"[test] strongly_preferred_chosen -> loss={loss.item():.6f} "
          f"(expected ~0) PASS")


def test_strongly_preferred_rejected_drives_loss_large():
    """Policy strongly prefers rejected over reference's baseline (large
    negative logits) -> sigmoid(beta*logits) -> 0 -> loss -> +inf (large)."""
    pol_chosen = torch.tensor([-100.0])
    pol_rejected = torch.tensor([0.0])
    ref_chosen = torch.tensor([-50.0])
    ref_rejected = torch.tensor([-50.0])
    loss = dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta=BETA)
    assert loss.item() > 5.0, f"expected large loss, got {loss.item():.6f}"
    print(f"[test] strongly_preferred_rejected -> loss={loss.item():.6f} "
          f"(expected large) PASS")


def test_invariant_to_constant_shift_in_reference():
    """Adding the same constant C to both reference log-probs must not
    change the loss - only the DIFFERENCE (pol - ref) per side, and the
    difference-of-differences, matters."""
    pol_chosen = torch.tensor([-10.0])
    pol_rejected = torch.tensor([-20.0])
    ref_chosen = torch.tensor([-15.0])
    ref_rejected = torch.tensor([-18.0])
    base_loss = dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta=BETA)
    for c in (-37.5, 0.0, 12.25, 1000.0):
        shifted_loss = dpo_loss(pol_chosen, pol_rejected,
                                 ref_chosen + c, ref_rejected + c, beta=BETA)
        assert abs(shifted_loss.item() - base_loss.item()) < 1e-5, (
            f"shift c={c}: expected {base_loss.item():.6f}, "
            f"got {shifted_loss.item():.6f}")
    print(f"[test] invariant_to_constant_reference_shift -> base_loss="
          f"{base_loss.item():.6f}, invariant across shifts PASS")


def test_batch_mean_matches_manual_average():
    """Sanity: batched call == mean of per-example scalar losses."""
    pol_chosen = torch.tensor([-10.0, -5.0, -30.0])
    pol_rejected = torch.tensor([-20.0, -25.0, -5.0])
    ref_chosen = torch.tensor([-15.0, -5.0, -15.0])
    ref_rejected = torch.tensor([-18.0, -22.0, -20.0])
    batched = dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta=BETA)
    manual = torch.stack([
        dpo_loss(pol_chosen[i:i+1], pol_rejected[i:i+1],
                  ref_chosen[i:i+1], ref_rejected[i:i+1], beta=BETA)
        for i in range(3)
    ]).mean()
    assert abs(batched.item() - manual.item()) < 1e-5
    print(f"[test] batch_mean_matches_manual_average -> "
          f"batched={batched.item():.6f} manual={manual.item():.6f} PASS")


if __name__ == "__main__":
    test_equal_logprobs_gives_log_half()
    test_strongly_preferred_chosen_drives_loss_toward_zero()
    test_strongly_preferred_rejected_drives_loss_large()
    test_invariant_to_constant_shift_in_reference()
    test_batch_mean_matches_manual_average()
    print("[test_dpo_loss] ALL PASS")
