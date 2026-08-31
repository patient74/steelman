"""
red_bandit.py
=============

Phase 2 of the Mastercard Innovation Challenge 2026 submission:
the tactical (numeric) layer of the "Generate" pillar.

For each attack family in ATTACK_PARAM_SCHEMA, maintains an
independent Gaussian-Thompson-sampling bandit over that family's
continuous [0,1] parameter vector. Each generation:

  1. bandit.propose(family) -> a params dict to try
  2. simulator.execute_attack_action(AttackAction(family, params, ...))
  3. blue.score(...) on the resulting transactions -> detection_margin
  4. bandit.update(family, params, reward) where reward is derived
     from detection_margin (see `margin_to_reward`)

This module is deliberately standalone and has no dependency on
an LLM - it is fully testable and demoable on its own (this was
an explicit design decision: the bandit is the safe fallback if
the LLM strategist, added in Phase 3, runs into trouble).

Algorithm choice: discretized Thompson sampling over each
parameter independently (a small grid per knob, e.g. 5-9 bins),
rather than continuous Bayesian optimization. This is a
deliberate simplicity choice:
  - trivial to implement correctly (no GP library, no kernel
    tuning) in the time available,
  - trivial to explain to judges - "here is the posterior belief
    over each bin, sampled from, in this table" - which is worth
    more for the novelty/explainability criteria than a marginal
    gain in sample efficiency from full continuous BO,
  - converges fast enough (tens of generations) given each
    family's evaluation ("try these params, see if they evade
    blue") is cheap relative to typical bandit applications.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ============================================================
# Reward shaping
# ============================================================
#
# detection_margin = fraud_probability - decision_threshold from
# BlueModel.score(). Red wants this LOW (below 0 = evaded
# entirely; even a smaller positive margin than before is
# meaningful partial progress, since it means red is getting
# closer to the boundary blue actually uses).
#
# reward is mapped to [0, 1] via a logistic squashing of
# -margin, so:
#   margin << 0 (fully evaded, confidently)      -> reward -> 1
#   margin == 0 (right at blue's decision edge)  -> reward = 0.5
#   margin >> 0 (caught, confidently)            -> reward -> 0
#
# This keeps the bandit's reward in the [0,1] range Beta-based
# Thompson sampling expects, while still being a CONTINUOUS
# signal (not the binary "caught y/n"), which is what lets the
# bandit distinguish "almost evaded" from "wildly overshot and
# got caught easily" even though both are nominally "detected".

def margin_to_reward(margin: float, steepness: float = 6.0) -> float:

    return float(
        1.0 / (1.0 + np.exp(steepness * margin))
    )


# ============================================================
# Per-knob discretized Thompson sampling
# ============================================================

@dataclass
class KnobBandit:
    """
    One Beta-Bernoulli-style Thompson sampling bandit over a
    single discretized [0,1] knob (e.g. mimicry_blend for
    BEHAVIORAL_MIMICRY).
    """

    n_bins: int = 7
    alpha: np.ndarray = field(default=None)
    beta: np.ndarray = field(default=None)

    def __post_init__(self):

        if self.alpha is None:
            self.alpha = np.ones(self.n_bins)

        if self.beta is None:
            self.beta = np.ones(self.n_bins)

    @property
    def bin_centers(self) -> np.ndarray:

        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        return (edges[:-1] + edges[1:]) / 2.0

    def sample_bin(
        self, rng: random.Random, epsilon: float = 0.0
    ) -> int:

        # Exploration floor: with probability epsilon, pick a
        # uniformly random bin instead of the Thompson-sampled
        # argmax. Pure Thompson sampling explores less and less
        # as a bin's alpha/beta mass grows - which is correct
        # for pure evasion-maximizing search, but WRONG for this
        # project's actual use case: the retrain step needs
        # variety (successful AND unsuccessful param values) to
        # have any chance of sharpening blue's decision boundary.
        # Without this floor, red converges onto 1-2 bins per
        # family and every retrain sees only near-duplicate
        # "already evasive" examples - verified in real runs
        # (Aug 2026) to produce flat-to-negative adaptive vs
        # frozen comparisons even after the threshold-drift bug
        # was fixed.

        if epsilon > 0.0 and rng.random() < epsilon:

            return rng.randrange(self.n_bins)

        samples = [
            np.random.beta(self.alpha[i], self.beta[i])
            for i in range(self.n_bins)
        ]

        return int(np.argmax(samples))

    def propose(
        self, rng: random.Random, epsilon: float = 0.0
    ) -> tuple[int, float]:

        b = self.sample_bin(rng, epsilon=epsilon)
        return b, float(self.bin_centers[b])

    def update(self, bin_idx: int, reward: float):

        # reward in [0,1] is treated as a "soft" Bernoulli
        # outcome: reward itself adds to alpha (success mass),
        # (1-reward) adds to beta (failure mass). This is a
        # standard extension of Beta-Bernoulli Thompson sampling
        # to continuous rewards in [0,1].
        reward = float(np.clip(reward, 0.0, 1.0))

        self.alpha[bin_idx] += reward
        self.beta[bin_idx] += (1.0 - reward)

    def posterior_means(self) -> np.ndarray:

        return self.alpha / (self.alpha + self.beta)

    def to_dict(self):
        return {
            "n_bins": self.n_bins,
            "alpha": self.alpha.tolist(),
            "beta": self.beta.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            n_bins=d["n_bins"],
            alpha=np.array(d["alpha"]),
            beta=np.array(d["beta"]),
        )


class FamilyBandit:
    """
    One KnobBandit per continuous parameter in a family's
    ATTACK_PARAM_SCHEMA entry. Families with an empty param
    schema (the original 6 v2.3 families - CARD_TESTING,
    ACCOUNT_TAKEOVER, DEVICE_COMPROMISE, VELOCITY_ABUSE,
    UNUSUAL_GEO, MERCHANT_ANOMALY) get no knobs and simply
    propose an empty params dict every time - they are not part
    of red's searchable action space, only the 7 parameterized
    families (A1, A3, A4, A6, A9, A30, A31) are.
    """

    def __init__(self, family: str, param_schema: dict, n_bins: int = 7):

        self.family = family
        self.param_names = list(param_schema.keys())
        self.knobs = {
            name: KnobBandit(n_bins=n_bins)
            for name in self.param_names
        }

    def propose(
        self, rng: random.Random, epsilon: float = 0.0
    ) -> tuple[dict, dict]:
        """
        Returns (params, bin_indices). bin_indices must be
        passed back to .update() so the correct posterior cell
        gets credited.
        """

        params = {}
        bin_indices = {}

        for name, knob in self.knobs.items():

            b, value = knob.propose(rng, epsilon=epsilon)
            params[name] = value
            bin_indices[name] = b

        return params, bin_indices

    def update(self, bin_indices: dict, reward: float):

        for name, b in bin_indices.items():
            self.knobs[name].update(b, reward)

    def posterior_summary(self) -> dict:

        return {
            name: {
                "bin_centers": knob.bin_centers.tolist(),
                "posterior_mean_reward": knob.posterior_means().tolist(),
            }
            for name, knob in self.knobs.items()
        }

    def to_dict(self):
        return {
            "family": self.family,
            "knobs": {
                name: knob.to_dict()
                for name, knob in self.knobs.items()
            },
        }

    @classmethod
    def from_dict(cls, d, param_schema):

        fb = cls(d["family"], param_schema, n_bins=1)
        fb.knobs = {
            name: KnobBandit.from_dict(kd)
            for name, kd in d["knobs"].items()
        }
        return fb


class RedBandit:
    """
    Top-level red-team tactical policy: one FamilyBandit per
    parameterized attack family.

    Usage:
        red = RedBandit(ATTACK_PARAM_SCHEMA)
        family = red.choose_family(rng)          # or externally chosen (Phase 3 LLM)
        params, handle = red.propose(family, rng)
        outcome = <run attack, score with blue>
        red.update(family, handle, reward)
    """

    def __init__(
        self,
        param_schema: dict,
        n_bins: int = 7,
        seed: int | None = None,
        epsilon: float = 0.15,
    ):

        self.param_schema = param_schema
        self.epsilon = epsilon

        # Only families with at least one continuous knob are
        # part of red's searchable space - see FamilyBandit
        # docstring.
        self.parameterized_families = [
            f for f, schema in param_schema.items() if schema
        ]

        self.families = {
            f: FamilyBandit(f, param_schema[f], n_bins=n_bins)
            for f in self.parameterized_families
        }

        self.rng = random.Random(seed)

        # Simple running tally per family, for reporting /
        # the "family selection frequency" chart, and to allow
        # a naive round-robin/uniform family CHOICE policy when
        # no LLM strategist is driving family selection (Phase 2
        # standalone mode).
        self.family_trial_counts = {
            f: 0 for f in self.parameterized_families
        }

        # Full history log - this feeds directly into
        # generation_log.parquet in Phase 4 and into the
        # convergence charts.
        self.history: list[dict] = []

    # --------------------------------------------------------

    def choose_family(self) -> str:
        """
        Standalone family-selection policy (used only when
        there is no LLM strategist choosing families - i.e.
        Phase 2 on its own). Simple upper-confidence-style
        selection: favor families tried least so far, breaking
        ties by which family currently has the highest posterior
        mean reward anywhere in its knob space.
        """

        least_tried = min(
            self.family_trial_counts.values()
        )

        candidates = [
            f
            for f, n in self.family_trial_counts.items()
            if n == least_tried
        ]

        if len(candidates) == 1:
            return candidates[0]

        def best_known_reward(f):
            fb = self.families[f]
            means = [
                knob.posterior_means().max()
                for knob in fb.knobs.values()
            ]
            return max(means) if means else 0.0

        return max(candidates, key=best_known_reward)

    # --------------------------------------------------------

    def propose(self, family: str, epsilon: float | None = None):

        if family not in self.families:
            raise ValueError(
                f"{family} has no continuous parameters "
                f"(check ATTACK_PARAM_SCHEMA) - it is not part "
                f"of red's tactical search space."
            )

        eps = self.epsilon if epsilon is None else epsilon

        params, bin_indices = self.families[family].propose(
            self.rng, epsilon=eps
        )

        self.family_trial_counts[family] += 1

        return params, bin_indices

    # --------------------------------------------------------

    def update(
        self,
        family: str,
        bin_indices: dict,
        detection_margin: float,
        extra_log: dict | None = None,
    ):

        reward = margin_to_reward(detection_margin)

        self.families[family].update(bin_indices, reward)

        record = {
            "family": family,
            "detection_margin": float(detection_margin),
            "reward": reward,
        }

        if extra_log:
            record.update(extra_log)

        self.history.append(record)

        return reward

    # --------------------------------------------------------

    def posterior_report(self) -> dict:

        return {
            f: fb.posterior_summary()
            for f, fb in self.families.items()
        }

    # --------------------------------------------------------

    def save(self, path: str | Path):

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "families": {
                f: fb.to_dict() for f, fb in self.families.items()
            },
            "family_trial_counts": self.family_trial_counts,
        }

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)

    # --------------------------------------------------------

    def load(self, path: str | Path):

        path = Path(path)

        with open(path, "r", encoding="utf-8") as fh:
            state = json.load(fh)

        for f, fd in state["families"].items():
            self.families[f] = FamilyBandit.from_dict(
                fd, self.param_schema[f]
            )

        self.family_trial_counts = state["family_trial_counts"]

        return self
