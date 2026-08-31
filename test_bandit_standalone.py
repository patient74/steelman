"""
test_bandit_standalone.py

Phase 2 integration smoke test: runs the bandit against a REAL
PaymentEnvironmentV24 instance and a REAL BlueModel (pre-trained
on the static baseline from Phase 1), for a handful of
generations, with no LLM involved. This is the "safe fallback"
path we discussed - it must work on its own before Phase 3 (LLM
strategist) is layered on top.

Not a unit test in the pytest sense - a runnable script that
prints what's happening so the loop can be sanity-checked by
eye before trusting it.
"""

import sys
from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent))
# sys.path.insert(0, "/home/claude/test")

import pandas as pd

from blue_model import BlueModel, build_graph_features
from red_bandit import RedBandit
from payment_environment_v24 import (
    ATTACK_PARAM_SCHEMA,
    AttackAction,
    PaymentEnvironmentV23 as PaymentEnvironment,
)


def main():

    print("=" * 60)
    print("Loading pre-trained baseline blue model...")
    print("=" * 60)

    blue = BlueModel()
    blue.load("output_v2/blue_model_baseline")
    print(f"Loaded. Decision threshold = {blue.decision_threshold:.4f}")

    print()
    print("=" * 60)
    print("Spinning up a fresh simulator instance for red to attack...")
    print("=" * 60)

    env = PaymentEnvironment(
        customers=1500,
        merchants=150,
        days=20,
        seed=123,
        calibration_path=(
            "data/calibration/"
            "master_calibration.json"
        ),
    )
    env.generate_entities()
    env.generate_base_transactions()

    print(f"Base transactions generated: {len(env.transactions):,}")

    red = RedBandit(ATTACK_PARAM_SCHEMA, n_bins=7, seed=7)

    n_generations = 12
    campaigns_per_generation = 4

    generation_log = []

    print()
    print("=" * 60)
    print(f"Running {n_generations} generations, "
          f"{campaigns_per_generation} campaigns each "
          f"(no LLM - bandit chooses family + params)")
    print("=" * 60)

    for gen in range(n_generations):

        gen_records = []

        for _ in range(campaigns_per_generation):

            family = red.choose_family()

            params, bin_indices = red.propose(family)

            action = AttackAction(
                family=family,
                campaign_size=5,
                params=params,
            )

            result = env.execute_attack_action(action)

            if result["actual_size"] == 0:
                # e.g. no eligible target (can happen early for
                # agent families before any campaigns landed) -
                # skip scoring, don't update the bandit on a
                # null trial.
                continue

            txn_ids = result["transaction_ids"]

            all_txns_df = pd.DataFrame(env.transactions)

            batch_df = all_txns_df[
                all_txns_df["transaction_id"].isin(txn_ids)
            ]

            graph_feats = build_graph_features(all_txns_df)

            scores = blue.score(
                batch_df, graph_features=graph_feats
            )

            mean_margin = float(
                scores.loc[txn_ids, "detection_margin"].mean()
            )

            detected_count = int(
                scores.loc[txn_ids, "flagged"].sum()
            )

            reward = red.update(
                family,
                bin_indices,
                mean_margin,
                extra_log={
                    "generation": gen,
                    "attack_id": result["attack_id"],
                    "actual_size": result["actual_size"],
                    "detected_count": detected_count,
                    "params": params,
                },
            )

            gen_records.append(
                {
                    "generation": gen,
                    "family": family,
                    "params": params,
                    "mean_detection_margin": mean_margin,
                    "detected_count": detected_count,
                    "actual_size": result["actual_size"],
                    "evasion_rate": 1
                    - (detected_count / result["actual_size"]),
                    "reward": reward,
                }
            )

        if gen_records:

            gen_df = pd.DataFrame(gen_records)
            generation_log.extend(gen_records)

            print(
                f"Gen {gen:2d} | "
                f"mean evasion_rate={gen_df['evasion_rate'].mean():.3f} | "
                f"mean margin={gen_df['mean_detection_margin'].mean():+.3f} | "
                f"mean reward={gen_df['reward'].mean():.3f} | "
                f"families this gen: "
                f"{gen_df['family'].value_counts().to_dict()}"
            )

    print()
    print("=" * 60)
    print("Done. Per-family posterior summary "
          "(best-known reward per knob):")
    print("=" * 60)

    for family, fb in red.families.items():

        if red.family_trial_counts[family] == 0:
            continue

        summary = fb.posterior_summary()

        print(f"\n{family} (trials={red.family_trial_counts[family]}):")

        for knob_name, knob_summary in summary.items():

            centers = knob_summary["bin_centers"]
            means = knob_summary["posterior_mean_reward"]

            best_idx = max(
                range(len(means)), key=lambda i: means[i]
            )

            print(
                f"  {knob_name:24s} best bin center="
                f"{centers[best_idx]:.2f}  "
                f"posterior_mean_reward={means[best_idx]:.3f}"
            )

    log_df = pd.DataFrame(generation_log)
    log_df.to_json(
        "output_v2/bandit_standalone_log.json",
        orient="records",
        indent=2,
    )

    print()
    print(f"\nSaved full trial log: "
          f"output_v2/bandit_standalone_log.json "
          f"({len(log_df)} trials)")

    print()
    print("Generation-level mean evasion rate (should trend up "
          "if the bandit is learning anything real):")
    print(
        log_df.groupby("generation")["evasion_rate"].mean()
        .to_string()
    )


if __name__ == "__main__":
    main()
