"""
run_closed_loop.py

Phase 4 of the Mastercard Innovation Challenge 2026 submission:
the actual CLOSED loop. Builds directly on the validated Phase
1 (blue_model.py), Phase 2 (red_bandit.py), and Phase 3
(red_strategist.py) - this script does not change any of their
interfaces, it only orchestrates them differently from
run_adversarial_loop_llm.py:

  run_adversarial_loop_llm.py (Phase 3): blue is FROZEN
  (loaded once from the static baseline, never retrained). This
  is the right test for "how evasive can red get against a
  fixed defense" - and the real Gemini run already confirmed
  the loop mechanics work end-to-end.

  run_closed_loop.py (THIS script, Phase 4): blue RETRAINS
  every `retrain_every` generations on all labeled data
  generated so far (base transactions + every red campaign's
  transactions, cumulative). This is what makes the claim
  "closed loop / self-improving defense" true rather than
  aspirational - and it's the only way to produce the
  static-vs-adaptive comparison chart that's the strongest
  single piece of evidence for the "closed-loop" judging
  criterion.

Two blue models are tracked side by side every generation:
  - `frozen_blue`: never retrained (Phase 1 baseline, held
    fixed) - this is the "static defense" comparison line.
  - `adaptive_blue`: retrained every `retrain_every`
    generations on cumulative data - this is the "our system"
    line.

Both are evaluated against the SAME transactions each
generation, so the chart is a fair apples-to-apples comparison:
does the adaptive line hold up better than the frozen line as
red keeps attacking?

*** RUN THIS ON YOUR OWN MACHINE, same as Phase 3 ***
Retraining XGBoost every few generations on a growing dataset is
real compute - not huge, but more than this sandboxed session
should be doing repeatedly. Recommended to start with a short
run (e.g. --generations 15 --retrain-every 5) to confirm timing
on your machine before committing to a long one.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from blue_model import BlueModel, build_graph_features
from red_bandit import RedBandit
from red_strategist import OpenAICompatibleClient, RedStrategist
from payment_environment_v24 import (
    ATTACK_PARAM_SCHEMA,
    AttackAction,
    PaymentEnvironmentV23 as PaymentEnvironment,
)


def run(
    backend: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
    n_generations: int,
    campaigns_per_generation: int,
    campaign_size: int,
    retrain_every: int,
    n_customers: int,
    n_merchants: int,
    n_days: int,
    output_dir: str,
    blue_model_path: str,
    calibration_path: str,
    baseline_transactions_path: str,
    seed: int,
    recalibrate_threshold: bool = False,
    bandit_epsilon: float = 0.15,
    family_exploration_rate: float = 0.25,
):

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading pre-trained baseline blue model "
          "(used as the FROZEN comparison line)...")
    frozen_blue = BlueModel()
    frozen_blue.load(blue_model_path)

    print("Loading a second copy as the ADAPTIVE model "
          "(this one will retrain)...")
    adaptive_blue = BlueModel()
    adaptive_blue.load(blue_model_path)

    print(f"Decision threshold (both start here): "
          f"{frozen_blue.decision_threshold:.4f}")

    print(
        "\nLoading realistic baseline dataset "
        f"(the diverse, ~2%-fraud population used to train the "
        f"Phase 1 model) from {baseline_transactions_path} ...\n"
        "This is mixed into every retrain alongside red's "
        "campaign transactions, so retraining doesn't overwrite "
        "the model's general fraud knowledge with only the "
        "narrow slice of families/params red happens to have "
        "explored so far."
    )

    baseline_df = pd.read_parquet(baseline_transactions_path)

    print(
        f"Baseline dataset: {len(baseline_df):,} transactions, "
        f"{baseline_df['fraud_label'].mean():.3%} fraud rate, "
        f"{baseline_df['fraud_label'].sum():.0f} fraud examples "
        f"across "
        f"{baseline_df.loc[baseline_df.fraud_label == 1, 'attack_family'].nunique()} "
        f"families"
    )

    print("\nSpinning up simulator...")
    env = PaymentEnvironment(
        customers=n_customers,
        merchants=n_merchants,
        days=n_days,
        seed=seed,
        calibration_path=calibration_path,
    )
    env.generate_entities()
    env.generate_base_transactions()
    print(f"Base transactions: {len(env.transactions):,}")

    red = RedBandit(
        ATTACK_PARAM_SCHEMA,
        n_bins=7,
        seed=seed,
        epsilon=bandit_epsilon,
    )

    if backend == "gemini":
        if not api_key:
            raise ValueError("--api-key required for --backend gemini")
        llm = OpenAICompatibleClient(
            base_url=(
                "https://generativelanguage.googleapis.com/"
                "v1beta/openai/"
            ),
            api_key=api_key,
            model=model,
            min_seconds_between_calls=4.5,
        )
    elif backend == "ollama":
        llm = OpenAICompatibleClient(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model=model,
        )
    elif backend == "custom":
        llm = OpenAICompatibleClient(
            base_url=base_url, api_key=api_key, model=model,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    strategist = RedStrategist(
        client=llm,
        parameterized_families=red.parameterized_families,
        fallback_family_chooser=red.choose_family,
        log_path=out / "strategist_reasoning_log.jsonl",
    )

    generation_log = []
    comparison_log = []  # frozen vs adaptive metrics, per generation

    print(
        f"\nRunning {n_generations} generations, "
        f"{campaigns_per_generation} campaigns/generation, "
        f"retraining adaptive_blue every {retrain_every} "
        f"generation(s), backend={backend} model={model}\n"
    )

    for gen in range(n_generations):

        t0 = time.time()

        # Red always attacks the CURRENTLY DEPLOYED adaptive
        # model's decisions (that's the model it's actually
        # trying to evade in a real closed loop) - reward comes
        # from adaptive_blue, not frozen_blue. frozen_blue is
        # scored in parallel purely for the comparison chart, it
        # never influences red's learning.

        decision = strategist.decide(
            generation=gen, history=red.history
        )

        gen_records = []

        for i in range(campaigns_per_generation):

            # Exploration override: with probability
            # family_exploration_rate, ignore the LLM's choice
            # for THIS campaign slot and force red's own
            # least-tried-family policy instead. The LLM
            # strategist is inherently exploit-only ("double
            # down on what worked" is a reasonable thing for it
            # to say every generation) - without this override,
            # red converges onto 1-2 families and every retrain
            # sees only near-duplicate examples from them, which
            # was found (real runs, Aug 2026) to produce flat-to
            # -negative adaptive-vs-frozen results even after
            # fixing the threshold-drift bug. This override is
            # what actually diversifies what blue gets to learn
            # from.

            if red.rng.random() < family_exploration_rate:
                family = red.choose_family()
                family_source = "exploration_override"
            else:
                family = decision.families[
                    i % len(decision.families)
                ]
                family_source = "strategist"

            # Knob-level exploration also applies inside
            # red.propose() via RedBandit's own epsilon
            # (constructor default 0.15) - that's independent of
            # this family-level override and always active.

            params, bin_indices = red.propose(family)

            action = AttackAction(
                family=family, campaign_size=campaign_size, params=params,
            )

            result = env.execute_attack_action(action)

            if result["actual_size"] == 0:
                continue

            txn_ids = result["transaction_ids"]

            all_txns_df = pd.DataFrame(env.transactions)

            batch_df = all_txns_df[
                all_txns_df["transaction_id"].isin(txn_ids)
            ]

            graph_feats = build_graph_features(all_txns_df)

            adaptive_scores = adaptive_blue.score(
                batch_df, graph_features=graph_feats
            )

            mean_margin = float(
                adaptive_scores.loc[
                    txn_ids, "detection_margin"
                ].mean()
            )

            detected_count = int(
                adaptive_scores.loc[txn_ids, "flagged"].sum()
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
                    "evasion_rate": 1
                    - (detected_count / result["actual_size"]),
                    "strategist_reasoning": decision.reasoning,
                    "strategist_parse_ok": decision.parse_ok,
                    "family_source": family_source,
                },
            )

            gen_records.append(red.history[-1])

        generation_log.extend(gen_records)

        # ---- comparison: how does the FROZEN model do against
        # this same generation's transactions? (evaluated, not
        # used for reward - purely for the chart) ----

        if gen_records:

            all_txns_df = pd.DataFrame(env.transactions)

            frozen_eval_rows = all_txns_df[
                all_txns_df["attack_id"].isin(
                    [r["attack_id"] for r in gen_records]
                )
            ]

            graph_feats_full = build_graph_features(all_txns_df)

            frozen_scores = frozen_blue.score(
                frozen_eval_rows, graph_features=graph_feats_full
            )

            adaptive_scores_full = adaptive_blue.score(
                frozen_eval_rows, graph_features=graph_feats_full
            )

            comparison_log.append(
                {
                    "generation": gen,
                    "n_transactions": len(frozen_eval_rows),
                    "frozen_mean_margin": float(
                        frozen_scores["detection_margin"].mean()
                    ),
                    "frozen_evasion_rate": float(
                        1 - frozen_scores["flagged"].mean()
                    ),
                    "adaptive_mean_margin": float(
                        adaptive_scores_full[
                            "detection_margin"
                        ].mean()
                    ),
                    "adaptive_evasion_rate": float(
                        1 - adaptive_scores_full["flagged"].mean()
                    ),
                    "adaptive_was_retrained_this_gen": bool(
                        (gen + 1) % retrain_every == 0
                    ),
                }
            )

        # ---- periodic retrain of adaptive_blue on cumulative
        # data (base transactions + every campaign so far) ----

        if (gen + 1) % retrain_every == 0:

            print(f"  [retraining adaptive_blue on cumulative "
                  f"data through generation {gen}...]")

            all_txns_df = pd.DataFrame(env.transactions)

            # Retrain on the realistic baseline population PLUS
            # everything red has generated so far - not on
            # env.transactions alone, which (in this closed-loop
            # script) contains ONLY red's campaign fraud and no
            # broad/diverse fraud signal at all, since
            # env.inject_fraud() is deliberately never called
            # here. Retraining on env.transactions alone was
            # found to make adaptive_blue WORSE than the frozen
            # baseline (verified on a real 12-generation run) -
            # it overwrites general fraud knowledge with an
            # extremely narrow, self-selected slice of whatever
            # families/params red has already found evasive.
            #
            # transaction_id collision guard: baseline_df and
            # env.transactions are from SEPARATE simulator
            # instances, both using sequential "T0000000001"-
            # style counters starting at 0 - concatenating them
            # directly would silently collide on
            # transaction_id and corrupt every index-based join
            # downstream (build_feature_matrix, score). Prefix
            # env's ids for the retrain set only; this is local
            # to this concat and doesn't affect red's own
            # bookkeeping (which uses the unprefixed ids
            # everywhere else).
            retrain_env_txns = all_txns_df.copy()
            retrain_env_txns["transaction_id"] = (
                "LIVE_" + retrain_env_txns["transaction_id"]
            )

            retrain_df = pd.concat(
                [baseline_df, retrain_env_txns],
                ignore_index=True,
            )

            assert retrain_df["transaction_id"].is_unique, (
                "transaction_id collision between baseline_df "
                "and live env transactions after prefixing - "
                "investigate before trusting retrain results"
            )

            graph_feats_full = build_graph_features(retrain_df)

            adaptive_blue.fit(
                retrain_df, graph_features=graph_feats_full
            )

            if recalibrate_threshold:

                adaptive_blue.calibrate_threshold(
                    retrain_df,
                    target_precision=0.5,
                    graph_features=graph_feats_full,
                )

            else:

                # Keep the threshold FIXED at whatever the
                # Phase 1 baseline calibrated (loaded at start of
                # run()). This isolates whether evasion-rate
                # changes reflect the model's actual learned
                # discrimination or just a shifting decision
                # boundary - recalibrating precision=0.5 on every
                # retrain was found (real Gemini run, Aug 2026)
                # to push the threshold up every time (0.484 ->
                # 0.587 -> 0.636 -> 0.668), which mechanically
                # suppresses recall regardless of what the model
                # actually learned. See run_closed_loop.py
                # docstring for the full writeup of this bug.
                pass

            print(
                f"  [retrain done - trained on "
                f"{len(retrain_df):,} transactions "
                f"({retrain_df['fraud_label'].sum():.0f} fraud, "
                f"{retrain_df['fraud_label'].mean():.3%} rate) "
                f"- new threshold: "
                f"{adaptive_blue.decision_threshold:.4f}]"
            )

        # ---- checkpoint every generation ----

        pd.DataFrame(generation_log).to_json(
            out / "generation_log.json", orient="records", indent=2
        )
        pd.DataFrame(comparison_log).to_json(
            out / "comparison_log.json", orient="records", indent=2
        )
        red.save(out / "bandit_final_state.json")

        elapsed = time.time() - t0

        if gen_records:
            mean_evasion = sum(
                r["evasion_rate"] for r in gen_records
            ) / len(gen_records)

            n_explored = sum(
                1 for r in gen_records
                if r.get("family_source") == "exploration_override"
            )

            comp = comparison_log[-1] if comparison_log else {}

            print(
                f"Gen {gen:3d} ({elapsed:5.1f}s) | "
                f"families={decision.families} | "
                f"adaptive_evasion={mean_evasion:.3f} | "
                f"frozen_evasion={comp.get('frozen_evasion_rate', float('nan')):.3f} "
                f"vs adaptive_evasion(full)="
                f"{comp.get('adaptive_evasion_rate', float('nan')):.3f} | "
                f"exploration_override={n_explored}/{len(gen_records)}"
            )
            print(f"           reasoning: {decision.reasoning[:150]}")
        else:
            print(f"Gen {gen:3d} ({elapsed:5.1f}s) | no campaigns landed")

    print("\nGenerating end-of-run debrief...")
    debrief = strategist.generate_debrief(red.history)

    with open(out / "final_debrief.txt", "w", encoding="utf-8") as f:
        f.write(debrief)

    print("\n" + "=" * 60)
    print("FINAL DEBRIEF")
    print("=" * 60)
    print(debrief)

    comp_df = pd.DataFrame(comparison_log)

    print("\n" + "=" * 60)
    print("STATIC (frozen) vs ADAPTIVE evasion rate per generation")
    print("(this is your headline closed-loop evidence chart)")
    print("=" * 60)
    if not comp_df.empty:
        print(
            comp_df[
                [
                    "generation",
                    "frozen_evasion_rate",
                    "adaptive_evasion_rate",
                    "adaptive_was_retrained_this_gen",
                ]
            ].to_string(index=False)
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend", choices=["gemini", "ollama", "custom"],
        default="gemini",
    )
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument(
        "--campaigns-per-generation", type=int, default=4
    )
    parser.add_argument(
        "--campaign-size", type=int, default=5,
        help="Number of transactions per individual campaign.",
    )
    parser.add_argument(
        "--retrain-every", type=int, default=5,
        help=(
            "Retrain adaptive_blue every N generations on all "
            "cumulative data. Smaller = more responsive defense "
            "but more compute; larger = cheaper but slower to "
            "adapt."
        ),
    )
    parser.add_argument("--customers", type=int, default=1500)
    parser.add_argument("--merchants", type=int, default=150)
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--output", default="output_v4")
    parser.add_argument(
        "--blue-model", default="output_v2/blue_model_baseline",
    )
    parser.add_argument(
        "--calibration",
        default="data/calibration/master_calibration.json",
    )
    parser.add_argument(
        "--baseline-transactions",
        default="data/blue_dataset_v2/transactions.parquet",
        help=(
            "Path to the realistic Phase-1-style baseline "
            "transactions.parquet (diverse, ~2%% fraud, all 13 "
            "families). Mixed into every retrain alongside "
            "red's campaign data - required, see comment in "
            "run() for why."
        ),
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--recalibrate-threshold",
        action="store_true",
        help=(
            "Recalibrate the decision threshold "
            "(target precision=0.5) on every retrain. Default "
            "is OFF (threshold stays fixed at the Phase 1 "
            "baseline value) because recalibrating was found to "
            "push the threshold monotonically upward every "
            "retrain in testing, mechanically suppressing "
            "recall regardless of what the model actually "
            "learned - see run_closed_loop.py docstring."
        ),
    )
    parser.add_argument(
        "--bandit-epsilon", type=float, default=0.15,
        help=(
            "Probability the knob-level bandit picks a "
            "uniformly random bin instead of its Thompson-"
            "sampled argmax. Forces param-value diversity within "
            "whichever family gets chosen, so retraining sees "
            "more than one narrow evasive param range per "
            "family."
        ),
    )
    parser.add_argument(
        "--family-exploration-rate", type=float, default=0.25,
        help=(
            "Probability that, for each individual campaign "
            "slot, red's own least-tried-family policy OVERRIDES "
            "the LLM strategist's family choice. The strategist "
            "alone is exploit-only ('double down on what "
            "worked'); this forces genuine family diversity into "
            "what blue retrains on. Found necessary in testing "
            "(Aug 2026) - without it, retraining on adversarially"
            "-selected examples alone produced flat-to-negative "
            "adaptive-vs-frozen comparisons even with the "
            "threshold-drift bug fixed."
        ),
    )

    args = parser.parse_args()

    run(
        backend=args.backend,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        n_generations=args.generations,
        campaigns_per_generation=args.campaigns_per_generation,
        campaign_size=args.campaign_size,
        retrain_every=args.retrain_every,
        n_customers=args.customers,
        n_merchants=args.merchants,
        n_days=args.days,
        output_dir=args.output,
        blue_model_path=args.blue_model,
        calibration_path=args.calibration,
        baseline_transactions_path=args.baseline_transactions,
        seed=args.seed,
        recalibrate_threshold=args.recalibrate_threshold,
        bandit_epsilon=args.bandit_epsilon,
        family_exploration_rate=args.family_exploration_rate,
    )


if __name__ == "__main__":
    main()
