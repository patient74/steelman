"""
run_adversarial_loop_llm.py

Phase 3 integration: strategist (LLM) + bandit (Phase 2) +
simulator + blue, combined into one generation loop.

*** THIS IS THE SCRIPT TO RUN ON YOUR OWN MACHINE. ***

Why: it needs a real LLM to be useful (the MockLLMClient tests
already confirmed the plumbing works - this script exercises it
against actual model outputs, which requires either:
  (a) a local Ollama server (recommended - free, open source,
      zero marginal cost per generation), or
  (b) a free-tier hosted OpenAI-compatible endpoint (Groq,
      OpenRouter free models, etc.)

Neither is available in this sandboxed session (no GPU, and no
way to run a persistent background server here), so this script
is handed off to you to run and report results back.

--------------------------------------------------------------
SETUP (option A - Gemini API, recommended for this project):
--------------------------------------------------------------
  Gemini has a generous free tier and exposes an OpenAI-
  compatible endpoint, so it plugs into the exact same
  OpenAICompatibleClient used for local/open models - no extra
  SDK or code path needed.

  1. Get a free API key: https://aistudio.google.com/apikey
  2. pip install openai pandas numpy xgboost shap networkx scikit-learn
  3. Run this script:
         python3 run_adversarial_loop_llm.py \\
             --backend gemini \\
             --api-key <your key> \\
             --model gemini-3.5-flash-lite \\
             --generations 50 \\
             --campaigns-per-generation 4

  Model choice matters here because of free-tier rate limits.
  Exact numbers per model change over time and Google doesn't
  publish per-model history - check
  https://ai.google.dev/gemini-api/docs/rate-limits for current
  values before a long run. Your Aug 2026 test run against
  gemini-3.5-flash-lite completed successfully at ~4.5s
  throttling between calls with no rate-limit errors across 20
  calls, so that throttle is a reasonable starting point, but
  watch stdout for "[rate limited, retrying...]" messages on a
  longer run and increase min_seconds_between_calls in
  OpenAICompatibleClient's construction below if you see them
  often.

  Honesty note for the write-up: Gemini's free tier is free to
  USE, but it is not open-weight/open-source like Llama or
  Qwen - the strategist's ARCHITECTURE is model-agnostic (any
  OpenAI-compatible endpoint), and swapping to a genuinely
  open-source local model is a one-flag change (--backend
  ollama, see option B), but the run you actually submit should
  describe Gemini accurately as "free-tier hosted," not
  "open source."

--------------------------------------------------------------
SETUP (option B - local Ollama, genuinely free + open source):
--------------------------------------------------------------
  1. Install Ollama: https://ollama.com/download
  2. In a terminal:  ollama pull llama3.1:8b
     (or a smaller/faster one: ollama pull qwen2.5:7b-instruct)
  3. In a terminal:  ollama serve
     (leave this running - it's the local model server)
  4. pip install openai pandas numpy xgboost shap networkx scikit-learn
  5. Run this script:
         python3 run_adversarial_loop_llm.py \\
             --backend ollama \\
             --model llama3.1:8b \\
             --generations 50 \\
             --campaigns-per-generation 4

--------------------------------------------------------------
SETUP (option C - other free hosted endpoint, e.g. Groq):
--------------------------------------------------------------
  1. Get a free API key from https://console.groq.com
  2. Run:
         python3 run_adversarial_loop_llm.py \\
             --backend custom \\
             --base-url https://api.groq.com/openai/v1 \\
             --api-key <your key> \\
             --model llama-3.1-8b-instant \\
             --generations 50 \\
             --campaigns-per-generation 4

--------------------------------------------------------------
WHAT TO PASTE BACK:
--------------------------------------------------------------
  - The full stdout of this script (it prints per-generation
    evasion rate, mean margin, and which family/families were
    chosen + why, every generation)
  - The contents of output_v3/generation_log.json
  - The contents of output_v3/strategist_reasoning_log.jsonl
  - The final debrief text printed at the end

That's everything needed to build the convergence charts and the
docx evidence in Phase 4/6 without needing to re-run anything.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # payment_environment_v24.py should sit alongside this script

import pandas as pd

from blue_model import BlueModel, build_graph_features
from red_bandit import RedBandit
from red_strategist import OpenAICompatibleClient, RedStrategist
from payment_environment_v24 import (
    ATTACK_PARAM_SCHEMA,
    AttackAction,
    PaymentEnvironmentV23 as PaymentEnvironment,
)


# Qualitative "intensity" hint from the strategist -> a soft
# nudge on which bin range the bandit should bias toward
# sampling from THIS round, without overriding the bandit's own
# learned posterior. Deliberately soft (only affects tie-breaks
# implicitly via which family/knob gets tried), so the bandit
# remains the source of truth for numeric convergence per our
# Phase 2/3 division of labor.
INTENSITY_HINT_BIN_RANGE = {
    "low": (0.0, 0.4),
    "medium": (0.3, 0.7),
    "high": (0.6, 1.0),
}


def run(
    backend: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
    n_generations: int,
    campaigns_per_generation: int,
    n_customers: int,
    n_merchants: int,
    n_days: int,
    output_dir: str,
    blue_model_path: str,
    calibration_path: str,
    seed: int,
):

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading pre-trained baseline blue model...")
    blue = BlueModel()
    blue.load(blue_model_path)
    print(f"Loaded. Decision threshold = {blue.decision_threshold:.4f}")

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

    red = RedBandit(ATTACK_PARAM_SCHEMA, n_bins=7, seed=seed)

    if backend == "ollama":
        llm = OpenAICompatibleClient(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model=model,
        )
    elif backend == "gemini":
        if not api_key:
            raise ValueError(
                "--api-key is required for --backend gemini "
                "(get a free key at "
                "https://aistudio.google.com/apikey)"
            )
        llm = OpenAICompatibleClient(
            base_url=(
                "https://generativelanguage.googleapis.com/"
                "v1beta/openai/"
            ),
            api_key=api_key,
            model=model,
            # gemini-3.5-flash-lite free tier is ~15 req/min;
            # stay comfortably under that proactively rather
            # than relying purely on retry-after-429, since
            # bursts still cost wall-clock time either way.
            min_seconds_between_calls=4.5,
        )
    elif backend == "custom":
        llm = OpenAICompatibleClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
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

    print(
        f"\nRunning {n_generations} generations, "
        f"{campaigns_per_generation} campaigns/generation, "
        f"backend={backend} model={model}\n"
    )

    for gen in range(n_generations):

        t0 = time.time()

        decision = strategist.decide(
            generation=gen, history=red.history
        )

        gen_records = []

        # Distribute campaigns_per_generation across whichever
        # families the strategist chose this round (round-robin
        # if it picked a combination).
        for i in range(campaigns_per_generation):

            family = decision.families[
                i % len(decision.families)
            ]

            params, bin_indices = red.propose(family)

            action = AttackAction(
                family=family,
                campaign_size=5,
                params=params,
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
                    "evasion_rate": 1
                    - (detected_count / result["actual_size"]),
                    "strategist_reasoning": decision.reasoning,
                    "strategist_parse_ok": decision.parse_ok,
                },
            )

            gen_records.append(red.history[-1])

        generation_log.extend(gen_records)

        # Checkpoint after every generation - a long rate-
        # limited run (Gemini free tier, or a slow local model)
        # can take a while; this way a Ctrl+C or a crash midway
        # still leaves usable partial results on disk instead of
        # losing the whole run.
        pd.DataFrame(generation_log).to_json(
            out / "generation_log.json", orient="records", indent=2
        )
        red.save(out / "bandit_final_state.json")

        elapsed = time.time() - t0

        if gen_records:
            mean_evasion = sum(
                r["evasion_rate"] for r in gen_records
            ) / len(gen_records)
            mean_margin_g = sum(
                r["detection_margin"] for r in gen_records
            ) / len(gen_records)

            print(
                f"Gen {gen:3d} ({elapsed:5.1f}s) | "
                f"families={decision.families} | "
                f"parse_ok={decision.parse_ok} | "
                f"mean_evasion={mean_evasion:.3f} | "
                f"mean_margin={mean_margin_g:+.3f}"
            )
            print(f"           reasoning: {decision.reasoning[:150]}")
        else:
            print(
                f"Gen {gen:3d} ({elapsed:5.1f}s) | "
                f"no campaigns landed (no eligible target?)"
            )

    log_df = pd.DataFrame(generation_log)

    print(f"\nFinal save: {out}/generation_log.json "
          f"({len(log_df)} trials)")
    print(f"Final save: {out}/bandit_final_state.json")

    print("\nGenerating end-of-run debrief...")
    debrief = strategist.generate_debrief(red.history)

    with open(
        out / "final_debrief.txt", "w", encoding="utf-8"
    ) as f:
        f.write(debrief)

    print("\n" + "=" * 60)
    print("FINAL DEBRIEF")
    print("=" * 60)
    print(debrief)

    print("\n" + "=" * 60)
    print("Generation-level mean evasion rate:")
    print("=" * 60)
    if not log_df.empty:
        print(
            log_df.groupby("generation")["evasion_rate"]
            .mean()
            .to_string()
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        choices=["gemini", "ollama", "custom"],
        default="gemini",
    )
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument(
        "--campaigns-per-generation", type=int, default=4
    )
    parser.add_argument("--customers", type=int, default=1500)
    parser.add_argument("--merchants", type=int, default=150)
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--output", default="output_v3")
    parser.add_argument(
        "--blue-model",
        default="output_v2/blue_model_baseline",
    )
    parser.add_argument(
        "--calibration",
        default="data/calibration/master_calibration.json",
    )
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

    run(
        backend=args.backend,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        n_generations=args.generations,
        campaigns_per_generation=args.campaigns_per_generation,
        n_customers=args.customers,
        n_merchants=args.merchants,
        n_days=args.days,
        output_dir=args.output,
        blue_model_path=args.blue_model,
        calibration_path=args.calibration,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
