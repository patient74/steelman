"""
run_baseline.py

Phase 1 deliverable: the static baseline.

Trains BlueModel once on a single simulator-generated batch
(default-param fraud - i.e. no red-team adversarial search
involved yet) using a train/test split, evaluates on held-out
data, and writes baseline_metrics.json + baseline_report.md.

This is the number every later adversarial-loop generation gets
compared against: "how much better/worse does blue do against
evolving red than it did against this static, non-adaptive
fraud population."
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from blue_model import BlueModel, build_graph_features


def main(
    transactions_path: str,
    output_dir: str = "output",
):

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {transactions_path} ...")
    df = pd.read_parquet(transactions_path)

    print(f"Loaded {len(df):,} transactions "
          f"({df['fraud_label'].mean():.4%} fraud rate)")

    # Graph features must be built over the FULL batch before
    # splitting - graph structure (device/merchant degree etc.)
    # is a property of the whole transaction population, not of
    # an individual row, and splitting first would silently
    # starve the graph of edges.
    print("Building graph features over full batch...")
    graph_feats = build_graph_features(df)

    train_df, test_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["fraud_label"],
    )

    print(
        f"Train: {len(train_df):,} rows "
        f"({train_df['fraud_label'].sum()} fraud) | "
        f"Test: {len(test_df):,} rows "
        f"({test_df['fraud_label'].sum()} fraud)"
    )

    model = BlueModel(decision_threshold=0.5)

    print("Fitting XGBoost...")
    model.fit(train_df, graph_features=graph_feats)

    print("Calibrating decision threshold "
          "(target precision = 0.5) on train set...")
    model.calibrate_threshold(
        train_df,
        target_precision=0.5,
        graph_features=graph_feats,
    )
    print(f"Calibrated threshold: {model.decision_threshold:.4f}")

    print("Evaluating on held-out test set...")
    result = model.evaluate(test_df, graph_features=graph_feats)

    print()
    print("=== BASELINE RESULTS (held-out test set) ===")
    print(f"Precision : {result.precision:.4f}")
    print(f"Recall    : {result.recall:.4f}")
    print(f"F1        : {result.f1:.4f}")
    print(f"AUC-PR    : {result.auc_pr:.4f}")
    print(f"AUC-ROC   : {result.auc_roc:.4f}")
    print(f"Threshold : {result.threshold:.4f}")
    print(f"Confusion matrix (rows=true, cols=pred) "
          f"[[TN,FP],[FN,TP]]:")
    print(f"  {result.confusion}")
    print()
    print("Per-family recall:")
    for family, stats in sorted(
        result.per_family.items(),
        key=lambda kv: kv[1]["recall"],
    ):
        print(
            f"  {family:28s} "
            f"n={stats['n']:4d}  "
            f"recall={stats['recall']:.4f}  "
            f"mean_prob={stats['mean_probability']:.4f}"
        )

    with open(
        out / "baseline_metrics.json", "w", encoding="utf-8"
    ) as f:
        json.dump(result.to_dict(), f, indent=2)

    importances = model.feature_importance(top_n=20)

    print()
    print("Top 20 feature importances:")
    print(importances.to_string())

    importances.to_json(
        out / "baseline_feature_importance.json", indent=2
    )

    model.save(out / "blue_model_baseline")

    # ---- markdown report ----

    lines = []
    lines.append("# Blue Model — Static Baseline Report\n")
    lines.append(
        f"Trained on {len(train_df):,} transactions, "
        f"evaluated on {len(test_df):,} held-out transactions "
        f"from the same simulator batch (default-parameter "
        f"fraud, no adversarial search).\n"
    )
    lines.append("## Overall metrics\n")
    lines.append(f"- **Precision**: {result.precision:.4f}")
    lines.append(f"- **Recall**: {result.recall:.4f}")
    lines.append(f"- **F1**: {result.f1:.4f}")
    lines.append(
        f"- **AUC-PR (average precision)**: {result.auc_pr:.4f}"
    )
    lines.append(f"- **AUC-ROC**: {result.auc_roc:.4f}")
    lines.append(
        f"- **Decision threshold** (calibrated for "
        f"precision >= 0.5): {result.threshold:.4f}\n"
    )
    lines.append("## Per-family recall (held-out)\n")
    lines.append("| Attack family | n (fraud) | Recall | Mean predicted probability |")
    lines.append("|---|---|---|---|")
    for family, stats in sorted(
        result.per_family.items(),
        key=lambda kv: kv[1]["recall"],
    ):
        lines.append(
            f"| {family} | {stats['n']} | "
            f"{stats['recall']:.4f} | "
            f"{stats['mean_probability']:.4f} |"
        )

    lines.append("\n## Top feature importances\n")
    lines.append("| Feature | Importance |")
    lines.append("|---|---|")
    for feat, val in importances.items():
        lines.append(f"| {feat} | {val:.4f} |")

    with open(
        out / "baseline_report.md", "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(lines))

    print()
    print(f"Saved: {out}/baseline_metrics.json")
    print(f"Saved: {out}/baseline_feature_importance.json")
    print(f"Saved: {out}/baseline_report.md")
    print(f"Saved: {out}/blue_model_baseline/ (model artifacts)")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transactions",
        default="data/blue_dataset_v2/transactions.parquet",
    )
    parser.add_argument("--output", default="output")

    args = parser.parse_args()

    main(args.transactions, args.output)
