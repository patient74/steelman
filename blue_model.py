"""
blue_model.py
=============

Phase 1 of the Mastercard Innovation Challenge 2026 submission:
the "Defend" pillar.

This module is intentionally decoupled from the red-team loop.
It only knows how to:
  1. turn raw simulator transactions into an observable feature
     matrix (no ground-truth leakage),
  2. fit / retrain a classifier on that feature matrix, and
  3. score new transactions, returning both a probability and a
     `detection_margin` (probability - decision_threshold) that
     the red-team loop (Phase 2-4) will consume as its reward
     signal.

Design choices, and why:
  - XGBoost, not deep learning: fast to retrain every generation
    (seconds, not minutes), strong on tabular data, gives
    feature_importances_/SHAP for the "novelty" and "detection
    efficacy" judging criteria for free.
  - Graph features via networkx: RELATIONAL_CAMOUFLAGE (A6) is
    deliberately built to be invisible in any single
    transaction's own fields - it can only be caught by looking
    at shared-entity structure across transactions. Without a
    graph feature, blue has *no* mechanism to ever catch it,
    which would make that whole attack family undetectable by
    construction and quietly invalidate a chunk of the
    "detection algorithm efficacy" evaluation.
  - AUC-PR (average precision), not AUC-ROC, as the headline
    metric: fraud rate here is ~2%, and ROC-AUC is well known to
    look artificially strong under heavy class imbalance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# LEAKAGE BOUNDARY
# ============================================================
#
# These are the ONLY columns blue is allowed to see. Every
# ground-truth / attack-metadata column produced by the
# simulator (fraud_label, attack_id, attack_family,
# attack_sequence, is_campaign_transaction, ground_truth_marker)
# is explicitly excluded here, in one place, so the boundary is
# auditable rather than implicit in whatever code happens to
# not-reference a column.

GROUND_TRUTH_COLUMNS = [
    "fraud_label",
    "attack_id",
    "attack_family",
    "attack_sequence",
    "is_campaign_transaction",
    "ground_truth_marker",
]

# Identifier columns: useful as join keys / for graph features,
# but must not be fed to the model directly as raw high-cardinality
# categoricals (that's target leakage via memorization, not signal).
ID_COLUMNS = [
    "transaction_id",
    "customer_id",
    "account_id",
    "card_id",
    "token_id",
    "device_id",
    "merchant_id",
    "issuer_id",
    "acquirer_id",
    "initiating_agent_id",
    "timestamp",
]

CATEGORICAL_FEATURES = [
    "currency",
    "customer_country",
    "customer_city",
    "merchant_country",
    "merchant_city",
    "channel",
    "mcc",
    "authentication_method",
    "authentication_result",
    "merchant_risk_tier",
]

NUMERIC_FEATURES = [
    "amount",
    "customer_txn_count_1h",
    "customer_txn_count_24h",
    "customer_amount_1h",
    "customer_amount_24h",
    "merchant_txn_count",
    "card_txn_count",
    "customer_total_txn",
    "device_customer_count_24h",
    "geo_distance_score",
    "device_trusted",
    "device_os_risk_score",
    "agent_scope_conformance_score",
    "agent_identity_confidence",
]

GRAPH_FEATURES = [
    "graph_device_degree",
    "graph_merchant_degree",
    "graph_shared_device_customers",
    "graph_customer_merchant_diversity",
    "graph_component_size",
    "graph_clustering_coeff",
]


def build_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a customer-device-merchant multigraph from the given
    transaction batch and derive per-transaction structural
    features from it.

    This is the ONLY mechanism in the whole detection pipeline
    that can catch RELATIONAL_CAMOUFLAGE (A6): that family is
    built specifically so no single transaction's own fields
    look anomalous, and the only signal is "this device/merchant
    touches an unusually large or unusually interconnected set
    of entities."

    Returned features are attached per transaction_id.

    Implementation note: uses vectorized pandas groupby instead
    of iterrows() for every quantity that can be computed as a
    simple aggregate (degree, shared-entity counts). networkx is
    used only for the two quantities that genuinely need graph
    structure (connected component size, clustering
    coefficient), built once over deduplicated
    (customer, device, merchant) edges rather than one edge per
    transaction row - a batch with repeat customer/device/
    merchant combinations does not need repeated edges, and
    iterating those duplicates was the main cost.
    """

    cust_col = df["customer_id"].values
    dev_col = df["device_id"].values
    merch_col = df["merchant_id"].values

    # ---- vectorized aggregate features ----

    device_customer_counts = (
        df.groupby("device_id")["customer_id"]
        .nunique()
    )

    customer_merchant_counts = (
        df.groupby("customer_id")["merchant_id"]
        .nunique()
    )

    device_degree = df.groupby("device_id").size()
    merchant_degree = df.groupby("merchant_id").size()

    # ---- graph-structural features (networkx, deduplicated edges) ----

    g = nx.Graph()

    unique_triples = (
        df[["customer_id", "device_id", "merchant_id"]]
        .drop_duplicates()
    )

    for cust, dev, merch in unique_triples.itertuples(
        index=False
    ):

        g.add_edge(("customer", cust), ("device", dev))
        g.add_edge(("customer", cust), ("merchant", merch))

    component_size = {}
    for comp in nx.connected_components(g):
        size = len(comp)
        for node in comp:
            component_size[node] = size

    clustering = nx.clustering(g)

    cust_component = pd.Series(
        {
            cust: component_size.get(("customer", cust), 1)
            for cust in df["customer_id"].unique()
        }
    )

    cust_clustering = pd.Series(
        {
            cust: clustering.get(("customer", cust), 0.0)
            for cust in df["customer_id"].unique()
        }
    )

    # ---- assemble per-row (vectorized joins, no iterrows) ----

    out = pd.DataFrame(
        {
            "transaction_id": df["transaction_id"].values,
            "graph_device_degree": df["device_id"]
            .map(device_degree)
            .values,
            "graph_merchant_degree": df["merchant_id"]
            .map(merchant_degree)
            .values,
            "graph_shared_device_customers": df["device_id"]
            .map(device_customer_counts)
            .values,
            "graph_customer_merchant_diversity": df["customer_id"]
            .map(customer_merchant_counts)
            .values,
            "graph_component_size": df["customer_id"]
            .map(cust_component)
            .values,
            "graph_clustering_coeff": df["customer_id"]
            .map(cust_clustering)
            .values,
        }
    )

    return out.set_index("transaction_id")


def build_feature_matrix(
    df: pd.DataFrame,
    graph_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Turn raw simulator transactions into a leakage-free feature
    matrix. If graph_features is None, they are computed fresh
    from df (this recomputes the graph over exactly the rows
    passed in, which is the correct behaviour for both training
    and scoring a specific batch).
    """

    missing_gt = [
        c for c in GROUND_TRUTH_COLUMNS if c not in df.columns
    ]
    if missing_gt:
        raise ValueError(
            f"Input is missing expected ground-truth columns "
            f"{missing_gt} - are you sure this is raw simulator "
            f"output?"
        )

    if graph_features is None:
        graph_features = build_graph_features(df)

    feat = df[
        ["transaction_id"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ].set_index("transaction_id")

    feat = feat.join(graph_features, how="left")

    feat["device_trusted"] = feat["device_trusted"].astype(int)

    feat = pd.get_dummies(
        feat,
        columns=CATEGORICAL_FEATURES,
        dummy_na=True,
    )

    return feat


@dataclass
class EvalResult:

    precision: float
    recall: float
    f1: float
    auc_pr: float
    auc_roc: float
    threshold: float
    confusion: list
    per_family: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc_pr": self.auc_pr,
            "auc_roc": self.auc_roc,
            "threshold": self.threshold,
            "confusion_matrix": self.confusion,
            "per_family": self.per_family,
        }


class BlueModel:
    """
    Blue-team detector. Stateful: retains its trained booster and
    feature-column schema across retrains so that scoring stays
    consistent with whatever generation last called .fit().
    """

    def __init__(self, decision_threshold: float = 0.5):

        self.model: xgb.XGBClassifier | None = None
        self.feature_columns: list[str] | None = None
        self.decision_threshold = decision_threshold

    # --------------------------------------------------------

    def _align_columns(self, feat: pd.DataFrame) -> pd.DataFrame:

        if self.feature_columns is None:
            return feat

        for col in self.feature_columns:
            if col not in feat.columns:
                feat[col] = 0

        extra = [
            c for c in feat.columns if c not in self.feature_columns
        ]

        feat = feat.drop(columns=extra)

        return feat[self.feature_columns]

    # --------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        graph_features: pd.DataFrame | None = None,
    ):
        """
        Fit (or fully refit) the model on the given labeled
        transaction batch. This is the "fresh retrain each
        generation" strategy we scoped: simpler and more
        experimentally defensible than incremental/warm-start
        updates, at the cost of discarding earlier generations'
        data unless the caller explicitly concatenates history
        into df before calling this.
        """

        feat = build_feature_matrix(df, graph_features)
        y = df.set_index("transaction_id").loc[
            feat.index, "fraud_label"
        ]

        self.feature_columns = feat.columns.tolist()

        # scale_pos_weight compensates for ~2% fraud prevalence -
        # without it xgboost tends to just predict the majority
        # class under this level of imbalance.
        n_pos = int(y.sum())
        n_neg = int((y == 0).sum())
        scale_pos_weight = (
            n_neg / n_pos if n_pos > 0 else 1.0
        )

        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=42,
        )

        self.model.fit(feat, y)

        return self

    # --------------------------------------------------------

    def score(
        self,
        df: pd.DataFrame,
        graph_features: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Score a batch of transactions. Returns a DataFrame
        indexed by transaction_id with `fraud_probability` and
        `detection_margin` (probability - decision_threshold).

        detection_margin is the signal the red-team loop
        consumes as its reward: continuous even when the
        transaction wasn't flagged, which is what makes
        bandit-style parameter search over evasion possible.
        """

        if self.model is None:
            raise RuntimeError(
                "BlueModel.score() called before .fit()"
            )

        feat = build_feature_matrix(df, graph_features)
        feat = self._align_columns(feat)

        proba = self.model.predict_proba(feat)[:, 1]

        out = pd.DataFrame(
            {
                "transaction_id": feat.index,
                "fraud_probability": proba,
                "detection_margin": proba
                - self.decision_threshold,
                "flagged": (
                    proba >= self.decision_threshold
                ).astype(int),
            }
        ).set_index("transaction_id")

        return out

    # --------------------------------------------------------

    def evaluate(
        self,
        df: pd.DataFrame,
        graph_features: pd.DataFrame | None = None,
    ) -> EvalResult:
        """
        Full evaluation against ground truth. This is the ONLY
        method in this class allowed to touch fraud_label /
        attack_family - everything above this line in the
        pipeline (fit, score) is leakage-free by construction.
        """

        scores = self.score(df, graph_features)

        truth = df.set_index("transaction_id").loc[
            scores.index
        ]

        y_true = truth["fraud_label"].values
        y_pred = scores["flagged"].values
        y_proba = scores["fraud_probability"].values

        precision = precision_score(
            y_true, y_pred, zero_division=0
        )
        recall = recall_score(
            y_true, y_pred, zero_division=0
        )
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc_pr = average_precision_score(y_true, y_proba)

        try:
            auc_roc = roc_auc_score(y_true, y_proba)
        except ValueError:
            auc_roc = float("nan")

        cm = confusion_matrix(y_true, y_pred).tolist()

        per_family = {}

        for family in truth["attack_family"].dropna().unique():

            family_mask = (
                truth["attack_family"] == family
            ).values

            # Detection efficacy PER FAMILY: of the fraud rows
            # belonging to this family, what fraction did blue
            # flag? (Recall-only, since precision isn't
            # meaningful sliced by attack family - false
            # positives aren't attributable to a fraud family.)
            fam_recall = recall_score(
                y_true[family_mask],
                y_pred[family_mask],
                zero_division=0,
            )

            per_family[family] = {
                "n": int(family_mask.sum()),
                "recall": float(fam_recall),
                "mean_probability": float(
                    y_proba[family_mask].mean()
                )
                if family_mask.sum() > 0
                else 0.0,
            }

        return EvalResult(
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            auc_pr=float(auc_pr),
            auc_roc=float(auc_roc),
            threshold=self.decision_threshold,
            confusion=cm,
            per_family=per_family,
        )

    # --------------------------------------------------------

    def feature_importance(self, top_n: int = 20) -> pd.Series:

        if self.model is None:
            raise RuntimeError(
                "BlueModel.feature_importance() called before "
                ".fit()"
            )

        importances = pd.Series(
            self.model.feature_importances_,
            index=self.feature_columns,
        ).sort_values(ascending=False)

        return importances.head(top_n)

    # --------------------------------------------------------

    def calibrate_threshold(
        self,
        df: pd.DataFrame,
        target_precision: float = 0.5,
        graph_features: pd.DataFrame | None = None,
    ) -> float:
        """
        Pick a decision threshold on a labeled batch such that
        precision is at least target_precision, choosing the
        threshold that maximizes recall subject to that
        constraint. Useful because with ~2% base fraud rate, a
        naive 0.5 threshold is not necessarily where you want to
        operate - this makes the false-positive/recall trade-off
        an explicit, reportable choice rather than an accident of
        XGBoost's default cutoff.
        """

        if self.model is None:
            raise RuntimeError(
                "BlueModel.calibrate_threshold() called before "
                ".fit()"
            )

        feat = build_feature_matrix(df, graph_features)
        feat = self._align_columns(feat)

        y_true = df.set_index("transaction_id").loc[
            feat.index, "fraud_label"
        ].values

        proba = self.model.predict_proba(feat)[:, 1]

        precisions, recalls, thresholds = (
            precision_recall_curve(y_true, proba)
        )

        # precision_recall_curve returns len(thresholds) + 1
        # precision/recall points; drop the last (threshold=inf)
        # point to align arrays.
        precisions = precisions[:-1]
        recalls = recalls[:-1]

        eligible = np.where(
            precisions >= target_precision
        )[0]

        if len(eligible) == 0:
            # Target precision unreachable at any threshold -
            # fall back to the threshold with best F1 instead of
            # silently returning something meaningless.
            f1s = (
                2
                * precisions
                * recalls
                / np.clip(precisions + recalls, 1e-9, None)
            )
            best_idx = int(np.argmax(f1s))
        else:
            best_idx = eligible[
                int(np.argmax(recalls[eligible]))
            ]

        self.decision_threshold = float(
            thresholds[best_idx]
        )

        return self.decision_threshold

    # --------------------------------------------------------

    def save(self, path: str | Path):

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save_model(
            str(path / "xgb_model.json")
        )

        with open(
            path / "meta.json", "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "feature_columns": self.feature_columns,
                    "decision_threshold": self.decision_threshold,
                },
                f,
                indent=2,
            )

    # --------------------------------------------------------

    def load(self, path: str | Path):

        path = Path(path)

        self.model = xgb.XGBClassifier()
        self.model.load_model(
            str(path / "xgb_model.json")
        )

        with open(
            path / "meta.json", "r", encoding="utf-8"
        ) as f:
            meta = json.load(f)

        self.feature_columns = meta["feature_columns"]
        self.decision_threshold = meta["decision_threshold"]

        return self
