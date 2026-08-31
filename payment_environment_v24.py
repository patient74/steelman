"""
Payment Environment V2.3
========================

Mastercard Innovation Challenge 2026
AI Defense Lab for Payment Security

Purpose
-------
Generate a synthetic, Mastercard-inspired payment ecosystem
calibrated against public datasets such as:

    - IEEE-CIS Fraud Detection
    - PaySim

V2.3 DESIGN
-----------
This version introduces a controlled fraud-injection architecture.

Key principles:

1. Normal transactions are generated first.
2. A hard fraud budget is calculated from the target fraud rate.
3. Fraud campaigns consume that budget.
4. Non-campaign fraud consumes the remaining budget.
5. Attack-family prevalence is explicitly controlled.
6. Campaigns create persistent multi-transaction patterns.
7. Legitimate transactions remain capable of looking suspicious.
8. Ground truth is separated from observable transaction features.
9. Validation fails if the fraud-rate tolerance is violated.

IMPORTANT
---------
This is NOT Mastercard proprietary data.

It does not attempt to reproduce Mastercard internal systems,
models, rules, infrastructure, or confidential data.

Public datasets are used only to calibrate statistical properties.

Generated data is completely synthetic and intended for:

    1. baseline fraud modelling
    2. red-team attack simulation
    3. blue-team detection
    4. adversarial evaluation
    5. synthetic data generation
    6. graph / campaign detection
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# RED / BLUE ADVERSARIAL-LOOP INTERFACES
# ============================================================
#
# These two dataclasses are the entire contract between an
# external red agent, this simulator, and an external blue
# detector. Nothing about the adversarial LOOP itself lives in
# this file — this file only needs to be able to (a) execute an
# AttackAction and (b) accept an AttackOutcome back so it can be
# logged. The search/learning logic is intentionally external.


@dataclass
class AttackAction:
    """
    One action chosen by the red agent for a single campaign.

    params values are expected to be normalized to [0, 1]. The
    simulator rescales each param internally according to
    ATTACK_PARAM_SCHEMA; the red agent does not need to know
    real-world units (currency, seconds, etc).
    """

    family: str
    campaign_size: int = 4
    params: dict = field(default_factory=dict)
    target_id: str | None = None  # None => simulator chooses a target

    def get(self, name, default=0.5):
        return float(self.params.get(name, default))


@dataclass
class AttackOutcome:
    """
    Per-campaign result reported back to the red agent after
    blue has scored the generated transactions. detection_margin
    is what makes bandit-style search over params possible: it
    is a continuous signal even when detected is False.
    """

    attack_id: str
    family: str
    params: dict
    transaction_ids: list
    detected_count: int = 0
    total_count: int = 0
    mean_detection_score: float = 0.0
    mean_detection_margin: float = 0.0

    @property
    def evasion_rate(self):
        if self.total_count == 0:
            return 0.0
        return 1.0 - (self.detected_count / self.total_count)


# ============================================================
# CONFIGURATION
# ============================================================

VERSION = "2.3"

DEFAULT_SEED = 42
DEFAULT_DAYS = 30
DEFAULT_CUSTOMERS = 10_000
DEFAULT_MERCHANTS = 1_000

DEFAULT_TARGET_FRAUD_RATE = 0.0025

# Fraud-rate validation tolerance.
# 0.25% target means approximately 0.20%-0.30% is accepted.
DEFAULT_FRAUD_TOLERANCE = 0.0005

DEFAULT_OUTPUT = Path(
    "data/simulation_v23"
)

SIMULATION_START = "2026-01-01"


# ============================================================
# ENUMERATIONS
# ============================================================

COUNTRIES = [
    "IN",
    "US",
    "GB",
    "SG",
    "AE",
    "AU",
]

COUNTRY_WEIGHTS = [
    0.72,
    0.08,
    0.06,
    0.04,
    0.05,
    0.05,
]


INDIA_CITIES = [
    "Mumbai",
    "Pune",
    "Bengaluru",
    "Delhi",
    "Hyderabad",
    "Chennai",
    "Kolkata",
    "Ahmedabad",
    "Jaipur",
    "Surat",
]

IN_CITY_WEIGHTS = [
    0.18,
    0.10,
    0.15,
    0.14,
    0.11,
    0.09,
    0.08,
    0.06,
    0.05,
    0.04,
]


PERSONAS = [
    "everyday_consumer",
    "digital_shopper",
    "business_professional",
    "high_value_consumer",
    "traveller",
    "subscription_user",
]

PERSONA_WEIGHTS = [
    0.36,
    0.28,
    0.14,
    0.06,
    0.08,
    0.08,
]


MCCS = [
    "GROCERY",
    "RETAIL",
    "RESTAURANT",
    "FUEL",
    "FASHION",
    "ELECTRONICS",
    "TRAVEL",
    "HOTEL",
    "AIRLINE",
    "HEALTHCARE",
    "UTILITIES",
]

MCC_WEIGHTS = [
    0.15,
    0.18,
    0.15,
    0.08,
    0.09,
    0.10,
    0.08,
    0.04,
    0.025,
    0.04,
    0.055,
]


CHANNELS = [
    "POS",
    "CONTACTLESS",
    "E_COMMERCE",
    "IN_APP",
    "RECURRING",
]

CHANNEL_WEIGHTS = [
    0.28,
    0.14,
    0.40,
    0.10,
    0.08,
]


AUTH_METHODS = [
    "PIN_LIKE",
    "CONTACTLESS_AUTH",
    "3DS_LIKE",
    "OTP_LIKE",
    "TOKENIZED",
    "BIOMETRIC_LIKE",
]


CARD_PRODUCTS = [
    "DEBIT",
    "CREDIT",
    "PREPAID",
]

CARD_PRODUCT_WEIGHTS = [
    0.55,
    0.40,
    0.05,
]


FRAUD_FAMILIES = [
    "CARD_TESTING",
    "ACCOUNT_TAKEOVER",
    "DEVICE_COMPROMISE",
    "VELOCITY_ABUSE",
    "UNUSUAL_GEO",
    "MERCHANT_ANOMALY",
    # ---- Mastercard Innovation Challenge 2026 additions ----
    "BEHAVIORAL_MIMICRY",       # A1
    "TEMPORAL_MIMICRY",         # A3
    "AMOUNT_MIMICRY",           # A4
    "ADAPTIVE_CARD_TESTING",    # A9 (parameterized variant of CARD_TESTING)
    "RELATIONAL_CAMOUFLAGE",    # A6
    "SYNTHETIC_IDENTITY",       # A11
    "AGENT_IDENTITY_SPOOFING",  # A30
    "AGENT_SCOPE_ABUSE",        # A31
]


# Explicit attack mix.
#
# This is intentionally configurable.
#
# The weights represent the intended distribution of FRAUD
# TRANSACTIONS across attack families, not the distribution
# of all transactions.
#
# NOTE: SYNTHETIC_IDENTITY is deliberately excluded from this
# table. It is not injected as a campaign against an existing
# entity; it is a customer-generation-time attack (see
# generate_customers) and is controlled by
# SYNTHETIC_IDENTITY_RATE instead.

ATTACK_FAMILY_WEIGHTS = {
    "CARD_TESTING": 0.10,
    "ACCOUNT_TAKEOVER": 0.10,
    "DEVICE_COMPROMISE": 0.10,
    "VELOCITY_ABUSE": 0.10,
    "UNUSUAL_GEO": 0.10,
    "MERCHANT_ANOMALY": 0.10,
    "BEHAVIORAL_MIMICRY": 0.12,
    "TEMPORAL_MIMICRY": 0.06,
    "AMOUNT_MIMICRY": 0.06,
    "ADAPTIVE_CARD_TESTING": 0.08,
    "RELATIONAL_CAMOUFLAGE": 0.10,
    "AGENT_IDENTITY_SPOOFING": 0.04,
    "AGENT_SCOPE_ABUSE": 0.04,
}


# ============================================================
# RED-AGENT PARAMETER SCHEMA
# ============================================================
#
# Every attack family exposes a small vector of CONTINUOUS,
# [0, 1]-normalized knobs. A red agent (bandit / evolutionary /
# RL) searches over this space. The simulator is responsible
# for rescaling each knob into a real-world quantity; the red
# agent never needs to know the real units.
#
# This is what makes the loop "adversarial" rather than a
# fixed set of canned attacks: the SAME family can look very
# different depending on the parameters chosen, and detection
# feedback (see AttackOutcome) is tied to those specific
# parameter values so the red agent can learn.

ATTACK_PARAM_SCHEMA = {

    "CARD_TESTING": {},
    "ACCOUNT_TAKEOVER": {},
    "DEVICE_COMPROMISE": {},
    "VELOCITY_ABUSE": {},
    "UNUSUAL_GEO": {},
    "MERCHANT_ANOMALY": {},

    "BEHAVIORAL_MIMICRY": {
        # 0 = generic/global fraud distribution, 1 = perfectly
        # mimics the victim's own historical behaviour across
        # amount, timing, and merchant/MCC choice.
        "mimicry_blend": (0.0, 1.0),
        # How far, within the mimicked distribution, the fraud
        # is pushed toward the tail (larger deviation = more
        # profitable for the attacker, more detectable).
        "amount_deviation": (0.0, 1.0),
    },

    "TEMPORAL_MIMICRY": {
        "mimicry_blend": (0.0, 1.0),
    },

    "AMOUNT_MIMICRY": {
        "mimicry_blend": (0.0, 1.0),
        "amount_deviation": (0.0, 1.0),
    },

    "ADAPTIVE_CARD_TESTING": {
        # How aggressively amounts grow after an approval.
        "escalation_rate": (0.0, 1.0),
        # How much a decline shrinks the next attempt / shifts
        # merchant or MCC.
        "retreat_sensitivity": (0.0, 1.0),
    },

    "RELATIONAL_CAMOUFLAGE": {
        # 0 = fraud concentrated on one entity (easy to catch
        # via simple thresholds), 1 = fraud thinly spread across
        # many entities in the graph (needs graph features).
        "spread_density": (0.0, 1.0),
    },

    "AGENT_IDENTITY_SPOOFING": {
        # How closely the spoofed agent mimics the real agent's
        # historical behavioural signature. 1 = near-perfect
        # impersonation.
        "spoofing_similarity": (0.0, 1.0),
    },

    "AGENT_SCOPE_ABUSE": {
        # How far outside the agent's declared scope (spend
        # limit / allowed MCCs / allowed merchants) the
        # transaction strays.
        "scope_violation_depth": (0.0, 1.0),
        # 0 = single large violation, 1 = slow gradual creep
        # across the campaign.
        "escalation_pace": (0.0, 1.0),
    },
}


DEFAULT_RED_PARAMS = {
    "mimicry_blend": 0.5,
    "amount_deviation": 0.5,
    "escalation_rate": 0.5,
    "retreat_sensitivity": 0.5,
    "spread_density": 0.5,
    "spoofing_similarity": 0.5,
    "scope_violation_depth": 0.5,
    "escalation_pace": 0.5,
}


# Fraction of (non-campaign-eligible) customers generated as
# synthetic identities (A11). Applied at generate_customers time.

SYNTHETIC_IDENTITY_RATE = 0.015


TOKEN_TYPES = [
    "DEVICE_TOKEN",
    "MERCHANT_TOKEN",
    "NETWORK_TOKEN",
]


# ============================================================
# PERSONA PARAMETERS
# ============================================================

PERSONA_CONFIG = {

    "everyday_consumer": {
        "daily_transactions": 0.65,
        "amount_multiplier": 0.70,
        "devices": (1, 2),
        "cards": (1, 2),
    },

    "digital_shopper": {
        "daily_transactions": 1.00,
        "amount_multiplier": 0.90,
        "devices": (1, 3),
        "cards": (1, 3),
    },

    "business_professional": {
        "daily_transactions": 1.25,
        "amount_multiplier": 1.25,
        "devices": (1, 3),
        "cards": (1, 3),
    },

    "high_value_consumer": {
        "daily_transactions": 1.10,
        "amount_multiplier": 2.60,
        "devices": (1, 3),
        "cards": (2, 4),
    },

    "traveller": {
        "daily_transactions": 0.80,
        "amount_multiplier": 1.20,
        "devices": (1, 4),
        "cards": (1, 3),
    },

    "subscription_user": {
        "daily_transactions": 0.80,
        "amount_multiplier": 0.65,
        "devices": (1, 2),
        "cards": (1, 2),
    },
}


# ============================================================
# CAMPAIGN CONFIGURATION
# ============================================================

CAMPAIGN_CONFIG = {

    "CARD_TESTING": {
        "min_size": 2,
        "max_size": 7,
        "target_type": "card",
    },

    "ACCOUNT_TAKEOVER": {
        "min_size": 3,
        "max_size": 10,
        "target_type": "customer",
    },

    "DEVICE_COMPROMISE": {
        "min_size": 3,
        "max_size": 10,
        "target_type": "device",
    },

    "VELOCITY_ABUSE": {
        "min_size": 4,
        "max_size": 12,
        "target_type": "customer",
    },

    "UNUSUAL_GEO": {
        "min_size": 2,
        "max_size": 8,
        "target_type": "customer",
    },

    "MERCHANT_ANOMALY": {
        "min_size": 4,
        "max_size": 15,
        "target_type": "merchant",
    },

    "BEHAVIORAL_MIMICRY": {
        "min_size": 2,
        "max_size": 8,
        "target_type": "customer",
    },

    "TEMPORAL_MIMICRY": {
        "min_size": 2,
        "max_size": 6,
        "target_type": "customer",
    },

    "AMOUNT_MIMICRY": {
        "min_size": 2,
        "max_size": 6,
        "target_type": "customer",
    },

    "ADAPTIVE_CARD_TESTING": {
        "min_size": 3,
        "max_size": 10,
        "target_type": "card",
    },

    "RELATIONAL_CAMOUFLAGE": {
        "min_size": 4,
        "max_size": 15,
        "target_type": "merchant",
    },

    "AGENT_IDENTITY_SPOOFING": {
        "min_size": 2,
        "max_size": 6,
        "target_type": "agent",
    },

    "AGENT_SCOPE_ABUSE": {
        "min_size": 2,
        "max_size": 8,
        "target_type": "agent",
    },
}


# Percentage of fraud budget that should normally be represented
# by multi-transaction campaigns.
#
# The remaining fraud becomes isolated fraud.

CAMPAIGN_FRACTION = 0.65


# ============================================================
# UTILITIES
# ============================================================

def weighted_choice(
    rng,
    values,
    weights,
):
    weights = np.asarray(
        weights,
        dtype=float,
    )

    weights = weights / weights.sum()

    return rng.choice(
        values,
        p=weights,
    )


def clamp(
    value,
    low,
    high,
):
    return max(
        low,
        min(
            high,
            value,
        ),
    )


def sigmoid(x):

    x = clamp(
        x,
        -50,
        50,
    )

    return 1.0 / (
        1.0 + math.exp(-x)
    )


def safe_json(obj):

    if isinstance(obj, dict):

        return {
            str(k): safe_json(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):

        return [
            safe_json(x)
            for x in obj
        ]

    if isinstance(
        obj,
        (
            np.integer,
        ),
    ):

        return int(obj)

    if isinstance(
        obj,
        (
            np.floating,
        ),
    ):

        return float(obj)

    if isinstance(
        obj,
        (
            np.bool_,
            bool,
        ),
    ):

        return bool(obj)

    if isinstance(
        obj,
        pd.Timestamp,
    ):

        return obj.isoformat()

    return obj


def choose_timestamp(
    rng,
    day_start,
):

    seconds = int(
        rng.integers(
            0,
            86400,
        )
    )

    return (
        day_start
        + pd.Timedelta(
            seconds=seconds
        )
    )


# ============================================================
# CALIBRATION
# ============================================================

class Calibration:

    def __init__(
        self,
        path,
    ):

        self.path = Path(path)

        if not self.path.exists():

            raise FileNotFoundError(
                f"Calibration file not found: "
                f"{self.path}"
            )

        with open(
            self.path,
            "r",
            encoding="utf-8",
        ) as f:

            self.data = json.load(f)

        self.ieee = (
            self.data
            .get("datasets", {})
            .get("IEEE-CIS", {})
        )

        self.paysim = (
            self.data
            .get("datasets", {})
            .get("PaySim", {})
        )

    # --------------------------------------------------------

    def get_ieee_amount_quantiles(
        self,
    ):

        try:

            return (
                self.ieee
                ["amounts"]
                ["overall"]
                ["quantiles"]
            )

        except KeyError:

            return {
                "0.001": 3.61,
                "0.25": 43.3,
                "0.50": 68.8,
                "0.75": 125.0,
                "0.90": 275.0,
                "0.95": 445.0,
                "0.99": 1104.0,
                "0.999": 2770.0,
            }

    # --------------------------------------------------------

    def sample_amount(
        self,
        rng,
        persona_multiplier=1.0,
    ):

        q = (
            self.get_ieee_amount_quantiles()
        )

        u = rng.random()

        if u < 0.25:

            lo = float(
                q["0.001"]
            )

            hi = float(
                q["0.25"]
            )

        elif u < 0.50:

            lo = float(
                q["0.25"]
            )

            hi = float(
                q["0.50"]
            )

        elif u < 0.75:

            lo = float(
                q["0.50"]
            )

            hi = float(
                q["0.75"]
            )

        elif u < 0.90:

            lo = float(
                q["0.75"]
            )

            hi = float(
                q["0.90"]
            )

        elif u < 0.95:

            lo = float(
                q["0.90"]
            )

            hi = float(
                q["0.95"]
            )

        elif u < 0.99:

            lo = float(
                q["0.95"]
            )

            hi = float(
                q["0.99"]
            )

        else:

            lo = float(
                q["0.99"]
            )

            hi = float(
                q["0.999"]
            )

        lo = max(
            lo,
            0.50,
        )

        hi = max(
            hi,
            lo + 0.01,
        )

        value = math.exp(
            rng.uniform(
                math.log(lo),
                math.log(hi),
            )
        )

        value *= persona_multiplier

        return round(
            clamp(
                value,
                1.0,
                250_000.0,
            ),
            2,
        )


# ============================================================
# PAYMENT ENVIRONMENT
# ============================================================

class PaymentEnvironmentV23:

    def __init__(
        self,
        customers=DEFAULT_CUSTOMERS,
        merchants=DEFAULT_MERCHANTS,
        days=DEFAULT_DAYS,
        seed=DEFAULT_SEED,
        calibration_path=(
            "data/calibration/"
            "master_calibration.json"
        ),
        output_dir=DEFAULT_OUTPUT,
        target_fraud_rate=(
            DEFAULT_TARGET_FRAUD_RATE
        ),
        fraud_tolerance=(
            DEFAULT_FRAUD_TOLERANCE
        ),
    ):

        self.num_customers = (
            customers
        )

        self.num_merchants = (
            merchants
        )

        self.days = days
        self.seed = seed

        self.target_fraud_rate = (
            target_fraud_rate
        )

        self.fraud_tolerance = (
            fraud_tolerance
        )

        self.rng = np.random.default_rng(
            seed
        )

        self.py_rng = random.Random(
            seed
        )

        self.calibration = Calibration(
            calibration_path
        )

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Entities
        # ----------------------------------------------------

        self.customers = None
        self.accounts = None
        self.cards = None
        self.devices = None
        self.merchants = None
        self.issuers = None
        self.acquirers = None
        self.tokens = None
        self.payment_agents = None
        self.agents_by_customer = defaultdict(list)

        # ----------------------------------------------------
        # Transactions/events
        # ----------------------------------------------------

        self.transactions = []
        self.events = []

        # ----------------------------------------------------
        # Attack registry
        # ----------------------------------------------------

        self.attack_registry = []
        self.attack_members = []

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        self.customer_cards = defaultdict(
            list
        )

        self.customer_devices = defaultdict(
            list
        )

        self.device_customers = defaultdict(
            set
        )

        self.customer_history = defaultdict(
            deque
        )

        # Permanent (non-rolling) per-customer behavioural
        # profile used by mimicry attacks (A1/A3/A4). Unlike
        # customer_history (48h rolling window, used for
        # velocity features), this accumulates for the whole
        # simulation so mimicry can reflect a customer's real
        # long-run distribution, not just their last two days.
        self.customer_profiles = defaultdict(
            lambda: {
                "amounts": [],
                "hours": [],
                "merchant_ids": [],
                "mccs": [],
            }
        )

        # Per-agent transaction history, used by
        # AGENT_IDENTITY_SPOOFING to compute how closely a given
        # transaction matches the agent's established signature.
        self.agent_history = defaultdict(
            lambda: {
                "amounts": [],
                "mccs": [],
            }
        )

        self.merchant_transactions = defaultdict(
            int
        )

        self.card_transactions = defaultdict(
            int
        )

        self.customer_transactions = defaultdict(
            int
        )

        self.customer_amount = defaultdict(
            float
        )

        self.transaction_counter = 0

        self.attack_counter = 0

    # ========================================================
    # ENTITY GENERATION
    # ========================================================

    def generate_issuers(
        self,
    ):

        rows = []

        names = [
            "Issuer_A",
            "Issuer_B",
            "Issuer_C",
            "Issuer_D",
            "Issuer_E",
            "Issuer_F",
        ]

        for i, name in enumerate(
            names
        ):

            rows.append(
                {
                    "issuer_id":
                        f"I{i:04d}",

                    "issuer_name":
                        name,

                    "country":
                        "IN",

                    "risk_tier":
                        self.py_rng.choice(
                            [
                                "LOW",
                                "MEDIUM",
                                "HIGH",
                            ]
                        ),
                }
            )

        self.issuers = pd.DataFrame(
            rows
        )

    # --------------------------------------------------------

    def generate_acquirers(
        self,
    ):

        rows = []

        for i in range(15):

            rows.append(
                {
                    "acquirer_id":
                        f"ACQ{i:04d}",

                    "country":
                        weighted_choice(
                            self.rng,
                            COUNTRIES,
                            COUNTRY_WEIGHTS,
                        ),

                    "risk_tier":
                        self.py_rng.choice(
                            [
                                "LOW",
                                "MEDIUM",
                                "HIGH",
                            ]
                        ),
                }
            )

        self.acquirers = pd.DataFrame(
            rows
        )

    # --------------------------------------------------------

    def generate_customers(
        self,
    ):

        rows = []

        for i in range(
            self.num_customers
        ):

            persona = weighted_choice(
                self.rng,
                PERSONAS,
                PERSONA_WEIGHTS,
            )

            country = weighted_choice(
                self.rng,
                COUNTRIES,
                COUNTRY_WEIGHTS,
            )

            if country == "IN":

                city = weighted_choice(
                    self.rng,
                    INDIA_CITIES,
                    IN_CITY_WEIGHTS,
                )

            else:

                city = "OTHER"

            # ------------------------------------------------
            # A11 - Synthetic identity.
            #
            # A small fraction of customers are generated as
            # synthetic identities: internally-plausible-looking
            # but structurally thin/inconsistent profiles
            # (very short account age combined with a
            # higher-value persona than usual, no natural
            # correlation between persona and age). This is a
            # customer-GENERATION-time attack, not a campaign
            # injected onto an existing customer, which mirrors
            # how synthetic identity fraud actually works in
            # real payment ecosystems (the identity itself is
            # fabricated before any transacting begins).
            # ------------------------------------------------

            is_synthetic = bool(
                self.rng.random()
                < SYNTHETIC_IDENTITY_RATE
            )

            account_age_days = int(
                self.rng.integers(
                    30,
                    4000,
                )
            )

            if is_synthetic:

                # Synthetic identities look "new" regardless of
                # the persona they were assigned - a thin-file
                # customer suddenly behaving like a high-value
                # consumer is exactly the anomaly a graph/velocity
                # -aware blue model should learn to catch.
                account_age_days = int(
                    self.rng.integers(
                        30,
                        120,
                    )
                )

            rows.append(
                {
                    "customer_id":
                        f"C{i:07d}",

                    "persona":
                        persona,

                    "home_country":
                        country,

                    "home_city":
                        city,

                    "account_age_days":
                        account_age_days,

                    "behaviour_profile_id":
                        f"BP_{persona}",

                    # Ground-truth marker only - must never be
                    # exposed to a blue-team feature set.
                    "is_synthetic_identity":
                        is_synthetic,
                }
            )

        self.customers = pd.DataFrame(
            rows
        )

    # --------------------------------------------------------

    def generate_accounts(
        self,
    ):

        rows = []

        for _, customer in (
            self.customers.iterrows()
        ):

            rows.append(
                {
                    "account_id":
                        f"A{len(rows):07d}",

                    "customer_id":
                        customer[
                            "customer_id"
                        ],

                    "issuer_id":
                        self.py_rng.choice(
                            self.issuers[
                                "issuer_id"
                            ].tolist()
                        ),

                    "account_type":
                        self.py_rng.choice(
                            [
                                "CHECKING",
                                "SAVINGS",
                                "CREDIT",
                            ]
                        ),

                    "available_balance":
                        round(
                            float(
                                self.rng.lognormal(
                                    10.5,
                                    1.0,
                                )
                            ),
                            2,
                        ),
                }
            )

        self.accounts = pd.DataFrame(
            rows
        )

    # --------------------------------------------------------

    def generate_cards(
        self,
    ):

        rows = []

        counter = 0

        for _, customer in (
            self.customers.iterrows()
        ):

            persona = customer[
                "persona"
            ]

            low, high = (
                PERSONA_CONFIG[
                    persona
                ]["cards"]
            )

            n_cards = int(
                self.rng.integers(
                    low,
                    high + 1,
                )
            )

            account_id = (
                self.accounts.loc[
                    self.accounts[
                        "customer_id"
                    ]
                    == customer[
                        "customer_id"
                    ],
                    "account_id",
                ].iloc[0]
            )

            issuer_id = (
                self.accounts.loc[
                    self.accounts[
                        "account_id"
                    ]
                    == account_id,
                    "issuer_id",
                ].iloc[0]
            )

            for _ in range(
                n_cards
            ):

                card_id = (
                    f"CARD{counter:09d}"
                )

                rows.append(
                    {
                        "card_id":
                            card_id,

                        "account_id":
                            account_id,

                        "customer_id":
                            customer[
                                "customer_id"
                            ],

                        "issuer_id":
                            issuer_id,

                        "product":
                            weighted_choice(
                                self.rng,
                                CARD_PRODUCTS,
                                CARD_PRODUCT_WEIGHTS,
                            ),

                        "status":
                            "ACTIVE",

                        "contactless_enabled":
                            bool(
                                self.rng.random()
                                < 0.88
                            ),

                        "ecommerce_enabled":
                            bool(
                                self.rng.random()
                                < 0.94
                            ),

                        "international_enabled":
                            bool(
                                self.rng.random()
                                < 0.72
                            ),
                    }
                )

                self.customer_cards[
                    customer[
                        "customer_id"
                    ]
                ].append(
                    card_id
                )

                counter += 1

        self.cards = pd.DataFrame(
            rows
        )

    # --------------------------------------------------------

    def generate_devices(
        self,
    ):

        rows = []

        counter = 0

        for _, customer in (
            self.customers.iterrows()
        ):

            persona = customer[
                "persona"
            ]

            low, high = (
                PERSONA_CONFIG[
                    persona
                ]["devices"]
            )

            n_devices = int(
                self.rng.integers(
                    low,
                    high + 1,
                )
            )

            for _ in range(
                n_devices
            ):

                device_id = (
                    f"D{counter:09d}"
                )

                rows.append(
                    {
                        "device_id":
                            device_id,

                        "primary_customer_id":
                            customer[
                                "customer_id"
                            ],

                        "device_type":
                            self.py_rng.choice(
                                [
                                    "ANDROID",
                                    "IOS",
                                    "WEB",
                                ]
                            ),

                        "device_age_days":
                            int(
                                self.rng.integers(
                                    1,
                                    1800,
                                )
                            ),

                        "trusted":
                            bool(
                                self.rng.random()
                                < 0.87
                            ),

                        "os_risk_score":
                            round(
                                float(
                                    self.rng.beta(
                                        2,
                                        15,
                                    )
                                ),
                                4,
                            ),
                    }
                )

                self.customer_devices[
                    customer[
                        "customer_id"
                    ]
                ].append(
                    device_id
                )

                self.device_customers[
                    device_id
                ].add(
                    customer[
                        "customer_id"
                    ]
                )

                counter += 1

        self.devices = pd.DataFrame(
            rows
        )

    # --------------------------------------------------------

    def generate_merchants(
        self,
    ):

        rows = []

        for i in range(
            self.num_merchants
        ):

            country = weighted_choice(
                self.rng,
                COUNTRIES,
                COUNTRY_WEIGHTS,
            )

            city = (
                weighted_choice(
                    self.rng,
                    INDIA_CITIES,
                    IN_CITY_WEIGHTS,
                )
                if country == "IN"
                else "OTHER"
            )

            rows.append(
                {
                    "merchant_id":
                        f"M{i:06d}",

                    "mcc":
                        weighted_choice(
                            self.rng,
                            MCCS,
                            MCC_WEIGHTS,
                        ),

                    "country":
                        country,

                    "city":
                        city,

                    "channel":
                        weighted_choice(
                            self.rng,
                            CHANNELS,
                            CHANNEL_WEIGHTS,
                        ),

                    "size_category":
                        self.py_rng.choice(
                            [
                                "SMALL",
                                "MEDIUM",
                                "LARGE",
                            ]
                        ),

                    "risk_tier":
                        self.py_rng.choice(
                            [
                                "LOW",
                                "MEDIUM",
                                "HIGH",
                            ]
                        ),

                    "merchant_age_days":
                        int(
                            self.rng.integers(
                                30,
                                5000,
                            )
                        ),
                }
            )

        self.merchants = pd.DataFrame(
            rows
        )

    # --------------------------------------------------------

    def generate_tokens(
        self,
    ):

        rows = []

        counter = 0

        for _, card in (
            self.cards.iterrows()
        ):

            if self.rng.random() >= 0.35:
                continue

            rows.append(
                {
                    "token_id":
                        f"TOK{counter:09d}",

                    "card_id":
                        card[
                            "card_id"
                        ],

                    "customer_id":
                        card[
                            "customer_id"
                        ],

                    "token_type":
                        self.py_rng.choice(
                            TOKEN_TYPES
                        ),

                    "status":
                        "ACTIVE",
                }
            )

            counter += 1

        self.tokens = pd.DataFrame(
            rows
        )

    # --------------------------------------------------------
    # A29-A34 support entity - autonomous payment agents.
    #
    # Only a subset of customers (digital_shopper /
    # business_professional / high_value_consumer personas, who
    # plausibly delegate purchasing to an assistant) get an
    # agent. Each agent has a DECLARED SCOPE which is the
    # authorization boundary A31 (scope abuse) violates, and a
    # behavioural signature which A30 (identity spoofing)
    # attempts to imitate.
    # --------------------------------------------------------

    def generate_payment_agents(
        self,
    ):

        eligible_personas = {
            "digital_shopper",
            "business_professional",
            "high_value_consumer",
            "subscription_user",
        }

        rows = []
        counter = 0

        for _, customer in (
            self.customers.iterrows()
        ):

            if (
                customer["persona"]
                not in eligible_personas
            ):
                continue

            if self.rng.random() >= 0.12:
                continue

            persona_mult = (
                PERSONA_CONFIG[
                    customer["persona"]
                ]["amount_multiplier"]
            )

            max_amount = round(
                float(
                    self.rng.uniform(
                        50.0,
                        400.0,
                    )
                )
                * persona_mult,
                2,
            )

            allowed_mcc_count = int(
                self.rng.integers(2, 5)
            )

            allowed_mccs = list(
                self.py_rng.sample(
                    MCCS,
                    k=min(
                        allowed_mcc_count,
                        len(MCCS),
                    ),
                )
            )

            rows.append(
                {
                    "agent_id":
                        f"AGT{counter:07d}",

                    "principal_customer_id":
                        customer["customer_id"],

                    "declared_max_amount":
                        max_amount,

                    "declared_allowed_mccs":
                        json.dumps(allowed_mccs),

                    "trust_level":
                        round(
                            float(
                                self.rng.uniform(0.55, 0.98)
                            ),
                            4,
                        ),

                    # Behavioural signature this agent is
                    # expected to exhibit - a compact vector
                    # used only to compute
                    # agent_identity_confidence at transaction
                    # time. Not itself exposed as a raw feature.
                    "_signature_amount_bias":
                        float(
                            self.rng.uniform(0.85, 1.15)
                        ),

                    "_signature_timing_bias":
                        float(
                            self.rng.uniform(0.0, 1.0)
                        ),
                }
            )

            counter += 1

        self.payment_agents = pd.DataFrame(rows)

        self.agents_by_customer = defaultdict(list)

        for _, row in self.payment_agents.iterrows():

            self.agents_by_customer[
                row["principal_customer_id"]
            ].append(row["agent_id"])

    # --------------------------------------------------------

    def generate_entities(
        self,
    ):

        print(
            "Generating issuers..."
        )

        self.generate_issuers()

        print(
            "Generating acquirers..."
        )

        self.generate_acquirers()

        print(
            "Generating customers..."
        )

        self.generate_customers()

        print(
            "Generating accounts..."
        )

        self.generate_accounts()

        print(
            "Generating cards..."
        )

        self.generate_cards()

        print(
            "Generating devices..."
        )

        self.generate_devices()

        print(
            "Generating merchants..."
        )

        self.generate_merchants()

        print(
            "Generating tokens..."
        )

        self.generate_tokens()

        print(
            "Generating payment agents..."
        )

        self.generate_payment_agents()

        print(
            f"Customers : "
            f"{len(self.customers):,}"
        )

        print(
            f"Accounts  : "
            f"{len(self.accounts):,}"
        )

        print(
            f"Cards     : "
            f"{len(self.cards):,}"
        )

        print(
            f"Devices   : "
            f"{len(self.devices):,}"
        )

        print(
            f"Merchants : "
            f"{len(self.merchants):,}"
        )

        print(
            f"Tokens    : "
            f"{len(self.tokens):,}"
        )

    # ========================================================
    # BEHAVIOUR
    # ========================================================

    def transaction_probability(
        self,
        customer,
    ):

        persona = customer[
            "persona"
        ]

        base = (
            PERSONA_CONFIG[
                persona
            ]["daily_transactions"]
        )

        activity = self.rng.lognormal(
            0,
            0.30,
        )

        return base * activity

    # --------------------------------------------------------

    def choose_card(
        self,
        customer_id,
    ):

        return self.py_rng.choice(
            self.customer_cards[
                customer_id
            ]
        )

    # --------------------------------------------------------

    def choose_device(
        self,
        customer_id,
    ):

        return self.py_rng.choice(
            self.customer_devices[
                customer_id
            ]
        )

    # --------------------------------------------------------

    def choose_merchant(
        self,
    ):

        weights = (
            self.merchants[
                "mcc"
            ]
            .map(
                dict(
                    zip(
                        MCCS,
                        MCC_WEIGHTS,
                    )
                )
            )
            .fillna(1.0)
            .to_numpy()
        )

        weights = (
            weights
            / weights.sum()
        )

        idx = self.rng.choice(
            len(self.merchants),
            p=weights,
        )

        return self.merchants.iloc[
            idx
        ]

    # --------------------------------------------------------

    def choose_channel(
        self,
        merchant,
    ):

        if self.rng.random() < 0.82:

            return merchant[
                "channel"
            ]

        return weighted_choice(
            self.rng,
            CHANNELS,
            CHANNEL_WEIGHTS,
        )

    # --------------------------------------------------------

    def choose_authentication(
        self,
        channel,
    ):

        if channel == "POS":

            return weighted_choice(
                self.rng,
                [
                    "PIN_LIKE",
                    "CONTACTLESS_AUTH",
                ],
                [
                    0.62,
                    0.38,
                ],
            )

        if channel == "CONTACTLESS":

            return "CONTACTLESS_AUTH"

        if channel == "E_COMMERCE":

            return weighted_choice(
                self.rng,
                [
                    "3DS_LIKE",
                    "OTP_LIKE",
                    "TOKENIZED",
                ],
                [
                    0.40,
                    0.30,
                    0.30,
                ],
            )

        if channel == "IN_APP":

            return weighted_choice(
                self.rng,
                [
                    "BIOMETRIC_LIKE",
                    "TOKENIZED",
                    "OTP_LIKE",
                ],
                [
                    0.35,
                    0.40,
                    0.25,
                ],
            )

        return "TOKENIZED"

    # ========================================================
    # HISTORY
    # ========================================================

    def customer_velocity(
        self,
        customer_id,
        current_time,
    ):

        history = (
            self.customer_history[
                customer_id
            ]
        )

        one_hour = (
            current_time
            - pd.Timedelta(
                hours=1
            )
        )

        twenty_four = (
            current_time
            - pd.Timedelta(
                hours=24
            )
        )

        txn_1h = 0
        txn_24h = 0
        amount_1h = 0.0
        amount_24h = 0.0

        for event in history:

            timestamp = event[
                "timestamp"
            ]

            amount = event[
                "amount"
            ]

            if timestamp >= one_hour:

                txn_1h += 1
                amount_1h += amount

            if timestamp >= twenty_four:

                txn_24h += 1
                amount_24h += amount

        return (
            txn_1h,
            txn_24h,
            amount_1h,
            amount_24h,
        )

    # ========================================================
    # BASE TRANSACTION GENERATION
    # ========================================================

    def create_base_transaction(
        self,
        customer,
        timestamp,
    ):

        customer_id = customer[
            "customer_id"
        ]

        card_id = self.choose_card(
            customer_id
        )

        device_id = self.choose_device(
            customer_id
        )

        merchant = self.choose_merchant()

        channel = self.choose_channel(
            merchant
        )

        auth_method = (
            self.choose_authentication(
                channel
            )
        )

        auth_result = "SUCCESS"

        r = self.rng.random()

        if r < 0.025:

            auth_result = "FAILED"

        elif r < 0.040:

            auth_result = "CHALLENGE"

        (
            txn_1h,
            txn_24h,
            amount_1h,
            amount_24h,
        ) = self.customer_velocity(
            customer_id,
            timestamp,
        )

        persona = customer[
            "persona"
        ]

        amount = (
            self.calibration.sample_amount(
                self.rng,
                PERSONA_CONFIG[
                    persona
                ]["amount_multiplier"],
            )
        )

        # ------------------------------------------------
        # Legitimate agent-initiated transactions.
        #
        # Without this, initiating_agent_id was only ever set
        # by fraud injection (AGENT_SCOPE_ABUSE /
        # AGENT_IDENTITY_SPOOFING), so blue had zero negative
        # examples of "normal" agent behaviour to learn from -
        # every agent-tagged row was fraud by construction,
        # which taught the model "agent present == fraud"
        # rather than "agent present AND out of scope == fraud".
        # A customer who owns a payment agent transacts through
        # it some of the time, WITHIN its declared scope, which
        # gives blue the negative class it needs.
        # ------------------------------------------------

        legit_agent_id = None

        owned_agents = self.agents_by_customer.get(
            customer_id
        )

        if owned_agents and self.rng.random() < 0.35:

            candidate_agent_id = self.py_rng.choice(
                owned_agents
            )

            agent_row = self.payment_agents.loc[
                self.payment_agents["agent_id"]
                == candidate_agent_id
            ].iloc[0]

            allowed_mccs = json.loads(
                agent_row["declared_allowed_mccs"]
            )

            # Only attribute this transaction to the agent if
            # the merchant we already picked actually falls
            # within its declared scope - a legitimate agent
            # transaction must, by definition, be in-scope.
            if merchant["mcc"] in allowed_mccs:

                legit_agent_id = candidate_agent_id

                declared_max = float(
                    agent_row["declared_max_amount"]
                )

                amount = round(
                    min(
                        float(amount),
                        declared_max
                        * float(
                            self.rng.uniform(0.15, 1.0)
                        ),
                    ),
                    2,
                )

        token_id = None

        token_rows = (
            self.tokens.loc[
                self.tokens[
                    "card_id"
                ]
                == card_id
            ]
        )

        if (
            not token_rows.empty
            and self.rng.random() < 0.40
        ):

            token_id = self.py_rng.choice(
                token_rows[
                    "token_id"
                ].tolist()
            )

        issuer_id = (
            self.cards.loc[
                self.cards[
                    "card_id"
                ]
                == card_id,
                "issuer_id",
            ].iloc[0]
        )

        account_id = (
            self.cards.loc[
                self.cards[
                    "card_id"
                ]
                == card_id,
                "account_id",
            ].iloc[0]
        )

        geo_score = (
            self.geo_distance_score(
                customer,
                merchant,
            )
        )

        device_row = (
            self.devices.loc[
                self.devices[
                    "device_id"
                ]
                == device_id
            ]
        )

        device_trusted = bool(
            device_row[
                "trusted"
            ].iloc[0]
        )

        device_risk = float(
            device_row[
                "os_risk_score"
            ].iloc[0]
        )

        txn_id = (
            f"T{self.transaction_counter:010d}"
        )

        self.transaction_counter += 1

        txn = {

            "transaction_id":
                txn_id,

            "timestamp":
                timestamp.isoformat(),

            "customer_id":
                customer_id,

            "account_id":
                account_id,

            "card_id":
                card_id,

            "token_id":
                token_id,

            "device_id":
                device_id,

            "merchant_id":
                merchant[
                    "merchant_id"
                ],

            "issuer_id":
                issuer_id,

            "acquirer_id":
                self.py_rng.choice(
                    self.acquirers[
                        "acquirer_id"
                    ].tolist()
                ),

            "amount":
                amount,

            "currency":
                "INR",

            "customer_country":
                customer[
                    "home_country"
                ],

            "customer_city":
                customer[
                    "home_city"
                ],

            "merchant_country":
                merchant[
                    "country"
                ],

            "merchant_city":
                merchant[
                    "city"
                ],

            "channel":
                channel,

            "mcc":
                merchant[
                    "mcc"
                ],

            "authentication_method":
                auth_method,

            "authentication_result":
                auth_result,

            "customer_txn_count_1h":
                txn_1h,

            "customer_txn_count_24h":
                txn_24h,

            "customer_amount_1h":
                round(
                    amount_1h,
                    2,
                ),

            "customer_amount_24h":
                round(
                    amount_24h,
                    2,
                ),

            "merchant_txn_count":
                self.merchant_transactions[
                    merchant[
                        "merchant_id"
                    ]
                ],

            "card_txn_count":
                self.card_transactions[
                    card_id
                ],

            "customer_total_txn":
                self.customer_transactions[
                    customer_id
                ],

            "device_customer_count_24h":
                len(
                    self.device_customers[
                        device_id
                    ]
                ),

            "geo_distance_score":
                round(
                    geo_score,
                    4,
                ),

            "merchant_risk_tier":
                merchant[
                    "risk_tier"
                ],

            "device_trusted":
                device_trusted,

            "device_os_risk_score":
                round(
                    device_risk,
                    4,
                ),

            # ------------------------------------------------
            # Agentic-commerce fields (A29-A34 support).
            #
            # initiating_agent_id is null for the vast majority
            # of transactions (a human initiated them directly).
            # The two score fields are neutral defaults for
            # non-agent transactions and are only meaningfully
            # populated for agent-initiated ones, including
            # AGENT_SCOPE_ABUSE / AGENT_IDENTITY_SPOOFING
            # campaigns.
            # ------------------------------------------------

            "initiating_agent_id":
                legit_agent_id,

            "agent_scope_conformance_score":
                1.0,

            "agent_identity_confidence":
                1.0,

            # ------------------------------------------------
            # Ground truth fields
            # ------------------------------------------------

            "fraud_label":
                0,

            "attack_id":
                None,

            "attack_family":
                None,

            "attack_sequence":
                None,

            "is_campaign_transaction":
                0,

            # Useful for evaluation but should NOT be used
            # as a blue-team feature.
            "ground_truth_marker":
                "LEGITIMATE",
        }

        return txn

    # --------------------------------------------------------

    def register_transaction_state(
        self,
        txn,
    ):

        customer_id = txn[
            "customer_id"
        ]

        timestamp = pd.Timestamp(
            txn[
                "timestamp"
            ]
        )

        amount = float(
            txn[
                "amount"
            ]
        )

        self.customer_history[
            customer_id
        ].append(
            {
                "timestamp":
                    timestamp,

                "amount":
                    amount,

                "merchant_id":
                    txn[
                        "merchant_id"
                    ],
            }
        )

        cutoff = (
            timestamp
            - pd.Timedelta(
                hours=48
            )
        )

        history = (
            self.customer_history[
                customer_id
            ]
        )

        while (
            history
            and history[0][
                "timestamp"
            ] < cutoff
        ):

            history.popleft()

        self.merchant_transactions[
            txn[
                "merchant_id"
            ]
        ] += 1

        self.card_transactions[
            txn[
                "card_id"
            ]
        ] += 1

        self.customer_transactions[
            customer_id
        ] += 1

        self.customer_amount[
            customer_id
        ] += amount

        # ------------------------------------------------
        # Permanent behavioural profile (mimicry attacks).
        # Capped length so memory stays bounded on long runs;
        # a few thousand samples is more than enough signal
        # for a distribution to mimic.
        # ------------------------------------------------

        profile = self.customer_profiles[
            customer_id
        ]

        profile["amounts"].append(amount)
        profile["hours"].append(timestamp.hour)
        profile["merchant_ids"].append(
            txn["merchant_id"]
        )
        profile["mccs"].append(
            txn.get("mcc")
        )

        if len(profile["amounts"]) > 2000:
            profile["amounts"].pop(0)
            profile["hours"].pop(0)
            profile["merchant_ids"].pop(0)
            profile["mccs"].pop(0)

        agent_id = txn.get("initiating_agent_id")

        if agent_id:

            ah = self.agent_history[agent_id]
            ah["amounts"].append(amount)
            ah["mccs"].append(txn.get("mcc"))

            if len(ah["amounts"]) > 500:
                ah["amounts"].pop(0)
                ah["mccs"].pop(0)

    # --------------------------------------------------------

    def geo_distance_score(
        self,
        customer,
        merchant,
    ):

        if (
            customer[
                "home_country"
            ]
            != merchant[
                "country"
            ]
        ):

            return 1.0

        if (
            customer[
                "home_city"
            ]
            != merchant[
                "city"
            ]
        ):

            return 0.35

        return 0.0

    # ========================================================
    # BASE SIMULATION
    # ========================================================

    def generate_base_transactions(
        self,
    ):

        start = pd.Timestamp(
            SIMULATION_START
        )

        total = 0

        for day in range(
            self.days
        ):

            day_start = (
                start
                + pd.Timedelta(
                    days=day
                )
            )

            daily = 0

            # Generate customers in randomized order so
            # temporal clustering isn't deterministic by
            # customer ID.

            indices = self.rng.permutation(
                len(self.customers)
            )

            for idx in indices:

                customer = (
                    self.customers.iloc[
                        idx
                    ]
                )

                expected = (
                    self.transaction_probability(
                        customer
                    )
                )

                n = int(
                    self.rng.poisson(
                        expected
                    )
                )

                for _ in range(n):

                    timestamp = (
                        choose_timestamp(
                            self.rng,
                            day_start,
                        )
                    )

                    txn = (
                        self.create_base_transaction(
                            customer,
                            timestamp,
                        )
                    )

                    self.register_transaction_state(
                        txn
                    )

                    self.transactions.append(
                        txn
                    )

                    daily += 1
                    total += 1

            print(
                f"Day "
                f"{day + 1:03d}/"
                f"{self.days:03d}: "
                f"{daily:,} transactions "
                f"(total {total:,})"
            )

    # ========================================================
    # FRAUD BUDGET
    # ========================================================

    def calculate_fraud_budget(
        self,
    ):

        total = len(
            self.transactions
        )

        target = int(
            round(
                total
                * self.target_fraud_rate
            )
        )

        target = max(
            target,
            1,
        )

        return target

    # ========================================================
    # ATTACK FAMILY SELECTION
    # ========================================================

    def choose_attack_family(
        self,
        remaining_budget,
    ):

        families = list(
            ATTACK_FAMILY_WEIGHTS.keys()
        )

        weights = [
            ATTACK_FAMILY_WEIGHTS[
                family
            ]
            for family in families
        ]

        # If the remaining budget is very small,
        # any family is still possible.

        return weighted_choice(
            self.rng,
            families,
            weights,
        )

    # ========================================================
    # CAMPAIGN SIZE
    # ========================================================

    def choose_campaign_size(
        self,
        family,
        remaining_budget,
    ):

        config = (
            CAMPAIGN_CONFIG[
                family
            ]
        )

        minimum = config[
            "min_size"
        ]

        maximum = min(
            config[
                "max_size"
            ],
            remaining_budget,
        )

        if maximum < minimum:

            return remaining_budget

        # Bias toward smaller campaigns.
        size = int(
            self.rng.integers(
                minimum,
                maximum + 1,
            )
        )

        return size

    # ========================================================
    # TARGET SELECTION
    # ========================================================

    def choose_campaign_target(
        self,
        family,
    ):

        target_type = (
            CAMPAIGN_CONFIG[
                family
            ]["target_type"]
        )

        if target_type == "card":

            return self.py_rng.choice(
                self.cards[
                    "card_id"
                ].tolist()
            )

        if target_type == "customer":

            return self.py_rng.choice(
                self.customers[
                    "customer_id"
                ].tolist()
            )

        if target_type == "device":

            return self.py_rng.choice(
                self.devices[
                    "device_id"
                ].tolist()
            )

        if target_type == "merchant":

            return self.py_rng.choice(
                self.merchants[
                    "merchant_id"
                ].tolist()
            )

        if target_type == "agent":

            if (
                self.payment_agents is None
                or self.payment_agents.empty
            ):

                return None

            return self.py_rng.choice(
                self.payment_agents[
                    "agent_id"
                ].tolist()
            )

        raise ValueError(
            f"Unknown target type: "
            f"{target_type}"
        )

    # ========================================================
    # CAMPAIGN CANDIDATE SELECTION
    # ========================================================

    def find_campaign_candidates(
        self,
        family,
        target,
    ):

        df = pd.DataFrame(
            self.transactions
        )

        if df.empty:

            return []

        target_type = (
            CAMPAIGN_CONFIG[
                family
            ]["target_type"]
        )

        if target_type == "card":

            candidates = df.index[
                df[
                    "card_id"
                ]
                == target
            ].tolist()

        elif target_type == "customer":

            candidates = df.index[
                df[
                    "customer_id"
                ]
                == target
            ].tolist()

        elif target_type == "device":

            candidates = df.index[
                df[
                    "device_id"
                ]
                == target
            ].tolist()

        elif target_type == "merchant":

            candidates = df.index[
                df[
                    "merchant_id"
                ]
                == target
            ].tolist()

        elif target_type == "agent":

            if (
                self.payment_agents is None
                or self.payment_agents.empty
                or target is None
            ):

                candidates = []

            else:

                agent_rows = self.payment_agents.loc[
                    self.payment_agents["agent_id"]
                    == target
                ]

                if agent_rows.empty:

                    candidates = []

                else:

                    principal_id = agent_rows.iloc[0][
                        "principal_customer_id"
                    ]

                    # Agent campaigns are injected onto the
                    # PRINCIPAL customer's own legitimate
                    # transactions (the agent transacts on
                    # their behalf), not onto some independent
                    # agent-owned transaction stream.
                    candidates = df.index[
                        df["customer_id"] == principal_id
                    ].tolist()

        else:

            candidates = []

        return candidates

    # ========================================================
    # ATTACK MODIFIERS
    # ========================================================

    def apply_attack_pattern(
        self,
        txn,
        family,
        sequence,
        params=None,
        campaign_state=None,
    ):

        """
        Modify observable transaction characteristics
        to make the attack family behaviorally meaningful.

        IMPORTANT:
        These modifications are intentionally imperfect.
        They should create signals, not deterministic rules.

        params: optional dict of normalized [0,1] red-agent
        knobs (see ATTACK_PARAM_SCHEMA). Families that predate
        the red/blue loop (CARD_TESTING, ACCOUNT_TAKEOVER,
        DEVICE_COMPROMISE, VELOCITY_ABUSE, UNUSUAL_GEO,
        MERCHANT_ANOMALY) ignore this and behave exactly as in
        v2.3 - unchanged, for backward compatibility.

        campaign_state: optional mutable dict the caller
        (inject_campaign) persists across sequence steps of the
        SAME campaign. Used by families whose behaviour depends
        on what happened earlier in the campaign (e.g. adaptive
        card testing reacting to the previous transaction's
        authentication_result).
        """

        params = params or {}
        campaign_state = (
            campaign_state
            if campaign_state is not None
            else {}
        )

        def p(name):
            return float(
                params.get(
                    name,
                    DEFAULT_RED_PARAMS.get(name, 0.5),
                )
            )

        if family == "CARD_TESTING":

            # Small amounts.
            txn[
                "amount"
            ] = round(
                float(
                    self.rng.uniform(
                        1.0,
                        20.0,
                    )
                ),
                2,
            )

            txn[
                "channel"
            ] = "E_COMMERCE"

            txn[
                "authentication_method"
            ] = self.py_rng.choice(
                [
                    "3DS_LIKE",
                    "OTP_LIKE",
                    "TOKENIZED",
                ]
            )

        elif family == "ACCOUNT_TAKEOVER":

            txn[
                "authentication_result"
            ] = self.py_rng.choice(
                [
                    "FAILED",
                    "CHALLENGE",
                    "SUCCESS",
                ]
            )

            txn[
                "device_trusted"
            ] = False

            txn[
                "device_os_risk_score"
            ] = round(
                float(
                    self.rng.uniform(
                        0.45,
                        0.95,
                    )
                ),
                4,
            )

            if self.rng.random() < 0.35:

                txn[
                    "channel"
                ] = "E_COMMERCE"

        elif family == "DEVICE_COMPROMISE":

            txn[
                "device_trusted"
            ] = False

            txn[
                "device_os_risk_score"
            ] = round(
                float(
                    self.rng.uniform(
                        0.55,
                        0.98,
                    )
                ),
                4,
            )

            txn[
                "device_customer_count_24h"
            ] = max(
                2,
                int(
                    txn[
                        "device_customer_count_24h"
                    ]
                ),
            )

        elif family == "VELOCITY_ABUSE":

            txn[
                "customer_txn_count_1h"
            ] = max(
                5,
                int(
                    txn[
                        "customer_txn_count_1h"
                    ]
                ),
            )

            txn[
                "customer_txn_count_24h"
            ] = max(
                10,
                int(
                    txn[
                        "customer_txn_count_24h"
                    ]
                ),
            )

        elif family == "UNUSUAL_GEO":

            txn[
                "geo_distance_score"
            ] = 1.0

            if (
                txn[
                    "customer_country"
                ]
                == txn[
                    "merchant_country"
                ]
            ):

                txn[
                    "merchant_country"
                ] = self.py_rng.choice(
                    [
                        c
                        for c in COUNTRIES
                        if c
                        != txn[
                            "customer_country"
                        ]
                    ]
                )

                txn[
                    "merchant_city"
                ] = "OTHER"

        elif family == "MERCHANT_ANOMALY":

            txn[
                "merchant_risk_tier"
            ] = "HIGH"

            if self.rng.random() < 0.30:

                txn[
                    "amount"
                ] = round(
                    float(
                        txn[
                            "amount"
                        ]
                    )
                    * self.rng.uniform(
                        2.0,
                        6.0,
                    ),
                    2,
                )

        # ====================================================
        # A1 - Behavioral transaction mimicry
        #
        # Blends the customer's OWN historical amount/timing/
        # merchant distribution with a generic fraud profile.
        # mimicry_blend=1.0 => looks almost exactly like this
        # customer's real behaviour (hard to catch on
        # univariate rules). amount_deviation controls how far
        # the mimicked amount is pushed toward this customer's
        # own upper tail, which is where the attacker's profit
        # motive shows up.
        # ====================================================

        elif family == "BEHAVIORAL_MIMICRY":

            blend = p("mimicry_blend")
            deviation = p("amount_deviation")

            profile = self.customer_profiles.get(
                txn["customer_id"]
            )

            if profile and len(profile["amounts"]) >= 5:

                own_amounts = sorted(profile["amounts"])
                n = len(own_amounts)

                # deviation pushes the sampled percentile
                # toward the customer's own tail.
                pct = clamp(
                    0.5 + deviation * 0.49,
                    0.01,
                    0.99,
                )

                own_value = own_amounts[
                    int(pct * (n - 1))
                ]

                generic_value = float(
                    self.calibration.sample_amount(
                        self.rng,
                        1.0,
                    )
                )

                txn["amount"] = round(
                    blend * own_value
                    + (1 - blend) * generic_value,
                    2,
                )

                if (
                    blend > 0.5
                    and profile["merchant_ids"]
                    and self.rng.random() < blend
                ):

                    txn["mcc"] = self.py_rng.choice(
                        [
                            m
                            for m in profile["mccs"]
                            if m
                        ]
                        or [txn["mcc"]]
                    )

            else:

                # Not enough history to mimic yet - fall back
                # to a generic small perturbation so the
                # campaign still injects something meaningful.
                txn["amount"] = round(
                    float(
                        self.rng.uniform(10.0, 60.0)
                    ),
                    2,
                )

        # ====================================================
        # A3 - Temporal mimicry
        #
        # Only the timestamp hour is nudged toward the
        # customer's own historical hour-of-day distribution.
        # Isolated from A1 so the writeup can show an ablation:
        # timing-only mimicry vs amount-only vs full blend.
        # ====================================================

        elif family == "TEMPORAL_MIMICRY":

            blend = p("mimicry_blend")

            profile = self.customer_profiles.get(
                txn["customer_id"]
            )

            if profile and len(profile["hours"]) >= 5 and (
                self.rng.random() < blend
            ):

                own_hour = self.py_rng.choice(
                    profile["hours"]
                )

                ts = pd.Timestamp(txn["timestamp"])

                new_ts = ts.replace(
                    hour=int(own_hour)
                )

                txn["timestamp"] = new_ts.isoformat()

        # ====================================================
        # A4 - Amount-distribution mimicry
        #
        # Same mechanism as the amount half of A1, kept as its
        # own family so amount-only mimicry can be measured and
        # searched over independently by the red agent.
        # ====================================================

        elif family == "AMOUNT_MIMICRY":

            blend = p("mimicry_blend")
            deviation = p("amount_deviation")

            profile = self.customer_profiles.get(
                txn["customer_id"]
            )

            if profile and len(profile["amounts"]) >= 5:

                own_amounts = sorted(profile["amounts"])
                n = len(own_amounts)

                pct = clamp(
                    0.5 + deviation * 0.49,
                    0.01,
                    0.99,
                )

                own_value = own_amounts[
                    int(pct * (n - 1))
                ]

                generic_value = float(
                    self.calibration.sample_amount(
                        self.rng,
                        1.0,
                    )
                )

                txn["amount"] = round(
                    blend * own_value
                    + (1 - blend) * generic_value,
                    2,
                )

        # ====================================================
        # A9 - Adaptive / distributed card testing
        #
        # Parameterized variant of CARD_TESTING that reacts to
        # the OUTCOME of the previous transaction in the same
        # campaign (campaign_state persists across sequence
        # steps): escalate the probe amount after a success,
        # retreat and rotate merchant/MCC after a decline. This
        # is the sequential-decision attack best suited to a
        # true red AGENT rather than a one-shot sampler.
        # ====================================================

        elif family == "ADAPTIVE_CARD_TESTING":

            escalation_rate = p("escalation_rate")
            retreat_sensitivity = p("retreat_sensitivity")

            last_amount = campaign_state.get(
                "last_amount", 1.0
            )

            last_result = campaign_state.get(
                "last_result", "SUCCESS"
            )

            if last_result == "SUCCESS":

                growth = 1.0 + escalation_rate * 4.0

                new_amount = clamp(
                    last_amount * growth,
                    1.0,
                    200.0,
                )

            else:

                shrink = 1.0 - retreat_sensitivity * 0.8

                new_amount = clamp(
                    last_amount * max(shrink, 0.1),
                    1.0,
                    200.0,
                )

                if (
                    retreat_sensitivity > 0.5
                    and self.rng.random()
                    < retreat_sensitivity
                ):

                    txn["mcc"] = self.py_rng.choice(MCCS)

            txn["amount"] = round(new_amount, 2)
            txn["channel"] = "E_COMMERCE"

            campaign_state["last_amount"] = new_amount
            campaign_state["last_result"] = txn[
                "authentication_result"
            ]

        # ====================================================
        # A6 - Relational camouflage
        #
        # Distributes fraud across many low-signal entities
        # instead of concentrating it. spread_density controls
        # how "thin" the campaign is spread; realized here as
        # keeping each individual transaction's velocity/amount
        # features unremarkable while attack_id ties them
        # together in attack_registry/attack_members for a
        # graph-aware blue model to discover.
        # ====================================================

        elif family == "RELATIONAL_CAMOUFLAGE":

            spread_density = p("spread_density")

            if spread_density > 0.5:

                # Thin spread: keep the amount close to the
                # global median so no single transaction is an
                # outlier - the ONLY signal is the shared
                # attack_id / entity graph, not any one field.
                txn["amount"] = round(
                    float(
                        self.calibration.sample_amount(
                            self.rng,
                            0.6,
                        )
                    ),
                    2,
                )

            else:

                # Concentrated: larger amounts, easier for a
                # non-graph model to catch, by design (this is
                # the low end of the parameter space red should
                # learn to avoid).
                txn["amount"] = round(
                    float(
                        self.calibration.sample_amount(
                            self.rng,
                            1.8,
                        )
                    ),
                    2,
                )

        # ====================================================
        # A30 - Agent identity spoofing
        #
        # A transaction claims to be initiated by a legitimate
        # agent_id, but its behavioural signature (amount scale,
        # timing) deviates from that agent's established
        # history. spoofing_similarity controls how close the
        # forgery is; agent_identity_confidence is the
        # observable feature blue can use to catch it.
        # ====================================================

        elif family == "AGENT_IDENTITY_SPOOFING":

            similarity = p("spoofing_similarity")

            target_agent_id = campaign_state.get(
                "agent_id"
            )

            if (
                target_agent_id
                and self.payment_agents is not None
                and not self.payment_agents.empty
                and target_agent_id
                in set(
                    self.payment_agents["agent_id"]
                )
            ):

                agent_row = self.payment_agents.loc[
                    self.payment_agents["agent_id"]
                    == target_agent_id
                ].iloc[0]

                txn["initiating_agent_id"] = (
                    target_agent_id
                )

                sig_bias = float(
                    agent_row["_signature_amount_bias"]
                )

                # similarity=1 => forged amount matches the
                # real agent's bias closely. similarity=0 =>
                # forged amount ignores it, producing a visible
                # mismatch.
                effective_bias = (
                    similarity * sig_bias
                    + (1 - similarity)
                    * self.rng.uniform(0.4, 2.2)
                )

                txn["amount"] = round(
                    float(txn["amount"])
                    * effective_bias,
                    2,
                )

                txn["agent_identity_confidence"] = round(
                    clamp(
                        similarity
                        + self.rng.uniform(-0.08, 0.08),
                        0.0,
                        1.0,
                    ),
                    4,
                )

        # ====================================================
        # A31 - Agent authorization/scope abuse
        #
        # A genuine agent transacts outside its own declared
        # scope (spend limit and/or allowed MCCs).
        # scope_violation_depth controls how far outside scope;
        # escalation_pace controls whether that happens abruptly
        # (low pace = first transaction in the campaign already
        # far outside scope) or gradually (high pace = violation
        # depth grows across sequence steps).
        # ====================================================

        elif family == "AGENT_SCOPE_ABUSE":

            depth = p("scope_violation_depth")
            pace = p("escalation_pace")

            target_agent_id = campaign_state.get(
                "agent_id"
            )

            if (
                target_agent_id
                and self.payment_agents is not None
                and not self.payment_agents.empty
                and target_agent_id
                in set(
                    self.payment_agents["agent_id"]
                )
            ):

                agent_row = self.payment_agents.loc[
                    self.payment_agents["agent_id"]
                    == target_agent_id
                ].iloc[0]

                txn["initiating_agent_id"] = (
                    target_agent_id
                )

                max_amount = float(
                    agent_row["declared_max_amount"]
                )

                allowed_mccs = json.loads(
                    agent_row["declared_allowed_mccs"]
                )

                # Gradual escalation: effective depth for THIS
                # step grows with sequence number when pace is
                # high, so early campaign transactions look
                # closer to legitimate and later ones drift. A
                # floor is applied so that even the very first
                # step of a slow-escalation campaign carries a
                # visible (if small) violation signal, rather
                # than looking indistinguishable from perfectly
                # in-scope - real slow-burn abuse still starts
                # somewhere outside scope, it doesn't start AT
                # zero.
                step_progress = clamp(
                    sequence
                    / max(
                        campaign_state.get(
                            "campaign_size", sequence
                        ),
                        1,
                    ),
                    0.0,
                    1.0,
                )

                if pace < 0.5:
                    effective_depth = depth
                else:
                    effective_depth = depth * clamp(
                        0.25 + 0.75 * step_progress,
                        0.0,
                        1.0,
                    )

                overshoot = 1.0 + effective_depth * 8.0

                txn["amount"] = round(
                    max_amount * overshoot,
                    2,
                )

                if (
                    allowed_mccs
                    and effective_depth > 0.15
                ):

                    txn["mcc"] = self.py_rng.choice(
                        [
                            m
                            for m in MCCS
                            if m not in allowed_mccs
                        ]
                        or MCCS
                    )

                txn["agent_scope_conformance_score"] = (
                    round(
                        clamp(
                            1.0 - effective_depth,
                            0.0,
                            1.0,
                        ),
                        4,
                    )
                )

        # Campaign sequence metadata.
        txn[
            "attack_sequence"
        ] = sequence

    # ========================================================
    # CAMPAIGN INJECTION
    # ========================================================

    def inject_campaign(
        self,
        family,
        requested_size,
        campaign_id,
        params=None,
        target=None,
    ):

        target = (
            target
            if target is not None
            else self.choose_campaign_target(
                family
            )
        )

        if target is None:

            # e.g. an "agent" target_type family requested
            # before any payment_agents exist.
            return 0

        candidates = (
            self.find_campaign_candidates(
                family,
                target,
            )
        )

        if not candidates:

            return 0

        # Campaigns should have temporal coherence.
        #
        # Sort candidate transactions by time and choose
        # a contiguous temporal region where possible.

        df = pd.DataFrame(
            self.transactions
        )

        candidates = sorted(
            candidates,
            key=lambda idx:
                pd.Timestamp(
                    df.loc[
                        idx,
                        "timestamp",
                    ]
                ),
        )

        size = min(
            requested_size,
            len(candidates),
        )

        if size <= 0:

            return 0

        # Choose a contiguous-ish window.
        if len(candidates) > size:

            start_max = (
                len(candidates)
                - size
            )

            start_idx = int(
                self.rng.integers(
                    0,
                    start_max + 1,
                )
            )

            selected = candidates[
                start_idx:
                start_idx + size
            ]

        else:

            selected = candidates

        actual_size = 0

        target_type = (
            CAMPAIGN_CONFIG[family]["target_type"]
        )

        campaign_state = {
            "campaign_size": len(selected),
        }

        if target_type == "agent":

            campaign_state["agent_id"] = target

        for sequence, idx in enumerate(
            selected,
            start=1,
        ):

            txn = self.transactions[
                idx
            ]

            # Never overwrite another attack.
            if txn[
                "fraud_label"
            ] == 1:

                continue

            txn[
                "fraud_label"
            ] = 1

            txn[
                "attack_id"
            ] = campaign_id

            txn[
                "attack_family"
            ] = family

            txn[
                "is_campaign_transaction"
            ] = 1

            txn[
                "ground_truth_marker"
            ] = "FRAUD"

            self.apply_attack_pattern(
                txn,
                family,
                sequence,
                params=params,
                campaign_state=campaign_state,
            )

            actual_size += 1

            self.attack_members.append(
                {
                    "attack_id":
                        campaign_id,

                    "transaction_id":
                        txn[
                            "transaction_id"
                        ],

                    "sequence":
                        sequence,

                    "attack_family":
                        family,
                }
            )

        if actual_size > 0:

            timestamps = [
                pd.Timestamp(
                    self.transactions[
                        idx
                    ][
                        "timestamp"
                    ]
                )
                for idx in selected
            ]

            self.attack_registry.append(
                {
                    "attack_id":
                        campaign_id,

                    "attack_family":
                        family,

                    "target_type":
                        CAMPAIGN_CONFIG[
                            family
                        ][
                            "target_type"
                        ],

                    "target_id":
                        target,

                    "transaction_count":
                        actual_size,

                    "start_time":
                        min(
                            timestamps
                        ).isoformat(),

                    "end_time":
                        max(
                            timestamps
                        ).isoformat(),

                    "campaign":
                        True,
                }
            )

        return actual_size

    # ========================================================
    # FRAUD INJECTION
    # ========================================================

    def inject_fraud(
        self,
    ):

        total = len(
            self.transactions
        )

        fraud_budget = (
            self.calculate_fraud_budget()
        )

        print(
            "\nFraud calibration:"
        )

        print(
            f"Total transactions : "
            f"{total:,}"
        )

        print(
            f"Target fraud rate  : "
            f"{self.target_fraud_rate:.4%}"
        )

        print(
            f"Fraud budget       : "
            f"{fraud_budget:,}"
        )

        campaign_budget = int(
            round(
                fraud_budget
                * CAMPAIGN_FRACTION
            )
        )

        isolated_budget = (
            fraud_budget
            - campaign_budget
        )

        print(
            f"Campaign budget    : "
            f"{campaign_budget:,}"
        )

        print(
            f"Isolated budget    : "
            f"{isolated_budget:,}"
        )

        remaining = campaign_budget

        # ----------------------------------------------------
        # Campaign generation
        # ----------------------------------------------------

        safety_counter = 0

        while (
            remaining > 0
            and safety_counter < 1000
        ):

            safety_counter += 1

            family = (
                self.choose_attack_family(
                    remaining
                )
            )

            size = (
                self.choose_campaign_size(
                    family,
                    remaining,
                )
            )

            # Ensure campaign generation does not create
            # excessive numbers of one-transaction attacks.
            if size < 2:

                break

            attack_id = (
                f"ATK{self.attack_counter:07d}"
            )

            self.attack_counter += 1

            created = (
                self.inject_campaign(
                    family,
                    size,
                    attack_id,
                )
            )

            if created <= 0:

                continue

            remaining -= created

        # ----------------------------------------------------
        # Isolated fraud
        # ----------------------------------------------------

        current_fraud = sum(
            int(
                txn[
                    "fraud_label"
                ]
            )
            for txn in self.transactions
        )

        isolated_remaining = max(
            0,
            fraud_budget
            - current_fraud,
        )

        print(
            f"Campaign fraud created: "
            f"{current_fraud:,}"
        )

        print(
            f"Remaining fraud budget: "
            f"{isolated_remaining:,}"
        )

        # Select only currently legitimate transactions.
        available = [
            i
            for i, txn
            in enumerate(
                self.transactions
            )
            if txn[
                "fraud_label"
            ] == 0
        ]

        if isolated_remaining > len(
            available
        ):

            isolated_remaining = len(
                available
            )

        if isolated_remaining > 0:

            selected = self.rng.choice(
                available,
                size=isolated_remaining,
                replace=False,
            )

            for idx in selected:

                txn = self.transactions[
                    int(idx)
                ]

                family = (
                    self.choose_attack_family(
                        isolated_remaining
                    )
                )

                attack_id = (
                    f"ATK{self.attack_counter:07d}"
                )

                self.attack_counter += 1

                txn[
                    "fraud_label"
                ] = 1

                txn[
                    "attack_id"
                ] = attack_id

                txn[
                    "attack_family"
                ] = family

                txn[
                    "is_campaign_transaction"
                ] = 0

                txn[
                    "ground_truth_marker"
                ] = "FRAUD"

                self.apply_attack_pattern(
                    txn,
                    family,
                    1,
                )

                self.attack_registry.append(
                    {
                        "attack_id":
                            attack_id,

                        "attack_family":
                            family,

                        "target_type":
                            "isolated",

                        "target_id":
                            None,

                        "transaction_count":
                            1,

                        "start_time":
                            txn[
                                "timestamp"
                            ],

                        "end_time":
                            txn[
                                "timestamp"
                            ],

                        "campaign":
                            False,
                    }
                )

                self.attack_members.append(
                    {
                        "attack_id":
                            attack_id,

                        "transaction_id":
                            txn[
                                "transaction_id"
                            ],

                        "sequence":
                            1,

                        "attack_family":
                            family,
                    }
                )

        # ----------------------------------------------------
        # Final exact budget enforcement
        # ----------------------------------------------------

        current_fraud = [
            i
            for i, txn
            in enumerate(
                self.transactions
            )
            if txn[
                "fraud_label"
            ] == 1
        ]

        # If campaign collisions resulted in too few fraud
        # transactions, fill the exact remainder.

        if len(current_fraud) < fraud_budget:

            remaining_needed = (
                fraud_budget
                - len(current_fraud)
            )

            available = [
                i
                for i, txn
                in enumerate(
                    self.transactions
                )
                if txn[
                    "fraud_label"
                ] == 0
            ]

            selected = self.rng.choice(
                available,
                size=min(
                    remaining_needed,
                    len(available),
                ),
                replace=False,
            )

            for idx in selected:

                txn = self.transactions[
                    int(idx)
                ]

                family = (
                    self.choose_attack_family(
                        remaining_needed
                    )
                )

                attack_id = (
                    f"ATK{self.attack_counter:07d}"
                )

                self.attack_counter += 1

                txn[
                    "fraud_label"
                ] = 1

                txn[
                    "attack_id"
                ] = attack_id

                txn[
                    "attack_family"
                ] = family

                txn[
                    "is_campaign_transaction"
                ] = 0

                txn[
                    "ground_truth_marker"
                ] = "FRAUD"

                self.apply_attack_pattern(
                    txn,
                    family,
                    1,
                )

                self.attack_registry.append(
                    {
                        "attack_id":
                            attack_id,

                        "attack_family":
                            family,

                        "target_type":
                            "isolated",

                        "target_id":
                            None,

                        "transaction_count":
                            1,

                        "start_time":
                            txn[
                                "timestamp"
                            ],

                        "end_time":
                            txn[
                                "timestamp"
                            ],

                        "campaign":
                            False,
                    }
                )

                self.attack_members.append(
                    {
                        "attack_id":
                            attack_id,

                        "transaction_id":
                            txn[
                                "transaction_id"
                            ],

                        "sequence":
                            1,

                        "attack_family":
                            family,
                    }
                )

        # If for any unexpected reason we exceeded the target,
        # remove excess fraud labels while preserving campaigns
        # where possible.

        current_fraud = [
            i
            for i, txn
            in enumerate(
                self.transactions
            )
            if txn[
                "fraud_label"
            ] == 1
        ]

        if len(current_fraud) > fraud_budget:

            excess = (
                len(current_fraud)
                - fraud_budget
            )

            # Prefer removing isolated fraud first.
            isolated = [
                i
                for i in current_fraud
                if self.transactions[
                    i
                ][
                    "is_campaign_transaction"
                ] == 0
            ]

            remove = isolated[
                :excess
            ]

            if len(remove) < excess:

                remaining_excess = (
                    excess
                    - len(remove)
                )

                campaign_rows = [
                    i
                    for i in current_fraud
                    if i not in remove
                ]

                remove.extend(
                    campaign_rows[
                        :remaining_excess
                    ]
                )

            for idx in remove:

                txn = self.transactions[
                    idx
                ]

                txn[
                    "fraud_label"
                ] = 0

                txn[
                    "attack_id"
                ] = None

                txn[
                    "attack_family"
                ] = None

                txn[
                    "attack_sequence"
                ] = None

                txn[
                    "is_campaign_transaction"
                ] = 0

                txn[
                    "ground_truth_marker"
                ] = "LEGITIMATE"

        # Rebuild attack artifacts from final truth.
        self.rebuild_attack_artifacts()

    # ========================================================
    # RED-AGENT FACING API
    # ========================================================
    #
    # This is the boundary the adversarial loop is built on top
    # of. An external red agent (bandit, evolutionary search,
    # RL policy - anything) calls execute_attack_action once per
    # chosen action. Everything about HOW the action was chosen,
    # and everything about scoring/retraining blue, lives outside
    # this file. This method only needs to:
    #   1. spend fraud budget by injecting one parameterized
    #      campaign into the live transaction stream, and
    #   2. hand back enough identifying information
    #      (transaction_ids) for an external blue model to score
    #      and for the caller to build an AttackOutcome once
    #      those scores exist.
    #
    # NOTE: this does not itself compute detection - detection
    # is blue's job, external to the simulator by design (the
    # simulator must not be able to see or influence blue's
    # verdict, or the adversarial evaluation would be invalid).

    def execute_attack_action(
        self,
        action: "AttackAction",
    ):

        if action.family not in FRAUD_FAMILIES:

            raise ValueError(
                f"Unknown attack family: "
                f"{action.family}"
            )

        if action.family not in CAMPAIGN_CONFIG:

            raise ValueError(
                f"Family {action.family} has no "
                f"CAMPAIGN_CONFIG entry and cannot be "
                f"injected as a red-agent campaign."
            )

        campaign_id = (
            f"ATK{self.attack_counter:07d}"
        )

        self.attack_counter += 1

        target = action.target_id

        if target is None:

            target = self.choose_campaign_target(
                action.family
            )

        created_before = len(self.attack_members)

        actual_size = self.inject_campaign(
            action.family,
            action.campaign_size,
            campaign_id,
            params=action.params,
            target=target,
        )

        if actual_size <= 0:

            return {
                "attack_id": campaign_id,
                "family": action.family,
                "params": dict(action.params),
                "transaction_ids": [],
                "actual_size": 0,
            }

        new_members = self.attack_members[
            created_before:
        ]

        transaction_ids = [
            m["transaction_id"] for m in new_members
        ]

        return {
            "attack_id": campaign_id,
            "family": action.family,
            "params": dict(action.params),
            "transaction_ids": transaction_ids,
            "actual_size": actual_size,
        }

    # ========================================================
    # ATTACK ARTIFACT REBUILD
    # ========================================================

    def rebuild_attack_artifacts(
        self,
    ):

        self.attack_registry = []
        self.attack_members = []

        df = pd.DataFrame(
            self.transactions
        )

        fraud_df = df.loc[
            df[
                "fraud_label"
            ] == 1
        ]

        for attack_id, group in (
            fraud_df.groupby(
                "attack_id",
                dropna=True,
            )
        ):

            group = group.sort_values(
                "timestamp"
            )

            family = (
                group[
                    "attack_family"
                ].iloc[0]
            )

            campaign = bool(
                len(group) > 1
                or group[
                    "is_campaign_transaction"
                ].iloc[0] == 1
            )

            target_type = (
                "isolated"
                if not campaign
                else CAMPAIGN_CONFIG[
                    family
                ][
                    "target_type"
                ]
            )

            target_id = None

            if campaign:

                if target_type == "customer":

                    target_id = (
                        group[
                            "customer_id"
                        ].mode().iloc[0]
                    )

                elif target_type == "card":

                    target_id = (
                        group[
                            "card_id"
                        ].mode().iloc[0]
                    )

                elif target_type == "device":

                    target_id = (
                        group[
                            "device_id"
                        ].mode().iloc[0]
                    )

                elif target_type == "merchant":

                    target_id = (
                        group[
                            "merchant_id"
                        ].mode().iloc[0]
                    )

            self.attack_registry.append(
                {
                    "attack_id":
                        attack_id,

                    "attack_family":
                        family,

                    "target_type":
                        target_type,

                    "target_id":
                        target_id,

                    "transaction_count":
                        len(group),

                    "start_time":
                        group[
                            "timestamp"
                        ].iloc[0],

                    "end_time":
                        group[
                            "timestamp"
                        ].iloc[-1],

                    "campaign":
                        campaign,
                }
            )

            for sequence, (_, row) in enumerate(
                group.iterrows(),
                start=1,
            ):

                self.attack_members.append(
                    {
                        "attack_id":
                            attack_id,

                        "transaction_id":
                            row[
                                "transaction_id"
                            ],

                        "sequence":
                            sequence,

                        "attack_family":
                            family,
                    }
                )

                # Ensure sequence is synchronized.
                idx = row.name

                self.transactions[
                    idx
                ][
                    "attack_sequence"
                ] = sequence

    # ========================================================
    # EVENTS
    # ========================================================

    def generate_events(
        self,
    ):

        self.events = []

        for txn in self.transactions:

            transaction_id = txn[
                "transaction_id"
            ]

            timestamp = pd.Timestamp(
                txn[
                    "timestamp"
                ]
            )

            base = {
                "transaction_id":
                    transaction_id,

                "timestamp":
                    timestamp.isoformat(),

                "customer_id":
                    txn[
                        "customer_id"
                    ],

                "merchant_id":
                    txn[
                        "merchant_id"
                    ],

                "event_source":
                    "synthetic_payment_network",
            }

            self.events.append(
                {
                    **base,
                    "event_type":
                        "AUTHORIZATION_REQUEST",
                }
            )

            self.events.append(
                {
                    **base,
                    "event_type":
                        "RISK_EVALUATION",
                }
            )

            if txn[
                "authentication_method"
            ] in [
                "3DS_LIKE",
                "OTP_LIKE",
                "BIOMETRIC_LIKE",
            ]:

                self.events.append(
                    {
                        **base,
                        "event_type":
                            "CUSTOMER_AUTHENTICATION",
                    }
                )

            if txn[
                "token_id"
            ] is not None:

                self.events.append(
                    {
                        **base,
                        "event_type":
                            "TOKEN_RESOLUTION",
                    }
                )

            if txn[
                "fraud_label"
            ] == 1:

                self.events.append(
                    {
                        **base,
                        "event_type":
                            "FRAUD_SIGNAL",
                    }
                )

            self.events.append(
                {
                    **base,
                    "event_type":
                        (
                            "AUTHORIZATION_DECLINED"
                            if txn[
                                "authentication_result"
                            ]
                            == "FAILED"
                            else
                            "AUTHORIZATION_APPROVED"
                        ),
                }
            )

    # ========================================================
    # VALIDATION
    # ========================================================

    def validation_report(
        self,
    ):

        df = pd.DataFrame(
            self.transactions
        )

        total = len(df)

        fraud = int(
            df[
                "fraud_label"
            ].sum()
        )

        actual_rate = (
            fraud
            / max(
                total,
                1,
            )
        )

        deviation = (
            actual_rate
            - self.target_fraud_rate
        )

        absolute_deviation = abs(
            deviation
        )

        relative_deviation = (
            absolute_deviation
            / max(
                self.target_fraud_rate,
                1e-12,
            )
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "PAYMENT ENVIRONMENT V2.3"
        )

        print(
            "VALIDATION REPORT"
        )

        print(
            "=" * 70
        )

        print(
            f"\nTransactions       : "
            f"{total:,}"
        )

        print(
            f"Fraud transactions : "
            f"{fraud:,}"
        )

        print(
            f"Target fraud rate  : "
            f"{self.target_fraud_rate:.4%}"
        )

        print(
            f"Actual fraud rate  : "
            f"{actual_rate:.4%}"
        )

        print(
            f"Absolute deviation : "
            f"{absolute_deviation:.4%}"
        )

        print(
            f"Relative deviation : "
            f"{relative_deviation:.2%}"
        )

        # ----------------------------------------------------
        # Amounts
        # ----------------------------------------------------

        print(
            "\nAmount distribution:"
        )

        amount_stats = (
            df[
                "amount"
            ].describe(
                percentiles=[
                    0.01,
                    0.05,
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                    0.95,
                    0.99,
                ]
            )
        )

        print(
            amount_stats
        )

        # ----------------------------------------------------
        # Channels
        # ----------------------------------------------------

        print(
            "\nChannel distribution:"
        )

        print(
            df[
                "channel"
            ].value_counts()
        )

        print(
            "\nChannel percentages:"
        )

        print(
            (
                df[
                    "channel"
                ]
                .value_counts(
                    normalize=True
                )
                * 100
            ).round(2)
        )

        # ----------------------------------------------------
        # MCC
        # ----------------------------------------------------

        print(
            "\nMCC distribution:"
        )

        print(
            df[
                "mcc"
            ].value_counts()
        )

        # ----------------------------------------------------
        # Persona
        # ----------------------------------------------------

        merged = df.merge(
            self.customers[
                [
                    "customer_id",
                    "persona",
                ]
            ],
            on="customer_id",
            how="left",
        )

        print(
            "\nPersona distribution:"
        )

        print(
            merged[
                "persona"
            ].value_counts()
        )

        # ----------------------------------------------------
        # Authentication
        # ----------------------------------------------------

        print(
            "\nAuthentication methods:"
        )

        print(
            df[
                "authentication_method"
            ].value_counts()
        )

        print(
            "\nAuthentication results:"
        )

        print(
            df[
                "authentication_result"
            ].value_counts()
        )

        # ----------------------------------------------------
        # Fraud families
        # ----------------------------------------------------

        fraud_df = df.loc[
            df[
                "fraud_label"
            ] == 1
        ]

        print(
            "\nFraud by attack family:"
        )

        print(
            fraud_df[
                "attack_family"
            ].value_counts()
        )

        print(
            "\nFraud family percentages:"
        )

        if not fraud_df.empty:

            print(
                (
                    fraud_df[
                        "attack_family"
                    ]
                    .value_counts(
                        normalize=True
                    )
                    * 100
                ).round(2)
            )

        # ----------------------------------------------------
        # Fraud amounts
        # ----------------------------------------------------

        print(
            "\nFraud amount statistics:"
        )

        if not fraud_df.empty:

            print(
                fraud_df[
                    "amount"
                ].describe()
            )

        # ----------------------------------------------------
        # Attack campaigns
        # ----------------------------------------------------

        attack_df = pd.DataFrame(
            self.attack_registry
        )

        print(
            "\nAttack campaigns:"
        )

        if attack_df.empty:

            print(
                "No attacks generated."
            )

        else:

            print(
                f"Unique attack IDs : "
                f"{attack_df['attack_id'].nunique():,}"
            )

            print(
                "Campaign size distribution:"
            )

            print(
                attack_df[
                    "transaction_count"
                ].describe()
            )

            print(
                "Multi-transaction campaigns : "
                f"{int((attack_df['transaction_count'] > 1).sum()):,}"
            )

        # ----------------------------------------------------
        # Devices
        # ----------------------------------------------------

        print(
            "\nDevice/customer relationships:"
        )

        device_counts = (
            self.device_customers
        )

        relationship_counts = [
            len(
                customers
            )
            for customers
            in device_counts.values()
        ]

        if relationship_counts:

            print(
                pd.Series(
                    relationship_counts
                ).describe()
            )

        # ----------------------------------------------------
        # Token rate
        # ----------------------------------------------------

        print(
            "\nTokenized transactions:"
        )

        print(
            f"{df['token_id'].notna().mean():.2%}"
        )

        # ----------------------------------------------------
        # Cross-country
        # ----------------------------------------------------

        cross_country = (
            df[
                "customer_country"
            ]
            != df[
                "merchant_country"
            ]
        )

        print(
            "\nCross-country transactions:"
        )

        print(
            f"{cross_country.mean():.2%}"
        )

        # ----------------------------------------------------
        # Hour
        # ----------------------------------------------------

        print(
            "\nTransactions by hour:"
        )

        hours = (
            pd.to_datetime(
                df[
                    "timestamp"
                ]
            )
            .dt.hour
            .value_counts()
            .sort_index()
        )

        print(
            hours
        )

        # ----------------------------------------------------
        # Fraud vs legitimate
        # ----------------------------------------------------

        print(
            "\nFraud vs legitimate amounts:"
        )

        print(
            df.groupby(
                "fraud_label"
            )[
                "amount"
            ].agg(
                [
                    "count",
                    "mean",
                    "median",
                    "std",
                ]
            )
        )

        # ====================================================
        # INTEGRITY CHECKS
        # ====================================================

        print(
            "\nData integrity checks:"
        )

        checks = {}

        checks[
            "transaction_id_unique"
        ] = (
            df[
                "transaction_id"
            ].is_unique
        )

        valid_customers = set(
            self.customers[
                "customer_id"
            ]
        )

        valid_merchants = set(
            self.merchants[
                "merchant_id"
            ]
        )

        valid_cards = set(
            self.cards[
                "card_id"
            ]
        )

        valid_devices = set(
            self.devices[
                "device_id"
            ]
        )

        checks[
            "customer_ids_valid"
        ] = (
            df[
                "customer_id"
            ]
            .isin(
                valid_customers
            )
            .all()
        )

        checks[
            "merchant_ids_valid"
        ] = (
            df[
                "merchant_id"
            ]
            .isin(
                valid_merchants
            )
            .all()
        )

        checks[
            "card_ids_valid"
        ] = (
            df[
                "card_id"
            ]
            .isin(
                valid_cards
            )
            .all()
        )

        checks[
            "device_ids_valid"
        ] = (
            df[
                "device_id"
            ]
            .isin(
                valid_devices
            )
            .all()
        )

        checks[
            "amounts_positive"
        ] = (
            df[
                "amount"
            ] > 0
        ).all()

        checks[
            "fraud_labels_binary"
        ] = set(
            df[
                "fraud_label"
            ].unique()
        ).issubset(
            {
                0,
                1,
            }
        )

        checks[
            "attack_ids_valid_for_fraud"
        ] = (
            df.loc[
                df[
                    "fraud_label"
                ] == 1,
                "attack_id",
            ]
            .notna()
            .all()
        )

        checks[
            "legitimate_has_no_attack"
        ] = (
            df.loc[
                df[
                    "fraud_label"
                ] == 0,
                "attack_id",
            ]
            .isna()
            .all()
        )

        checks[
            "fraud_rate_within_tolerance"
        ] = (
            absolute_deviation
            <= self.fraud_tolerance
        )

        for name, passed in (
            checks.items()
        ):

            print(
                f"  {name:<38} "
                f"{'PASS' if passed else 'FAIL'}"
            )

        # ----------------------------------------------------
        # Attack membership integrity
        # ----------------------------------------------------

        membership_df = pd.DataFrame(
            self.attack_members
        )

        if not membership_df.empty:

            membership_ids = set(
                membership_df[
                    "transaction_id"
                ]
            )

            fraud_ids = set(
                fraud_df[
                    "transaction_id"
                ]
            )

            checks[
                "attack_membership_matches_fraud"
            ] = (
                membership_ids
                == fraud_ids
            )

        else:

            checks[
                "attack_membership_matches_fraud"
            ] = fraud == 0

        # ----------------------------------------------------
        # Exact budget check
        # ----------------------------------------------------

        expected_budget = (
            self.calculate_fraud_budget()
        )

        checks[
            "fraud_budget_exact"
        ] = (
            fraud
            == expected_budget
        )

        print(
            "\nFinal validation:"
        )

        print(
            f"Expected fraud budget : "
            f"{expected_budget:,}"
        )

        print(
            f"Actual fraud count    : "
            f"{fraud:,}"
        )

        print(
            f"Exact budget match    : "
            f"{'PASS' if checks['fraud_budget_exact'] else 'FAIL'}"
        )

        validation_passed = all(
            checks.values()
        )

        print(
            "\nValidation status:"
        )

        print(
            "PASS"
            if validation_passed
            else "FAIL"
        )

        if not validation_passed:

            raise RuntimeError(
                "V2.3 validation failed. "
                "Do not use this simulation dataset "
                "for model training until the failure "
                "has been investigated."
            )

        print(
            "\nValidation complete."
        )

        return {
            "validation_passed":
                validation_passed,

            "transactions":
                total,

            "fraud_transactions":
                fraud,

            "target_fraud_rate":
                self.target_fraud_rate,

            "actual_fraud_rate":
                actual_rate,

            "absolute_deviation":
                absolute_deviation,

            "relative_deviation":
                relative_deviation,

            "fraud_budget":
                expected_budget,

            "checks":
                checks,
        }

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        validation,
    ):

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        tables = {

            "customers":
                self.customers,

            "accounts":
                self.accounts,

            "cards":
                self.cards,

            "devices":
                self.devices,

            "merchants":
                self.merchants,

            "issuers":
                self.issuers,

            "acquirers":
                self.acquirers,

            "tokens":
                self.tokens,

            "transactions":
                pd.DataFrame(
                    self.transactions
                ),

            "events":
                pd.DataFrame(
                    self.events
                ),

            "attack_registry":
                pd.DataFrame(
                    self.attack_registry
                ),

            "attack_members":
                pd.DataFrame(
                    self.attack_members
                ),
        }

        for name, df in (
            tables.items()
        ):

            path = (
                self.output_dir
                / f"{name}.parquet"
            )

            df.to_parquet(
                path,
                index=False,
            )

            print(
                f"Saved {name}: "
                f"{len(df):,} rows → "
                f"{path}"
            )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata = {

            "environment":
                "PaymentEnvironmentV23",

            "version":
                VERSION,

            "seed":
                self.seed,

            "days":
                self.days,

            "customers":
                len(
                    self.customers
                ),

            "merchants":
                len(
                    self.merchants
                ),

            "cards":
                len(
                    self.cards
                ),

            "devices":
                len(
                    self.devices
                ),

            "tokens":
                len(
                    self.tokens
                ),

            "transactions":
                len(
                    self.transactions
                ),

            "events":
                len(
                    self.events
                ),

            "fraud_transactions":
                validation[
                    "fraud_transactions"
                ],

            "target_fraud_rate":
                self.target_fraud_rate,

            "actual_fraud_rate":
                validation[
                    "actual_fraud_rate"
                ],

            "fraud_budget":
                validation[
                    "fraud_budget"
                ],

            "attack_families":
                ATTACK_FAMILY_WEIGHTS,

            "campaign_fraction":
                CAMPAIGN_FRACTION,

            "calibration":
                str(
                    Path(
                        "data/calibration/"
                        "master_calibration.json"
                    )
                ),

            "validation":
                validation,

            "warning":
                (
                    "Synthetic environment. "
                    "Not Mastercard proprietary data."
                ),
        }

        with open(
            self.output_dir
            / "metadata.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                safe_json(
                    metadata
                ),
                f,
                indent=2,
            )

        # ----------------------------------------------------
        # Validation JSON
        # ----------------------------------------------------

        with open(
            self.output_dir
            / "validation_report.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                safe_json(
                    validation
                ),
                f,
                indent=2,
            )

        print(
            "\nSimulation saved to:"
        )

        print(
            self.output_dir
        )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Generate Payment Environment V2.3."
        )
    )

    parser.add_argument(
        "--customers",
        type=int,
        default=DEFAULT_CUSTOMERS,
    )

    parser.add_argument(
        "--merchants",
        type=int,
        default=DEFAULT_MERCHANTS,
    )

    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--target-fraud-rate",
        type=float,
        default=DEFAULT_TARGET_FRAUD_RATE,
    )

    parser.add_argument(
        "--fraud-tolerance",
        type=float,
        default=DEFAULT_FRAUD_TOLERANCE,
    )

    parser.add_argument(
        "--calibration",
        type=str,
        default=(
            "data/calibration/"
            "master_calibration.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(
            DEFAULT_OUTPUT
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Starting Payment Environment V2.3"
    )

    print(
        "=" * 70
    )

    print(
        f"Version    : {VERSION}"
    )

    print(
        f"Seed       : {args.seed}"
    )

    print(
        f"Customers  : {args.customers:,}"
    )

    print(
        f"Merchants  : {args.merchants:,}"
    )

    print(
        f"Days       : {args.days}"
    )

    print(
        f"Target fraud rate : "
        f"{args.target_fraud_rate:.4%}"
    )

    environment = (
        PaymentEnvironmentV23(
            customers=args.customers,
            merchants=args.merchants,
            days=args.days,
            seed=args.seed,
            calibration_path=args.calibration,
            output_dir=args.output,
            target_fraud_rate=(
                args.target_fraud_rate
            ),
            fraud_tolerance=(
                args.fraud_tolerance
            ),
        )
    )

    print(
        "\nGenerating synthetic payment environment..."
    )

    environment.generate_entities()

    print(
        "\nGenerating base transactions..."
    )

    environment.generate_base_transactions()

    print(
        "\nInjecting controlled fraud..."
    )

    environment.inject_fraud()

    print(
        "\nGenerating authorization events..."
    )

    environment.generate_events()

    validation = (
        environment.validation_report()
    )

    environment.save(
        validation
    )


if __name__ == "__main__":
    main()