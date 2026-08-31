# Steelman

**Forging the fraud your defenses haven't seen yet.**

Steelman is a closed-loop red-team / blue-team system for card payment fraud. An
AI red team invents GenAI-style payment fraud inside a simulated card payment
ecosystem. An AI blue team has to detect it. The attacks become training data,
and the detector's blind spots become the next round of attacks.

The name comes from argument: *steelmanning* means building
the strongest possible version of a position before you respond to it. Steelman
does the same thing to its own defenses — it builds the most convincing fraud
it can, so whatever passes the test is actually resilient.

## What's here

Three pillars, one loop:

| Pillar | What it does | Where |
|---|---|---|
| **Identify & Generate** | A payment ecosystem simulator (9 entity tables, 13 GenAI-native attack families) | `payment_environment_v24.py` |
| **Generate** | An AI red team — an LLM strategist plus a statistical search algorithm — that picks attacks and tunes them | `red_strategist.py`, `red_bandit.py` |
| **Defend** | A detector trained on the ecosystem's own transaction data, retrained on what the red team generates | `blue_model.py` |

The three pieces are wired together by the run scripts, which is where the loop
actually happens.

## The ecosystem

Every transaction Steelman generates sits at the centre of nine linked entity
tables — customers, accounts, cards, tokens, devices, merchants, issuers,
acquirers, and autonomous payment agents — calibrated against public statistics
from the IEEE-CIS Fraud Detection and PaySim datasets, at a realistic 2% fraud
rate. Calibration profiles for both are in `data/calibration/`.

## The attacks

Thirteen attack families, each a small space of tunable parameters rather than
a fixed pattern — behavioural, temporal, and amount mimicry; adaptive card
testing; relational camouflage; synthetic identity; account takeover; device
compromise; velocity abuse; unusual geography; merchant anomaly; and, newest,
agent identity spoofing and agent scope abuse — attacks on AI assistants
transacting on a customer's behalf.

## The red team

Each generation, a large language model (gemini 3.5 flash here) reads the recent history of
attacks — which family, which parameters, how close each one came to evading
detection — and decides what to try next, with its reasoning logged in plain
language. A Thompson-sampling bandit fills in the exact numeric parameters
within whichever family the strategist picks.

## The detector

A gradient-boosted classifier trained on observable transaction fields only,
with graph features (shared-device customer counts, network structure) built
specifically to catch attacks that don't look anomalous in any single
transaction. On held-out data:

| Metric | Value |
|---|---|
| Precision | 0.32 |
| Recall | 0.56 |
| F1 | 0.41 |
| AUC-PR | 0.51 |

Classical fraud (account takeover, device compromise, velocity abuse) is
caught at 100% recall. The hardest GenAI-native mimicry attacks are caught
far less often — that gap is what the closed loop keeps working on.

## Running it

```bash
pip install -r requirements.txt   # or see below
```

**1. Generate the base dataset and train the static detector:**

```bash
python payment_environment_v24.py --customers 1500 --merchants 150 --days 20 \
    --output data/blue_dataset_v2
python run_baseline.py
```

**2. Run the closed loop** (needs a free Gemini API key — see
[aistudio.google.com/apikey](https://aistudio.google.com/apikey)):

```bash
python run_closed_loop.py \
    --backend gemini \
    --api-key YOUR_KEY \
    --model gemini-3.5-flash-lite \
    --generations 30 \
    --campaigns-per-generation 20 \
    --retrain-every 5
```

This writes `generation_log.json`, `comparison_log.json`,
`strategist_reasoning_log.jsonl`, and `final_debrief.txt` to `output_v4/` —
a full per-generation record of what the red team tried, what the detector
caught, and why.

`run_adversarial_loop_llm.py` runs the same red team against a *frozen*
detector (no retraining) — useful for isolating how evasive the red team gets
on its own, before the detector starts adapting.

`test_bandit_standalone.py` runs the search algorithm on its own, no LLM
required — a fast way to sanity-check the loop mechanics.

### Dependencies

```
pandas, numpy, xgboost, scikit-learn, networkx, shap, openai
```

## Repository layout

```
payment_environment_v24.py     the simulator: entities, transactions, attacks
blue_model.py                   the detector: features, training, scoring
red_bandit.py                   parameter search (Thompson sampling)
red_strategist.py               LLM strategist + reasoning log
run_baseline.py                 trains the static detector, no red team
run_closed_loop.py               the full loop: red team + retraining detector
run_adversarial_loop_llm.py      red team against a frozen detector
test_bandit_standalone.py        search algorithm only, no LLM

data/
  blue_dataset_v2/                generated transactions
  calibration/                    IEEE-CIS / PaySim calibration profiles

output_v2/blue_model_baseline/   the trained static detector
output_v3/                        frozen-detector run: logs, reasoning, debrief
output_v4/                        full closed-loop run: logs, reasoning, debrief
```

## What's next

The closed loop's biggest open question: retraining the detector on the red
team's *successful* attacks alone gives only a small, inconsistent
improvement. The next step is training on its failed attempts too, so the
detector learns both sides of the decision boundary instead of just the side
that fooled it.
