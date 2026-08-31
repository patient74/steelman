# Blue Model — Static Baseline Report

Trained on 38,498 transactions, evaluated on 16,500 held-out transactions from the same simulator batch (default-parameter fraud, no adversarial search).

## Overall metrics

- **Precision**: 0.3189
- **Recall**: 0.5576
- **F1**: 0.4057
- **AUC-PR (average precision)**: 0.5124
- **AUC-ROC**: 0.8428
- **Decision threshold** (calibrated for precision >= 0.5): 0.4840

## Per-family recall (held-out)

| Attack family | n (fraud) | Recall | Mean predicted probability |
|---|---|---|---|
| UNUSUAL_GEO | 13 | 0.0000 | 0.1263 |
| TEMPORAL_MIMICRY | 21 | 0.0952 | 0.2242 |
| BEHAVIORAL_MIMICRY | 24 | 0.1250 | 0.2280 |
| AMOUNT_MIMICRY | 15 | 0.2667 | 0.2795 |
| RELATIONAL_CAMOUFLAGE | 55 | 0.3091 | 0.3170 |
| CARD_TESTING | 22 | 0.3182 | 0.3323 |
| MERCHANT_ANOMALY | 31 | 0.3871 | 0.3774 |
| AGENT_SCOPE_ABUSE | 7 | 0.5714 | 0.5504 |
| ADAPTIVE_CARD_TESTING | 19 | 0.7368 | 0.6184 |
| AGENT_IDENTITY_SPOOFING | 9 | 0.7778 | 0.7261 |
| ACCOUNT_TAKEOVER | 45 | 1.0000 | 0.9594 |
| VELOCITY_ABUSE | 28 | 1.0000 | 0.9937 |
| DEVICE_COMPROMISE | 41 | 1.0000 | 0.9894 |

## Top feature importances

| Feature | Importance |
|---|---|
| device_customer_count_24h | 0.0726 |
| customer_txn_count_24h | 0.0371 |
| channel_E_COMMERCE | 0.0222 |
| device_os_risk_score | 0.0206 |
| merchant_city_Pune | 0.0204 |
| agent_identity_confidence | 0.0204 |
| agent_scope_conformance_score | 0.0194 |
| merchant_risk_tier_HIGH | 0.0191 |
| customer_country_IN | 0.0184 |
| customer_city_Delhi | 0.0164 |
| merchant_city_Chennai | 0.0163 |
| merchant_country_US | 0.0162 |
| mcc_ELECTRONICS | 0.0161 |
| authentication_result_CHALLENGE | 0.0154 |
| merchant_city_OTHER | 0.0150 |
| amount | 0.0150 |
| authentication_method_PIN_LIKE | 0.0149 |
| customer_city_OTHER | 0.0148 |
| mcc_FASHION | 0.0146 |
| merchant_country_AU | 0.0140 |