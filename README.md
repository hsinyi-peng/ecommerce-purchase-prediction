# E-commerce Purchase Prediction

Predicting whether a browsing session ends in a purchase from user behavior signals (time on site, pages viewed, discounting, device, traffic source), using logistic regression, a classification tree (CART), and a random forest.

## Overview

Most e-commerce sessions don't convert. This project builds a binary classifier — `purchased` (1) vs. `not purchased` (0) — from session-level behavioral features, and compares three modeling approaches on held-out data.

## Dataset

`data/ecommerce_customer_behavior.csv` — 25,000 rows, 29 columns. Source: [Indian E-commerce Customer Behavior and Purchase (Kaggle)](https://www.kaggle.com/datasets/kundanbedmutha/indian-e-commerce-customer-behavior-and-purchase).

Categorical fields are pre-encoded as integers by the source dataset (e.g. `device_type`, `user_type`, `marketing_channel`, `payment_method`). Features used in modeling:

| Column | Description |
|---|---|
| `purchased` | Target: 1 if the session ended in a purchase, else 0 |
| `unit_price` | Price of the viewed/purchased product |
| `discount_percent` | Discount applied, as a percentage |
| `pages_viewed` | Number of pages viewed in the session |
| `time_on_site_sec` | Session duration in seconds |
| `device_type` | Encoded device category (desktop/mobile/tablet) |
| `user_type` | Encoded new vs. returning visitor flag |
| `marketing_channel` | Encoded acquisition channel |

The full raw file also includes session/product identifiers, cart and review behavior, and visit timing fields not used in this modeling pass.

## Repository structure

```
.
├── analysis/
│   └── purchase_prediction_analysis.R   # full pipeline: EDA -> models -> evaluation
├── data/
│   └── ecommerce_customer_behavior.csv
├── outputs/                             # generated on run: EDA & model plots (PNG)
└── README.md
```

## Methodology

1. **Feature selection** — subset to `purchased` plus 7 behavioral/session predictors; drop rows with missing values; encode categoricals as factors.
2. **EDA** — price distribution, time-on-site density by purchase outcome, purchases by rating, time-on-site boxplot by outcome.
3. **Train/test split** — 80/20.
4. **Models**
   - Logistic regression
   - CART (`rpart`, `cp = 0.0003`)
   - Random forest (`randomForest`, 500 trees)
5. **Evaluation** — ROC AUC for all three models on the same plot, misclassification rate at a 0.3 cutoff, and a cutoff-sensitivity sweep (0.2–0.6) to see how each model's error rate responds to threshold choice.

## How to run

```bash
git clone <this-repo-url>
cd ecommerce-purchase-prediction
Rscript analysis/purchase_prediction_analysis.R
```

Requires R (≥ 4.0) with packages: `dplyr`, `ggplot2`, `ROCR`, `rpart`, `rpart.plot`, `randomForest` — the script installs any that are missing on first run. Plots are written to `outputs/`.

## Results

| Model | Test AUC | Test Misclassification (cutoff 0.3) |
|---|---|---|
| Logistic regression | 0.561 | 0.231 |
| CART | 0.563 | 0.230 |
| Random forest | 0.526 | 0.259 |

Logistic regression and CART perform comparably and modestly outperform random forest here. All three show real (if modest) predictive lift over chance — unlike a pure-noise baseline, session behavior (particularly time on site and pages viewed) carries some signal about purchase intent, but no single model dominates and none reaches strong discriminative power (AUC > 0.7) on these features alone. Adding product-category or price-sensitivity interactions would be a natural next step to improve separation.

## License

[MIT](LICENSE)
