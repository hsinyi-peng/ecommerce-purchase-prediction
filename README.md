# E-commerce Purchase Prediction

A conversion-rate diagnostic: does session-level behavior (time on site, pages viewed, discounting, device, traffic source) actually explain whether a browsing session ends in a purchase? Tested with logistic regression, a classification tree (CART), and a random forest.

## The Business Question

**North Star metric:** session-to-purchase conversion rate.

Most e-commerce sessions don't convert, and when conversion drops, the first thing a growth team wants to know is *which lever moved*. This project treats that as a testable question: take the behavioral signals a team could plausibly act on or attribute a drop to — time on site, pages viewed, discount depth, device, new-vs-returning visitor, acquisition channel — and check whether they actually predict `purchased` (1) vs. `not purchased` (0) on held-out sessions.

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
   - CART (`rpart`, `cp = 0.002` — pruned deliberately conservative so any split it does find is a robust pattern, not noise fit to a weak signal)
   - Random forest (`randomForest`, 500 trees)
5. **Evaluation** — ROC AUC for all three models on the same plot, misclassification rate at a 0.3 cutoff, and a cutoff-sensitivity sweep (0.2–0.6) to see how each model's error rate responds to threshold choice.

## Findings

| Model | Test AUC | Test Misclassification (cutoff 0.3) |
|---|---|---|
| Logistic regression | 0.561 | 0.231 |
| CART (pruned) | 0.500 | 0.229 |
| Random forest | 0.526 | 0.259 |

**The headline finding is the CART result, not the logistic one.** Pruned conservatively (so it only keeps a split if the pattern is strong enough to be worth trusting), the tree finds *no split at all* — AUC lands exactly at 0.500, chance level. That's a clean, honest answer to the business question: on this feature set, there is no robust decision rule of the form "sessions with X behavior convert more." Logistic regression and random forest do edge out chance (AUC 0.56 and 0.53), so there's some real signal in time-on-site and pages-viewed — just not enough to act on with confidence, and not enough for any model to reach strong discrimination (AUC > 0.7).

**What this means for root-cause diagnosis:** if conversion rate drops, this feature set alone can't tell you why. Time on site and pages viewed carry a little signal, but discount depth, device, and channel don't move the needle much in isolation. The practical next step isn't tuning these models further — it's instrumenting more of the funnel (price relative to competitors, cart/checkout friction, prior purchase history, promotional timing) so there's an actual lever to pull when the metric moves.

## How to run

```bash
git clone <this-repo-url>
cd ecommerce-purchase-prediction
Rscript analysis/purchase_prediction_analysis.R
```

Requires R (≥ 4.0) with packages: `dplyr`, `ggplot2`, `ROCR`, `rpart`, `rpart.plot`, `randomForest` — the script installs any that are missing on first run. Plots are written to `outputs/`.

## License

[MIT](LICENSE)
