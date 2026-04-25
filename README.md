# MarketLab — Event-Aware Algorithmic Trading for Indian Equity Markets

> Code and supplementary materials for the paper *Bridging the Prediction–Profitability Gap: A Regime-Aware Gating Framework for Machine-Learning Trading in Indian Equity Markets* (Anzar Shaikh, 2026).

[![Paper Viewer](https://img.shields.io/badge/Paper-View_Online-blue)](https://marketlab-paper.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0005--7844--5792-green)](https://orcid.org/0009-0005-7844-5792)

---

## TL;DR

We trained 1,386 ML models on twenty years of NSE data. The best model reported $R^2 = 0.9986$ but lost **−14.02%** annually trading the 2022–2024 crisis period — losing to passive buy-and-hold by 13 percentage points.

We then built a **regime-aware gating framework**: a two-layer architecture where ML generates positions and a separate event-classification system *gates* exposure during detected crises (Russia–Ukraine, Fed 75bp, SVB, US–Iran). Same ML predictions, with the gate applied → **+9.92%** annualized return, Sharpe **−0.89 → +0.24**, max drawdown reduced 51.3%, $p = 1.9 \times 10^{-13}$.

Two ablation studies show the headline $R^2$ was a feature-engineering artifact (autoregressive leakage). The gating framework works *despite* — actually *because of* — the underlying ML being weak.

---

## What's in this repo

```
marketlab-event_aware/
├── README.md                    ← you are here
├── LICENSE                      ← MIT
├── requirements.txt             ← Python dependencies
├── .gitignore
│
├── paper/
│   ├── main.pdf                 ← the 21-page paper
│   └── main.tex                 ← LaTeX source
│
├── notebooks/                   ← Jupyter notebooks for each phase
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_classical_ml_training.ipynb
│   ├── 04_deep_learning_models.ipynb
│   ├── 05_baseline_backtest.ipynb
│   ├── 06_event_taxonomy.ipynb
│   ├── 07_finbert_classification.ipynb
│   ├── 08_intelligent_backtest.ipynb
│   ├── 09_case_studies.ipynb
│   └── 10_ablations.ipynb
│
├── src/                         ← reusable Python modules
│   ├── data/                    ← yfinance fetcher, NSE calendar, cleaning
│   ├── features/                ← 325 technical-indicator pipeline
│   ├── models/                  ← 25 classical + 5 deep architectures
│   ├── events/                  ← FinBERT classifier, taxonomy, scoring
│   ├── backtest/                ← portfolio simulation, risk metrics
│   └── utils/
│
├── figures/                     ← 14 publication figures (PDF + PNG)
│
├── results/
│   ├── results_day1/            ← classical ML R² per stock
│   ├── results_day2/            ← deep learning results
│   ├── results_day3/            ← feature importance (SHAP)
│   ├── results_day3_5/
│   ├── results_day4/            ← initial 5-stock backtest
│   ├── results_day5/            ← event taxonomy + historical events
│   ├── results_day6/            ← prototype intelligent backtest
│   ├── results_day7/            ← validation summary
│   ├── results_day10/           ← 40-event classification (90% accuracy)
│   ├── results_day11/           ← full backtest, risk metrics, stat tests
│   ├── results_day12/           ← production architecture spec
│   ├── results_day13_5/         ← case studies, economic impact
│   └── results_ablation/        ← Ablation 1 & 2 outputs
│
└── data/
    └── samples/                 ← 2–3 example stock CSVs only
                                   (full 20-year data not committed; fetch via yfinance)
```

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/anzr101/marketlab-event_aware.git
cd marketlab-event_aware
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Fetch the data

The full 50-stock × 20-year dataset is not committed (size). Re-fetch with:

```bash
python src/data/fetch_nse.py --tickers configs/nifty50.txt --start 2004-01-01 --end 2024-12-31 --out data/raw/
```

This pulls from `yfinance` and takes ~10 minutes.

### 3. Reproduce the headline result

```bash
# Classical ML training (this is the slow step: ~2-4 hours on a laptop)
python src/models/train_classical.py --config configs/baseline.yaml

# Run the event-gated backtest
python src/backtest/run_intelligent_backtest.py --config configs/backtest.yaml

# Generate paper figures
python src/utils/make_figures.py
```

Or open the notebooks in order — they walk through every phase.

### 4. Reproduce the ablation studies

```bash
python src/models/ablation1_return_targets.py
python src/models/ablation2_feature_ablation.py
```

---

## Headline results

| Strategy | Annual Return | Sharpe | Sortino | Max DD | End Value |
|---|---:|---:|---:|---:|---:|
| Buy & Hold | −0.85% | −0.282 | −0.482 | −26.97% | ₹91,110 |
| ML Model (ungated) | **−14.02%** | **−0.889** | −1.477 | −47.14% | ₹60,270 |
| **Event-Gated** | **+9.92%** | **+0.237** | **+0.444** | **−22.95%** | **₹1,27,300** |

Statistical significance:
- Paired *t*-test (gated vs ML-only): $t = 7.49$, $p = 1.9 \times 10^{-13}$
- Diebold–Mariano: $DM = 2.21$, $p = 0.027$

### Crisis case studies (₹1,00,000 portfolio)

| Event | Date | Lead time | ML loss | Gated loss | Saved |
|---|---|---|---:|---:|---:|
| Russia invades Ukraine | 2022-02-24 | 3.75 h | −₹10,200 | −₹600 | ₹9,600 |
| Fed 75bp hike | 2022-06-15 | 15.25 h | −₹3,300 | −₹990 | ₹2,310 |
| Silicon Valley Bank | 2023-03-10 | 1.25 h | −₹7,660 | −₹1,860 | ₹5,800 |
| US–Iran / Red Sea | 2024-01-11 | 3.5 h | −₹4,960 | −₹800 | ₹4,160 |
| **Total** | | | **−₹26,120** | **−₹4,250** | **₹21,870** |

### Ablation studies (the integrity check)

**Ablation 1 — Return-target retrain.** Same models, same features, same CV splits — only the prediction target changes from price level to next-day return.

| Statistic | Level target | Return target |
|---|---:|---:|
| Median $R^2$ | 0.981 | **−0.152** |
| Best $R^2$ | 0.998 | −0.0004 |
| % with $R^2 < 0$ | 29.8% | **100.0%** |

**Ablation 2 — Feature ablation.** Removing 10 short-lag features (`ema_3`, `sma_3`, `ema_5`, `sma_5`, `vwap_5`, `close_lag_1`, `high_lag_1`, `low_lag_1`, `open_lag_1`, `tenkan_sen`) from the 325-feature set: linear-model median $R^2$ changes from 0.9907 to 0.9896 — essentially zero. The autoregressive leak is **pervasive** across the feature category, not localized.

---

## The framework, formally

We define the prediction–profitability gap as the Sharpe-ratio shortfall of the MSE-optimal forecaster relative to the Sharpe-optimal trading policy:

$$
G(f) = \mathrm{Sharpe}\!\left(\pi_{f^{\star}_{\mathrm{MSE}}}\right) - \mathrm{Sharpe}\!\left(\pi^{\star}_{\mathrm{Sharpe}}\right)
$$

Under non-stationary regimes, this gap is provably non-vanishing (Proposition 1 in the paper).

The **regime-aware gating framework** introduces an attenuation function $\alpha(R_t) \in [0, 1]$ over a regime detector $R_t$, with gated weights:

$$
w^{\mathrm{gated}}_{i,t} = \alpha(R_t) \cdot w^{\mathrm{ML}}_{i,t} \cdot \mathbf{1}\{i \in \mathcal{D}(R_t)\}
$$

where $\mathcal{D}(R_t)$ restricts the universe to defensive sectors during high-impact regimes. The gating layer is **non-expansive** (cannot increase exposure), **deterministic** given $R_t$, and **interpretable** (every trade traceable to a categorized event) — properties required for SEBI audit compliance.

We instantiate $R_t$ via a six-category event-classification pipeline (FinBERT + keywords + zero-shot BART) achieving 90.0% accuracy on 40 validated events.

---

## Requirements

- Python 3.10+
- See `requirements.txt` for full dependency list
- Core: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `tensorflow`, `transformers`, `yfinance`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`

GPU optional (used for FinBERT inference and deep-learning models). FinBERT runs fine on CPU.

---

## Citation

If this work is useful, please cite:

```bibtex
@misc{shaikh2026marketlab,
  author       = {Anzar Shaikh},
  title        = {Bridging the Prediction-Profitability Gap:
                  A Regime-Aware Gating Framework for Machine-Learning
                  Trading in Indian Equity Markets},
  year         = {2026},
  howpublished = {\url{https://github.com/anzr101/marketlab-event_aware}},
  note         = {Preprint}
}
```

---

## Limitations

This work has known limitations explicitly documented in the paper (Section 9):

1. NSE-only — generalization to other emerging markets not yet tested.
2. 782 trading days is substantial but short relative to a full equity cycle.
3. Decomposition ablation (event-only vs ML+event) is deferred to a companion study.
4. Backtest uses archived headlines, not a live-feed snapshot — minor headline-wording drift possible.
5. Gating layer most valuable in eventful periods; contributes little in quiet markets.

---

## License

MIT License — see [LICENSE](LICENSE).

Paper text and figures are CC BY 4.0.

---

## Contact

**Anzar Shaikh** — B.E. Artificial Intelligence and Data Science, University of Mumbai
- Email: anzarsk098@gmail.com
- ORCID: [0009-0005-7844-5792](https://orcid.org/0009-0005-7844-5792)
- Paper viewer: [marketlab-paper.streamlit.app](https://marketlab-paper.streamlit.app)

For media enquiries: subject *MarketLab — Press*

---

## Acknowledgments

Open-source maintainers of `scikit-learn`, `TensorFlow`, `XGBoost`, `LightGBM`, the HuggingFace `transformers` library, and the ProsusAI FinBERT contributors — without whose work this study would not have been possible.
