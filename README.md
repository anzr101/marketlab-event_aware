# Bridging the Prediction–Profitability Gap

### A Regime-Aware Gating Framework for Machine-Learning Trading in Indian Equity Markets

---

## Overview

This repository contains the full implementation and research work behind a key finding:

> **Machine learning models can achieve near-perfect predictive accuracy and still lose money in real markets.**

This project investigates that paradox and proposes a system-level solution.


* 📊 Data: NSE (2004–2024), 50 stocks
* 🤖 Models: 1,386 trained models across ML + Deep Learning

---

## Key Insight

Most ML trading systems optimize for:

* Mean Squared Error (MSE)
* R² score
* Directional accuracy

But real-world trading depends on:

* Risk-adjusted returns
* Drawdown control
* Regime awareness

### Core Finding

* Best model: **R² ≈ 0.998**
* Trading performance: **−14.02% annualized return**

➡️ High accuracy ≠ profitability

---

## Solution: Regime-Aware Gating Framework

We propose a **two-layer architecture**:

### 1. ML Signal Layer

* Generates predictions using classical ML + deep learning models
* Outputs position sizing signals

### 2. Event-Driven Gating Layer

* Detects high-impact real-world events
* Dynamically reduces exposure
* Overrides ML decisions during regime shifts

---

## Results

| Strategy           | Annual Return | Sharpe    | Max Drawdown |
| ------------------ | ------------- | --------- | ------------ |
| Buy & Hold         | −0.85%        | −0.28     | −26.97%      |
| ML Model           | −14.02%       | −0.89     | −47.14%      |
| Event-Gated System | **+9.92%**    | **+0.24** | **−22.95%**  |

### Impact

* 🔺 Return improvement: **+23.94 percentage points**
* 🔻 Drawdown reduction: **51.3%**
* 📈 Portfolio: ₹1,00,000 → ₹1,27,300

---

## Methodology

### Dataset

* 20 years of NSE data (2004–2024)
* 50 large-cap stocks
* 6 sectors
* Daily OHLCV

---

### Feature Engineering

* 325 technical indicators:

  * Moving averages (SMA, EMA, VWAP)
  * Momentum (RSI, MACD, ROC)
  * Volatility (ATR, Bollinger Bands)
  * Volume indicators (OBV, A/D)
  * Trend indicators (ADX, Ichimoku)

---

### Model Training

* **1,386 total models**

  * 25+ classical ML algorithms:

    * Linear, Ridge, Lasso
    * Random Forest, XGBoost, LightGBM
    * SVM, KNN, Ensemble models
  * Deep learning:

    * LSTM, GRU, BiLSTM
    * CNN-LSTM
    * Transformer

---

## Critical Experiments (Ablation Studies)

### Ablation 1: Target Shift (Price → Returns)

* Median R²: **0.981 → −0.152**
* **100% models failed (negative R²)**

➡️ Reveals models were exploiting **autoregressive leakage**, not true prediction

---

### Ablation 2: Feature Removal

* Removing short-lag features had **almost no effect**
* Leakage is **distributed across feature space**

➡️ Problem is structural, not localized

---

## Event Intelligence System

### Event Categories

* Geopolitical
* Economic Policy
* Corporate
* Regulatory
* Natural Disaster
* Technological

---

### Pipeline

1. News ingestion (APIs, RSS, feeds)
2. Keyword-based scoring
3. FinBERT sentiment analysis
4. Zero-shot classification (fallback)
5. Impact scoring (0–10 scale)

---

### Action Mapping

| Impact Score | Action                            |
| ------------ | --------------------------------- |
| 9–10         | Exit all positions                |
| 7–8          | Reduce to 30% (defensive sectors) |
| 5–6          | Reduce to 60%                     |
| 3–4          | Monitor                           |
| 0–2          | Normal trading                    |

---

## Case Studies

### Real-world events tested:

* Russia–Ukraine war (2022)
* Fed rate hikes (2022)
* Silicon Valley Bank collapse (2023)
* US–Iran conflict (2024)

### Result:

* Loss reduction up to **94% per event**
* Total savings: **₹21,870 across 4 events**

---

## Architecture

```id="3o71x6"
ML Models → Predictions → Position Sizing
                          ↓
                 Event Detection System
                          ↓
                 Gating Function (α)
                          ↓
                 Final Trading Decisions
```

---

## Why This Matters

### Problem in current ML-finance systems:

* Over-reliance on historical patterns
* No adaptation to regime shifts
* Misleading evaluation metrics

---

### Contribution:

* Formalizes **prediction–profitability gap**
* Introduces **risk-first architecture**
* Demonstrates importance of **external information (events)**

---




---

## Author

**Anzar Shaikh**
B.E. Artificial Intelligence and Data Science
University of Mumbai

* ORCID: 0009-0005-7844-5792
* Email: [anzarsk098@gmail.com](mailto:anzarsk098@gmail.com)

---

## Citation

If you use this work:

```id="i6xwls"
@article{shaikh2026prediction,
  title={Bridging the Prediction–Profitability Gap},
  author={Shaikh, Anzar},
  year={2026}
}
```

---

## Final Takeaway

> Improving *when to trust a model* is more important than improving the model itself.
