# Sentiment Analysis & Consumer Behavior Prediction (Publication-Grade)

This repo upgrades a master's thesis prototype into a reproducible, research-grade system centered on
**Aspect-Based Sentiment Analysis (ABSA)** and **multi-task learning** for predicting consumer behavior.

## Key Contributions
- **Aspect extraction + aspect sentiment** (ABSA) with a hybrid approach:
  1) noun-phrase mining (spaCy) + 2) seed-lexicon aspect mapping + 3) optional topic modeling
- **Multi-task Transformer**: shared encoder with heads for:
  - Aspect extraction (token classification)
  - Overall sentiment (sequence classification)
  - Purchase recommendation prediction (classification with auxiliary engineered features)
- **Rigorous experimental framework**: baselines, cross-validation, temporal split, statistical tests.
- **Interpretability**: attention visualization, SHAP, LIME, counterfactuals.
- **Business intelligence**: aspect-level root causes, ROI/revenue impact estimation, segmentation.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Data Format
Expected columns:
- `review_text` (required)
- `rating` (1-5, required for some baselines)
- optional: `doRecommend`, `verified_purchase`, `helpful_votes`, `total_votes`, `timestamp`, `reviewer_id`, `product_id`

## Run Streamlit Dashboard
```bash
streamlit run app.py
```

## Run Experiments (CLI)
```bash
python -m src.utils.experiment_manager --config config/config.yaml --data data/raw/reviews.csv
```

## Notes / TODO
- Translation is optional and uses `googletrans`; it can be rate-limited. For production, replace with a paid API.
- Some modules include TODOs for setting paths / credentials and for adding dataset-specific label mappings.
