"""Streamlit Dashboard (publication-grade).

This dashboard provides:
1) Data Explorer
2) Model Training (baselines + behavior model)
3) Evaluation Dashboard (metrics + calibration)
4) Interpretability View (SHAP/LIME hooks)
5) Business Intelligence (aspect breakdown + revenue impact)
6) Prediction Interface (single + batch)

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.data.data_loader import load_reviews_csv
from src.data.data_preprocessing import clean_dataframe, detect_language_fast, translate_to_english
from src.data.data_quality import profile_dataset, remove_duplicates
from src.features.aspect_extraction import AspectExtractor
from src.features.advanced_features import AdvancedFeatureEngineer
from src.models.sentiment_model import get_sentiment_pipeline, predict_sentiment, predict_sentiment_batched
from src.business.revenue_impact import estimate_revenue_impact
from src.business.recommendations import top_negative_aspects
from src.utils.logger import get_logger
log = get_logger(__name__)

@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load CSV/TSV bytes into a DataFrame, remove duplicates, and clean."""
    if filename.lower().endswith(".tsv"):
        df = pd.read_csv(io.BytesIO(file_bytes), sep="\t")
    else:
        df = pd.read_csv(io.BytesIO(file_bytes))

    df = remove_duplicates(df, text_col="review_text")
    df = clean_dataframe(df)
    return df




@st.cache_resource
def _sentiment_pipe():
    return get_sentiment_pipeline()


@st.cache_resource
def _absa():
    return AspectExtractor(use_embedding_similarity=False)



@st.cache_resource
def _feature_engineer():
    return AdvancedFeatureEngineer()


def _sidebar_config():
    st.sidebar.header("⚙️ Settings")
    translate = st.sidebar.checkbox("Translate non-English (best effort)", value=False)
    max_rows = st.sidebar.slider("Max rows to process (speed)", 500, 10000, 3000, step=500)
    return {"translate": translate, "max_rows": max_rows}


def view_data_explorer(df: pd.DataFrame):
    st.subheader("1) Data Explorer")
    st.write("Shape:", df.shape)
    st.dataframe(df.head(200), use_container_width=True)

    rep = profile_dataset(df).__dict__
    st.json(rep)

    if "rating" in df.columns:
        st.bar_chart(df["rating"].value_counts().sort_index())


def compute_sentiment_and_absa(df: pd.DataFrame, max_rows: int):
    pipe = _sentiment_pipe()
    absa = _absa()

    work = df.head(max_rows).copy()

    labels, conf = predict_sentiment(pipe, work["review_text"].tolist())
    work["sentiment_label"] = labels
    work["sentiment_confidence"] = conf

    aspect_results = [absa.extract(t, sentiment_pipe=pipe) for t in work["review_text"].tolist()]
    all_aspects = sorted({a for r in aspect_results for a in r.aspect_sentiment.keys()})
    for a in all_aspects:
        work[f"aspect_sent_{a}"] = [r.aspect_sentiment.get(a, 0.0) for r in aspect_results]
        work[f"aspect_conf_{a}"] = [r.aspect_confidence.get(a, 0.0) for r in aspect_results]
    return work


def view_training(df: pd.DataFrame) -> None:
    st.subheader("2) Model Training")
    st.info(
        "This dashboard focuses on reproducible pipelines. Full experimental suite is available via CLI (MLflow)."
    )

    cfg = st.session_state.get("cfg", {})
    max_rows = int(cfg.get("max_rows", 3000))
    batch_size = int(cfg.get("inference_batch_size", 32))

    st.caption(f"Max rows (for dashboard speed): {max_rows:,} | Inference batch size: {batch_size}")

    if st.button("Compute sentiment + ABSA features"):
        # 1) Limit rows for dashboard speed
        work_df = df.head(max_rows).copy()
        st.write(f"Processing {len(work_df):,} reviews…")

        progress = st.progress(0.0)

        # 2) Load sentiment pipeline once
        pipe = get_sentiment_pipeline(
            model_name=cfg.get("sentiment_model_name", "cardiffnlp/twitter-roberta-base-sentiment-latest"),
            max_length=int(cfg.get("sentiment_max_length", 256)),
        )

        # 3) Batched sentiment inference
        labels, scores = predict_sentiment_batched(
            pipe,
            work_df["review_text"].astype(str).tolist(),
            batch_size=batch_size,
            progress_cb=lambda p: progress.progress(min(0.7, float(p) * 0.7)),  # map to first 70%
        )

        work_df["sentiment_label"] = labels
        work_df["sentiment_confidence"] = scores

        # 4) ABSA aspect extraction
        st.write("✅ Sentiment done. Starting ABSA…")
        absa = _absa()

        texts = work_df["review_text"].astype(str).tolist()
        aspect_results = []
        n = len(texts)

        for i, t in enumerate(texts):
            aspect_results.append(absa.extract(t, sentiment_pipe=pipe))

            # Update progress occasionally
            if i % 50 == 0 or i == n - 1:
                progress.progress(min(1.0, 0.7 + 0.3 * ((i + 1) / n)))  # remaining 30%

        all_aspects = sorted({a for r in aspect_results for a in r.aspect_sentiment.keys()})
        for a in all_aspects:
            work_df[f"aspect_sent_{a}"] = [r.aspect_sentiment.get(a, 0.0) for r in aspect_results]
            work_df[f"aspect_conf_{a}"] = [r.aspect_confidence.get(a, 0.0) for r in aspect_results]

        work_df["aspects_raw"] = [r.aspects for r in aspect_results]

        # 5) Advanced features (drop overlaps BEFORE concat)
        st.write("✅ ABSA done. Starting advanced features…")
        fe = _feature_engineer()

        feats = fe.transform(work_df).reset_index(drop=True)
        base = work_df.reset_index(drop=True)

        # ✅ Prevent duplicate columns when concatenating
        overlap = sorted(set(base.columns) & set(feats.columns))
        if overlap:
            st.warning(f"Overlapping feature columns dropped from advanced features: {overlap}")
            feats = feats.drop(columns=overlap)

        work_df = pd.concat([base, feats], axis=1)

        # ✅ Final safety: ensure no duplicate column names exist
        if work_df.columns.duplicated().any():
            dupes = work_df.columns[work_df.columns.duplicated()].tolist()
            st.warning(f"Duplicate columns detected and removed: {dupes}")
            work_df = work_df.loc[:, ~work_df.columns.duplicated()].copy()

        # 6) Store for other tabs
        st.session_state["work_df"] = work_df

        progress.progress(1.0)
        st.write("✅ Features done.")
        st.success("Computed sentiment + ABSA + advanced features.")

        



def view_evaluation(work_df: pd.DataFrame):
    st.subheader("3) Evaluation Dashboard")
    if "doRecommend" not in work_df.columns:
        st.warning("No doRecommend column found; evaluation limited to sentiment/ABSA analytics.")
        return
    st.write("Label distribution:")
    y = work_df["doRecommend"].astype(str).str.lower().map({"true": 1, "false": 0, "yes": 1, "no": 0}).fillna(0).astype(int)
    st.bar_chart(y.value_counts())

    st.write("Note: full metrics + statistical testing available via `src/utils/experiment_manager.py`.")


def view_interpretability(work_df: pd.DataFrame) -> None:
    """
    Streamlit-safe interpretability view.

    Philosophy:
    - Keep Streamlit fast and responsive (no heavy SHAP/LIME computation here).
    - Provide lightweight, useful previews + clear pointers to offline research-grade tools.

    Expected columns (if available):
    - sentiment_label, sentiment_confidence
    - aspect_sent_<aspect>, aspect_conf_<aspect>
    """
    st.subheader("4) Interpretability")

    st.info(
        "This Streamlit tab shows **lightweight interpretability previews** to keep the UI responsive.\n\n"
        "Full research-grade explainability (SHAP/LIME/counterfactuals/attention) lives in "
        "`src/interpretability/` and is best run via CLI/notebooks for heavier plots and large datasets."
    )

    # --- Quick health checks ---
    n_rows = len(work_df)
    st.caption(f"Rows available for interpretation: {n_rows:,}")

    # --- 1) Overall sentiment distribution (fast & informative) ---
    if "sentiment_label" in work_df.columns:
        st.markdown("### Overall sentiment distribution")
        st.bar_chart(work_df["sentiment_label"].value_counts())
    else:
        st.warning("`sentiment_label` not found. Run **Compute sentiment + ABSA features** first.")

    # --- 2) Confidence distribution ---
    if "sentiment_confidence" in work_df.columns:
        st.markdown("### Sentiment confidence (model certainty)")
        st.write(work_df["sentiment_confidence"].describe())
        # Optional quick histogram-like view using bins
        try:
            bins = pd.cut(work_df["sentiment_confidence"], bins=10)
            st.bar_chart(bins.value_counts().sort_index())
        except Exception:
            pass  # keep UI robust

    # --- 3) Aspect-level overview (mean sentiment + prevalence) ---
    aspect_sent_cols = [c for c in work_df.columns if c.startswith("aspect_sent_")]
    aspect_conf_cols = [c for c in work_df.columns if c.startswith("aspect_conf_")]

    if aspect_sent_cols:
        st.markdown("### Aspect-level interpretability (ABSA preview)")

        # Build a compact summary table
        rows = []
        for col in aspect_sent_cols:
            aspect = col.replace("aspect_sent_", "")
            sent = pd.to_numeric(work_df[col], errors="coerce")
            prevalence = float((sent.fillna(0) != 0).mean())  # non-zero indicates detected
            mean_sent = float(sent.fillna(0).mean())

            conf_col = f"aspect_conf_{aspect}"
            if conf_col in work_df.columns:
                conf = pd.to_numeric(work_df[conf_col], errors="coerce")
                mean_conf = float(conf.fillna(0).mean())
            else:
                mean_conf = float("nan")

            rows.append(
                {
                    "aspect": aspect,
                    "prevalence_%": round(prevalence * 100, 2),
                    "mean_sentiment": round(mean_sent, 4),
                    "mean_confidence": None if np.isnan(mean_conf) else round(mean_conf, 4),
                }
            )

        summary = pd.DataFrame(rows).sort_values(["prevalence_%", "mean_sentiment"], ascending=[False, True])
        st.dataframe(summary, use_container_width=True)

        # Let user inspect one aspect
        chosen = st.selectbox("Inspect a single aspect", summary["aspect"].tolist())
        chosen_col = f"aspect_sent_{chosen}"
        st.write(f"Distribution for **{chosen}** sentiment values:")
        try:
            st.bar_chart(work_df[chosen_col].fillna(0).value_counts().sort_index())
        except Exception:
            st.write(work_df[chosen_col].describe())

    else:
        st.info("No `aspect_sent_*` columns found yet. Compute ABSA features first.")

    # --- 4) Example cases (fast and very useful) ---
    st.markdown("### Example cases (quick inspection)")
    col_text = "review_text" if "review_text" in work_df.columns else None
    if col_text:
        # Show a few low-confidence items if available, otherwise random
        if "sentiment_confidence" in work_df.columns:
            sample = work_df.sort_values("sentiment_confidence", ascending=True).head(5)
            st.caption("Showing 5 lowest-confidence predictions (often most informative to inspect).")
        else:
            sample = work_df.sample(min(5, n_rows), random_state=42)
            st.caption("Showing 5 random samples.")

        for idx, row in sample.iterrows():
            with st.expander(f"Row {idx}"):
                st.write(row[col_text])
                if "sentiment_label" in row:
                    st.write("Sentiment:", row.get("sentiment_label"))
                if "sentiment_confidence" in row:
                    st.write("Confidence:", float(row.get("sentiment_confidence", 0.0)))

                # Show aspects if present
                if aspect_sent_cols:
                    aspect_preview = {c.replace("aspect_sent_", ""): float(row.get(c, 0.0) or 0.0) for c in aspect_sent_cols}
                    # keep only non-zero aspects for readability
                    aspect_preview = {k: v for k, v in aspect_preview.items() if abs(v) > 1e-9}
                    st.write("Aspects (non-zero):", aspect_preview if aspect_preview else "None detected")
    else:
        st.info("`review_text` column not found. Cannot show example cases.")

    # --- 5) Pointers to research-grade interpretability code ---
    st.markdown("### Research-grade explainability (offline)")
    st.code(
        "from src.interpretability.shap_analysis import explain_tree_global\n"
        "from src.interpretability.lime_analysis import explain_text_local\n"
        "from src.interpretability.counterfactuals import generate_counterfactuals\n",
        language="python",
    )

    st.caption(
        "Tip: Run heavy explainability on a **small stratified sample** first (e.g., 1k rows) "
        "and save plots to disk/MLflow artifacts."
    )



def view_business_intel(work_df: pd.DataFrame):
    st.subheader("5) Business Intelligence")
    neg = top_negative_aspects(work_df, k=6)
    st.write("Top negative aspects (mean sentiment):")
    st.dataframe(neg, use_container_width=True)

    impact = estimate_revenue_impact(work_df, revenue_per_recommend=st.slider("Revenue per recommendation ($)", 1.0, 200.0, 20.0))
    st.write("Estimated revenue loss by aspect:")
    st.dataframe(impact, use_container_width=True)


def view_prediction() -> None:
    st.subheader("6) Prediction Interface")
    text = st.text_area("Enter a review text", height=180)

    if st.button("Predict"):
        if not text.strip():
            st.warning("Please enter a review.")
            return

        pipe = _sentiment_pipe()
        absa = _absa()

        # Sentiment
        res = pipe(text)[0]
        st.write("### Overall Sentiment")
        st.json(res)

        # ABSA
        st.write("### Aspect Breakdown (ABSA)")
        ar = absa.extract(text, sentiment_pipe=pipe)
        st.json(
            {
                "aspects": ar.aspects,
                "aspect_sentiment": ar.aspect_sentiment,
                "aspect_confidence": ar.aspect_confidence,
            }
        )



def main():
    st.set_page_config(page_title="ABSA + Multi-Task Sentiment Project", layout="wide")
    st.title("📊 ABSA + Consumer Behavior Prediction Dashboard (Research-Grade)")

    cfg = _sidebar_config()
    st.session_state["cfg"] = cfg

    uploaded = st.file_uploader("Upload review CSV/TSV", type=["csv", "tsv"])
    if uploaded is None:
        st.stop()

    df = load_and_clean(uploaded.getvalue(), uploaded.name)

    max_rows = int(cfg.get("max_rows", 3000))
    if len(df) > max_rows:
        df = df.head(max_rows).copy()
        st.info(f"Using first {len(df):,} rows for dashboard speed (max_rows={max_rows}).")

    st.success(f"Loaded dataset with {len(df):,} rows")
    # --- Add doRecommend proxy label if missing (enables evaluation) ---
    if "doRecommend" not in df.columns and "rating" in df.columns:
        df["doRecommend"] = (pd.to_numeric(df["rating"], errors="coerce") >= 4).astype(int)
        st.info("Created proxy label doRecommend = 1 if rating >= 4 (for evaluation metrics).")


    # Optional translation (safe sampling)
    if cfg.get("translate", False):
        sample_texts = df["review_text"].astype(str).head(200).tolist()
        is_english = detect_language_fast(sample_texts, sample_size=5)
        if not is_english:
            st.warning("Non-English detected; translating (best effort).")
            df["review_text"] = translate_to_english(df["review_text"].astype(str).tolist())
        else:
            st.info("English detected; translation skipped.")

    tabs = st.tabs(["Data Explorer", "Model Training", "Evaluation", "Interpretability", "Business Intel", "Predict"])

    with tabs[0]:
        view_data_explorer(df)

    with tabs[1]:
        view_training(df)

    work_df = st.session_state.get("work_df")

    with tabs[2]:
        if work_df is not None:
            view_evaluation(work_df)
        else:
            st.info("Compute sentiment + ABSA features first.")

    with tabs[3]:
        if work_df is not None:
            view_interpretability(work_df)
        else:
            st.info("Compute sentiment + ABSA features first.")

    with tabs[4]:
        if work_df is not None:
            view_business_intel(work_df)
        else:
            st.info("Compute sentiment + ABSA features first.")

    with tabs[5]:
        view_prediction()



if __name__ == "__main__":
    main()
