"""
StyleSense ML Dashboard
========================
A Streamlit dashboard that loads the trained scikit-learn + spaCy pipeline
and provides four interactive views:

  1. Dataset Overview    — data distributions, class balance, category stats
  2. Model Performance   — real metrics, confusion matrix, per-class report
  3. Feature Analysis    — live logistic regression coefficients from the model
  4. Live Prediction     — runs the actual pipeline on any review you type

Usage:
    # From the project directory:
    streamlit run dashboard.py

Requirements:
    pip install streamlit scikit-learn spacy pandas numpy matplotlib seaborn
"""

import pickle
import re
import sys
import warnings

# Import pipeline classes so pickle can deserialize pipeline_payload.pkl
# pipeline_model.py must be in the same directory as dashboard.py
sys.path.insert(0, ".")
from pipeline_model import (
    get_feature_names,  # noqa: F401  (needed for pickle)
    FullFeatureUnion,
    TextCombiner,
    load_nlp,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StyleSense ML Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Load pipeline payload (cached so it only runs once)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading trained pipeline…")
def load_payload():
    with open("pipeline_payload.pkl", "rb") as f:
        return pickle.load(f)


payload   = load_payload()
pipeline  = payload["pipeline"]
X_test    = payload["X_test"]
y_test    = payload["y_test"]
y_pred    = payload["y_pred"]
y_proba   = payload["y_proba"]
df        = payload["df"]
feat_meta = payload["feature_names"]
# NUMERICAL_FEATURES and CATEGORICAL_FEATURES are imported from pipeline_model

# ──────────────────────────────────────────────────────────────────────────────
# Helper — extract feature importance from the fitted pipeline
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_feature_importance():
    from pipeline_model import get_feature_names
    all_names = get_feature_names(pipeline)
    coefs     = pipeline.named_steps["clf"].coef_[0]
    return pd.DataFrame({"feature": all_names, "coefficient": coefs}).sort_values(
        "coefficient", key=abs, ascending=False
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛍️ StyleSense")
    st.markdown("**ML Pipeline Dashboard**")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Dataset Overview", "🎯 Model Performance",
         "🔍 Feature Analysis", "✨ Live Prediction"],
    )
    st.markdown("---")
    st.markdown("**Pipeline summary**")
    st.markdown(f"- Samples: `{len(df):,}`")
    st.markdown(f"- Train: `{len(X_test) * 9:,}` · Test: `{len(X_test):,}`")
    st.markdown(f"- Features: Numerical + Categorical + TF-IDF + spaCy NLP")
    st.markdown(f"- Classifier: Logistic Regression (`C=10`, balanced)")
    st.markdown("---")
    st.caption("Model loaded from `pipeline_payload.pkl`")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Dataset Overview
# ──────────────────────────────────────────────────────────────────────────────
if page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    st.markdown("Exploratory analysis of the 18,442-review women's clothing dataset.")

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Reviews",    f"{len(df):,}")
    c2.metric("Features",         "8")
    c3.metric("Recommend Rate",   f"{df['Recommended IND'].mean():.1%}")
    c4.metric("Departments",      df["Department Name"].nunique())
    c5.metric("Product Classes",  df["Class Name"].nunique())

    st.markdown("---")

    # Class distribution + Age distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Target Class Distribution")
        counts = df["Recommended IND"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.bar(
            ["Not Recommended", "Recommended"],
            counts.values,
            color=["#d4607a", "#2d7d72"],
            edgecolor="white",
            linewidth=1.2,
        )
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                    f"{val:,}\n({val/len(df):.1%})", ha="center", fontsize=9, fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_ylim(0, counts.max() * 1.2)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()
        st.caption("⚠️ Class imbalance (~82% positive) handled via `class_weight='balanced'`")

    with col2:
        st.subheader("Recommendation Rate by Age Group")
        df_age = df.copy()
        df_age["age_bin"] = pd.cut(df_age["Age"], bins=range(15, 85, 5))
        age_stats = df_age.groupby("age_bin", observed=True)["Recommended IND"].agg(["mean", "count"]).reset_index()
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(
            range(len(age_stats)),
            age_stats["mean"] * 100,
            color="#2d7d72", alpha=0.85, edgecolor="white"
        )
        ax.axhline(df["Recommended IND"].mean() * 100, color="#d4607a",
                   linestyle="--", linewidth=1.5, label="Overall avg")
        ax.set_xticks(range(len(age_stats)))
        ax.set_xticklabels(
            [str(b).replace("(","").replace("]","").replace(", ","-") for b in age_stats["age_bin"]],
            rotation=45, ha="right", fontsize=8
        )
        ax.set_ylabel("Recommendation Rate (%)")
        ax.set_ylim(60, 95)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Department + Division charts
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Recommendation Rate by Department")
        dept = (
            df.groupby("Department Name")["Recommended IND"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=True)
        )
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors = ["#2d7d72" if v >= df["Recommended IND"].mean() else "#d4607a"
                  for v in dept["mean"]]
        bars = ax.barh(dept.index, dept["mean"] * 100, color=colors, edgecolor="white")
        ax.axvline(df["Recommended IND"].mean() * 100, color="gray",
                   linestyle="--", linewidth=1, label="Overall avg")
        for bar, (_, row) in zip(bars, dept.iterrows()):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{row['mean']:.1%}  (n={row['count']:,})",
                    va="center", fontsize=8)
        ax.set_xlabel("Recommendation Rate (%)")
        ax.set_xlim(0, 105)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col4:
        st.subheader("Review Length Distribution")
        df_len = df.copy()
        df_len["word_count"] = df_len["Review Text"].fillna("").apply(lambda x: len(x.split()))
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for label, color, name in [(0, "#d4607a", "Not Recommended"), (1, "#2d7d72", "Recommended")]:
            subset = df_len[df_len["Recommended IND"] == label]["word_count"]
            ax.hist(subset, bins=40, alpha=0.6, color=color, label=name, edgecolor="white")
        ax.set_xlabel("Review Word Count")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 250)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Raw Data Sample")
    st.dataframe(df.sample(10, random_state=42).reset_index(drop=True), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Model Performance
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🎯 Model Performance":
    st.title("🎯 Model Performance")
    st.markdown("All metrics computed from the **real pipeline** on the 10% held-out test set.")

    # Metric KPIs — computed from actual pipeline predictions
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec  = recall_score(y_test, y_pred, average="weighted")
    f1   = f1_score(y_test, y_pred, average="weighted")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",          f"{acc:.2%}")
    c2.metric("Precision (wtd)",   f"{prec:.2%}")
    c3.metric("Recall (wtd)",      f"{rec:.2%}")
    c4.metric("F1-Score (wtd)",    f"{f1:.2%}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Not Recommended", "Recommended"]
        )
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Confusion Matrix — Test Set", fontsize=11, pad=12)
        st.pyplot(fig)
        plt.close()

        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        | | Count |
        |---|---|
        | ✅ True Negatives  | {tn:,} |
        | ✅ True Positives  | {tp:,} |
        | ❌ False Positives | {fp:,} |
        | ❌ False Negatives | {fn:,} |
        """)

    with col2:
        st.subheader("Per-Class Metrics")
        report = classification_report(
            y_test, y_pred,
            target_names=["Not Recommended", "Recommended"],
            output_dict=True
        )
        report_df = pd.DataFrame(report).T.drop("accuracy").round(3)
        st.dataframe(report_df, use_container_width=True)

        # Per-class bar chart
        fig, ax = plt.subplots(figsize=(5, 3.2))
        classes = ["Not Recommended", "Recommended"]
        metrics_list = ["precision", "recall", "f1-score"]
        x = np.arange(len(classes))
        width = 0.25
        colors_bar = ["#d4607a", "#2d7d72", "#7c6bb0"]

        for i, (metric, color) in enumerate(zip(metrics_list, colors_bar)):
            vals = [report[cls][metric] for cls in classes]
            ax.bar(x + i * width, vals, width, label=metric.title(), color=color,
                   edgecolor="white", linewidth=0.8)

        ax.set_xticks(x + width)
        ax.set_xticklabels(classes, fontsize=9)
        ax.set_ylim(0.5, 1.02)
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Prediction Confidence Distribution")
    st.markdown("Histogram of the model's predicted probability for the positive class across all test samples.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for ax, label, color, name in [
        (axes[0], 0, "#d4607a", "Not Recommended (actual)"),
        (axes[1], 1, "#2d7d72", "Recommended (actual)"),
    ]:
        mask = y_test.values == label
        probs = y_proba[mask, 1]
        ax.hist(probs, bins=30, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Decision boundary")
        ax.set_title(f"Predicted P(Recommended) — {name}")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Feature Analysis
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Feature Analysis":
    st.title("🔍 Feature Analysis")
    st.markdown(
        "Logistic regression coefficients extracted **live from the fitted pipeline**. "
        "Positive values push toward *Recommended*, negative toward *Not Recommended*."
    )

    feat_df = get_feature_importance()
    total   = len(feat_df)
    st.caption(f"Total features in model: **{total:,}** "
               f"(structured: {len(NUMERICAL_FEATURES) + 20}, "
               f"TF-IDF: 500, spaCy NLP: 16)")

    st.markdown("---")

    # Filter controls
    col_ctrl1, col_ctrl2 = st.columns([1, 3])
    with col_ctrl1:
        n_show = st.slider("Features to show (each side)", 5, 25, 15)
    with col_ctrl2:
        feat_filter = st.radio(
            "Filter by type",
            ["All", "TF-IDF words", "spaCy NLP", "Structured"],
            horizontal=True
        )

    tfidf_names = set(pipeline.named_steps["features"].tfidf_.get_feature_names_out())
    spacy_names = {
        "char_count","word_count","sent_count","excl_count","quest_count",
        "avg_word_length","adj_count","adv_count","verb_count","noun_count",
        "adj_ratio","adv_ratio","negation_count","neg_adj_count",
        "unique_lemma_ratio","clothing_ents","fit_ents","material_ents",
        "sentiment_pos_ents","net_sentiment_ents"
    }

    if feat_filter == "TF-IDF words":
        filtered = feat_df[feat_df["feature"].isin(tfidf_names)]
    elif feat_filter == "spaCy NLP":
        filtered = feat_df[feat_df["feature"].isin(spacy_names)]
    elif feat_filter == "Structured":
        filtered = feat_df[~feat_df["feature"].isin(tfidf_names) &
                           ~feat_df["feature"].isin(spacy_names)]
    else:
        filtered = feat_df

    top_pos = filtered[filtered["coefficient"] > 0].head(n_show)
    top_neg = filtered[filtered["coefficient"] < 0].tail(n_show)
    combined = pd.concat([top_pos, top_neg]).sort_values("coefficient")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Top {n_show} → Recommended ✓")
        fig, ax = plt.subplots(figsize=(6, max(3, n_show * 0.38)))
        ax.barh(top_pos["feature"], top_pos["coefficient"],
                color="#2d7d72", edgecolor="white")
        ax.set_xlabel("Coefficient")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader(f"Top {n_show} → Not Recommended ✗")
        fig, ax = plt.subplots(figsize=(6, max(3, n_show * 0.38)))
        ax.barh(top_neg["feature"], top_neg["coefficient"].abs(),
                color="#d4607a", edgecolor="white")
        ax.set_xlabel("|Coefficient|")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("spaCy NLP Feature Coefficients")
    st.markdown(
        "These 16 features come directly from `FullFeatureUnion` "
        "— engineered using spaCy token attributes and EntityRuler NER."
    )

    spacy_coefs = feat_df[feat_df["feature"].isin(spacy_names)].sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#d4607a" if c < 0 else "#2d7d72" for c in spacy_coefs["coefficient"]]
    ax.barh(spacy_coefs["feature"], spacy_coefs["coefficient"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Logistic Regression Coefficient")
    ax.set_title("spaCy-derived Feature Importances")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander("📋 Full feature coefficient table"):
        st.dataframe(
            filtered.reset_index(drop=True),
            use_container_width=True
        )


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Live Prediction
# ──────────────────────────────────────────────────────────────────────────────
elif page == "✨ Live Prediction":
    st.title("✨ Live Prediction")
    st.markdown(
        "Enter a review below. The **actual trained scikit-learn pipeline** "
        "will process it with spaCy NLP and return a real prediction."
    )

    st.markdown("---")
    col_form, col_result = st.columns([1, 1])

    with col_form:
        title    = st.text_input("Review Title", placeholder="e.g. Love this top!")
        review   = st.text_area("Review Text", height=140,
                                placeholder="e.g. This dress is absolutely gorgeous and fits perfectly…")
        col_a, col_b = st.columns(2)
        age      = col_a.number_input("Customer Age", min_value=18, max_value=90, value=38)
        feedback = col_b.number_input("Helpful Votes", min_value=0, value=0)
        col_c, col_d = st.columns(2)
        dept     = col_c.selectbox("Department",
                                   ["Tops", "Dresses", "Bottoms", "Jackets", "Intimate", "Trend"])
        division = col_d.selectbox("Division", ["General", "General Petite", "Initmates"])
        cls_name = st.selectbox("Class Name",
                                ["Knits","Dresses","Blouses","Pants","Jeans","Sweaters",
                                 "Fine gauge","Skirts","Jackets","Shorts","Lounge","Trend","Layering","Intimates"])
        clothing_id = st.number_input("Clothing ID", min_value=1, max_value=9999, value=862)

        predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

    with col_result:
        if predict_btn:
            if not review.strip():
                st.warning("Please enter some review text first.")
            else:
                # Build input DataFrame in the exact same schema as training data
                input_df = pd.DataFrame([{
                    "Clothing ID":             int(clothing_id),
                    "Age":                     int(age),
                    "Title":                   title,
                    "Review Text":             review,
                    "Positive Feedback Count": int(feedback),
                    "Division Name":           division,
                    "Department Name":         dept,
                    "Class Name":              cls_name,
                }])

                with st.spinner("Running pipeline…"):
                    prediction = pipeline.predict(input_df)[0]
                    proba      = pipeline.predict_proba(input_df)[0]

                conf       = max(proba)
                is_rec     = prediction == 1
                label_text = "Recommended ✓" if is_rec else "Not Recommended ✗"
                color      = "green" if is_rec else "red"

                st.markdown(f"### :{color}[{label_text}]")
                st.metric("Confidence", f"{conf:.1%}")
                st.progress(float(conf))

                c1, c2 = st.columns(2)
                c1.metric("P(Recommended)",    f"{proba[1]:.3f}")
                c2.metric("P(Not Recommended)", f"{proba[0]:.3f}")

                # Show spaCy NLP signal breakdown for this review
                st.markdown("---")
                st.markdown("**🔬 spaCy NLP breakdown for this review**")

                # Run spaCy directly on the combined text to show entities
                import spacy
                from spacy.lang.en import English

                feat_union = pipeline.named_steps["features"]
                combined_text = (title + " " + review).strip()

                # Use the fitted spacy extractor
                hc_extractor = feat_union.hc_extractor_
                doc = next(hc_extractor.nlp_.pipe([combined_text]))

                ents = [(e.text, e.label_) for e in doc.ents]
                sents = list(doc.sents)

                col_e1, col_e2 = st.columns(2)
                col_e1.markdown(f"**Sentences detected:** {len(sents)}")
                col_e1.markdown(f"**Word count:** {len([t for t in doc if t.is_alpha])}")
                col_e1.markdown(f"**Exclamation marks:** {combined_text.count('!')}")

                if ents:
                    col_e2.markdown("**Named entities found:**")
                    label_colors = {
                        "CLOTHING": "🟦", "FIT": "🟨",
                        "MATERIAL": "🟩", "SENTIMENT_POS": "🟢", "SENTIMENT_NEG": "🔴"
                    }
                    for text_ent, label in ents:
                        icon = label_colors.get(label, "⚪")
                        col_e2.markdown(f"{icon} `{text_ent}` → **{label}**")
                else:
                    col_e2.markdown("_No named entities detected_")

                # Show the TF-IDF tokens spaCy produced
                normalizer = feat_union.spacy_norm_
                norm_text  = normalizer.transform(
                    pd.Series([combined_text])
                ).iloc[0]
                with st.expander("📝 Tokens after spaCy normalization (fed to TF-IDF)"):
                    st.code(norm_text)

        else:
            st.info(
                "Fill in the form on the left and click **Predict**.\n\n"
                "The prediction runs through the real pipeline:\n\n"
                "1. spaCy tokenizes & normalizes the text\n"
                "2. TF-IDF vectorizes the normalized tokens\n"
                "3. FullFeatureUnion computes 16 NLP features\n"
                "4. Logistic Regression outputs a probability\n"
            )

        st.markdown("---")
        st.subheader("Try these example reviews")

        examples = [
            {
                "title": "Absolutely love this!",
                "review": "This blouse is beautiful and flattering. Perfect fit, comfortable fabric. "
                          "I love it and would definitely recommend to everyone! Fits true to size.",
                "dept": "Tops", "cls": "Blouses",
            },
            {
                "title": "Very disappointed — returning",
                "review": "The quality is terrible and the sizing runs way too small. "
                          "The material looks cheap and it started pilling after one wash. Returning.",
                "dept": "Bottoms", "cls": "Pants",
            },
            {
                "title": "Nice but runs large",
                "review": "The dress is pretty and the fabric is soft cotton. "
                          "However it runs large so I had to size down. "
                          "Overall I'm happy with it and would recommend sizing down.",
                "dept": "Dresses", "cls": "Dresses",
            },
        ]

        for ex in examples:
            with st.expander(f'"{ex["title"]}"'):
                st.markdown(f"*{ex['review']}*")
                st.caption(f"Department: {ex['dept']} · Class: {ex['cls']}")
                if st.button(f"Predict this →", key=ex["title"]):
                    ex_df = pd.DataFrame([{
                        "Clothing ID": 862, "Age": 35,
                        "Title": ex["title"], "Review Text": ex["review"],
                        "Positive Feedback Count": 5,
                        "Division Name": "General",
                        "Department Name": ex["dept"],
                        "Class Name": ex["cls"],
                    }])
                    pred  = pipeline.predict(ex_df)[0]
                    proba = pipeline.predict_proba(ex_df)[0]
                    label = "✅ Recommended" if pred == 1 else "❌ Not Recommended"
                    st.success(f"**{label}** — confidence: {max(proba):.1%}")
