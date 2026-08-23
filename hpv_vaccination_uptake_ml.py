"""
HPV Vaccination Uptake in Ethiopia -- Explainable Machine Learning Pipeline
============================================================================
Full analysis code for:
"An Explainable Machine Learning Approach to Human Papillomavirus
Vaccination Uptake Among Adolescent Girls in Ethiopia"
2024--25 Ethiopia Demographic and Health Survey (DHS-8), girls aged 15-17.

This single file runs the complete pipeline end to end, in order:
  1. Exploratory data analysis (chi-square/Cramer's V, correlation, VIF)
  2. Preprocessing + LASSO feature selection (1-SE rule)
  3. Table 1: sample characteristics
  4. Untuned comparison of six ML algorithms
  5. Logistic regression: adjusted odds ratios + forest plot
  6. Hyperparameter tuning (RandomizedSearchCV) for all six models
  7. SHAP analysis (logistic regression + XGBoost)
  8. Sensitivity analysis (10 repeated random train/test splits)
  9. Calibration analysis (Brier scores, reliability diagrams)
  10. Decision curve analysis (net benefit)
  11. Confusion matrices for all tuned models

Data: place the cleaned DHS-8 extract at data/hpv_vaccination_clean.dta
(relative to this script). The DHS-8 dataset itself is not redistributed
here; registered users can request it at
https://dhsprogram.com/data/available-datasets.cfm

Usage:
    pip install -r requirements.txt
    python hpv_vaccination_uptake_ml.py

All figures (.png) and tables (.csv) are written to the data/ folder.
"""

import os
import warnings
import pickle
import pyreadstat
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2_contingency
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_val_score, cross_validate, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_curve, auc, roc_auc_score, confusion_matrix,
                              accuracy_score, precision_score, recall_score,
                              f1_score, brier_score_loss)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import shap


def run_eda():
    """
    EXPLORATORY DATA ANALYSIS
    """
    print("\n" + "="*70)
    print("  EXPLORATORY DATA ANALYSIS")
    print("="*70)

    matplotlib.rcParams['font.size'] = 9

    # ======================================================================
    # 0. SETUP
    # ======================================================================
    # Portable data directory: assumes this script sits in scripts/ and
    # input/output files live in a sibling data/ folder (see README.md)
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.chdir(DATA_DIR)

    DATA_FILE = "hpv_vaccination_clean.dta"

    df, meta = pyreadstat.read_dta(DATA_FILE)
    print("Loaded:", df.shape)


    def get_labels(var):
        """Return the {code: label} dict for a variable, or {} if none."""
        vlabel_name = meta.variable_to_label.get(var)
        return meta.value_labels.get(vlabel_name, {})


    NATIONAL_RATE = 100 * df["hpv_vaccinated"].mean()


    # ======================================================================
    # 1. DESCRIPTIVE SUMMARY (printed to console / log)
    # ======================================================================
    print("\n=== Descriptive summary: N and % vaccinated by covariate ===")

    CAT_VARS = ["education", "wealth", "residence", "region", "religion", "insurance",
                "hh_head_sex", "media_cat", "literacy", "mobile_owner", "internet_use",
                "currently_working", "access_permission", "access_money", "access_distance",
                "water_source", "toilet_type", "electricity"]

    for var in CAT_VARS:
        print(f"\n--- {var} ---")
        vmap = get_labels(var)
        grp = df.groupby(var)["hpv_vaccinated"].agg(["count", "mean"])
        for val, row in grp.iterrows():
            lbl = vmap.get(val, str(val))
            print(f"   {lbl:35s} n={int(row['count']):5d}  vaccinated%={100*row['mean']:.1f}")


    # ======================================================================
    # 2. PANEL FIGURE 1 - socioeconomic, media, and household-amenity covariates
    # ======================================================================
    def bar_panel(ax, var, title, order=None, vmax_scale=95):
        vmap = get_labels(var)
        grp = df.groupby(var)["hpv_vaccinated"].agg(["count", "mean"]).reset_index()
        grp["label"] = grp[var].map(vmap)
        if order:
            grp["label"] = pd.Categorical(grp["label"], categories=order, ordered=True)
            grp = grp.sort_values("label")
        # Bar color encodes the bar's own vaccination rate via viridis, normalized
        # on a fixed 0-vmax_scale scale so shades are comparable ACROSS panels.
        norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax_scale)
        colors = matplotlib.colormaps["viridis"](norm(grp["mean"] * 100))
        ax.bar(grp["label"], grp["mean"] * 100, color=colors, width=0.6)
        ax.axhline(NATIONAL_RATE, linestyle="--", color="firebrick", linewidth=0.8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_ylabel("% vaccinated")
        ax.set_ylim(0, 95)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")


    fig, axes = plt.subplots(4, 3, figsize=(13, 15))

    bar_panel(axes[0, 0], "education", "A. Education",
              ["No education", "Primary", "Secondary", "Higher"])
    bar_panel(axes[0, 1], "wealth", "B. Wealth quintile",
              ["Poorest", "Poorer", "Middle", "Richer", "Richest"])
    bar_panel(axes[0, 2], "residence", "C. Residence")
    bar_panel(axes[1, 0], "religion", "D. Religion")
    bar_panel(axes[1, 1], "insurance", "E. Health insurance")
    bar_panel(axes[1, 2], "media_cat", "F. Mass media exposure",
              ["No exposure", "Low", "Moderate", "High"])
    bar_panel(axes[2, 0], "literacy", "G. Literacy",
              ["Cannot read at all", "Reads part of sentence", "Reads whole sentence", "Other/Unknown"])
    bar_panel(axes[2, 1], "internet_use", "H. Internet use",
              ["Never", "Last 12 months", "Before last 12 months"])
    bar_panel(axes[2, 2], "mobile_owner", "I. Mobile phone ownership")
    bar_panel(axes[3, 0], "water_source", "J. Water source",
              ["Piped", "Other improved", "Unimproved", "Other"])
    bar_panel(axes[3, 1], "toilet_type", "K. Toilet type",
              ["Flush/pour-flush", "Improved pit latrine", "Unimproved", "No facility/Other"])
    bar_panel(axes[3, 2], "electricity", "L. Electricity")

    plt.tight_layout()
    plt.savefig("eda_panel1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved eda_panel1.png")


    # ======================================================================
    # 3. REGION CHART (separate, sorted, horizontal -- 14 categories)
    # ======================================================================
    vmap = get_labels("region")
    grp = df.groupby("region")["hpv_vaccinated"].agg(["count", "mean"]).reset_index()
    grp["label"] = grp["region"].map(vmap)
    grp = grp.sort_values("mean")

    fig, ax = plt.subplots(figsize=(9, 6))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=85)
    colors = matplotlib.colormaps["viridis"](norm(grp["mean"] * 100))
    ax.barh(grp["label"], grp["mean"] * 100, color=colors)
    ax.axvline(NATIONAL_RATE, linestyle="--", color="firebrick", linewidth=0.8)
    ax.text(NATIONAL_RATE + 0.5, -0.8, f"National average: {NATIONAL_RATE:.1f}%",
            color="firebrick", fontsize=8)
    ax.set_xlabel("% vaccinated")
    ax.set_xlim(0, 85)
    plt.tight_layout()
    plt.savefig("eda_region.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved eda_region.png")


    # ======================================================================
    # 4. PANEL FIGURE 2 - access barriers, household head, currently working, age
    # ======================================================================
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    def bar_panel2(ax, var, title, order=None):
        bar_panel(ax, var, title, order=order, vmax_scale=75)
        ax.set_ylim(0, 75)

    bar_panel2(axes[0, 0], "access_permission", "M. Barrier: permission")
    bar_panel2(axes[0, 1], "access_money", "N. Barrier: money")
    bar_panel2(axes[0, 2], "access_distance", "O. Barrier: distance")
    bar_panel2(axes[1, 0], "hh_head_sex", "P. Household head sex")
    bar_panel2(axes[1, 1], "currently_working", "Q. Currently working")

    # Age: only 3 discrete values (15/16/17), treat as categorical
    grp = df.groupby("age")["hpv_vaccinated"].agg(["count", "mean"]).reset_index()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=75)
    colors = matplotlib.colormaps["viridis"](norm(grp["mean"] * 100))
    axes[1, 2].bar(grp["age"].astype(int).astype(str), grp["mean"] * 100,
                   color=colors, width=0.5)
    axes[1, 2].axhline(NATIONAL_RATE, linestyle="--", color="firebrick", linewidth=0.8)
    axes[1, 2].set_title("R. Age", fontsize=9, fontweight="bold")
    axes[1, 2].set_ylabel("% vaccinated")
    axes[1, 2].set_ylim(0, 75)

    plt.tight_layout()
    plt.savefig("eda_panel2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved eda_panel2.png")


    # ======================================================================
    # 5. CORRELATION HEATMAP (all variables)
    # ======================================================================
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(11, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis", vmin=-1, vmax=1,
                annot_kws={"size": 6.5}, ax=ax, square=True, cbar_kws={"shrink": 0.8})
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig("eda_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved eda_correlation.png")


    # ======================================================================
    # 6. CHI-SQUARE TEST + CRAMER'S V (association strength, ranked)
    # ======================================================================
    # Chi-square tests whether each covariate is significantly associated
    # with the outcome; Cramer's V gives an effect-size for that association
    # (0=none, and conventionally: <0.1 negligible, 0.1-0.3 weak/moderate,
    # >0.3 stronger), which is a more appropriate measure than raw Pearson
    # correlation for these nominal/ordinal-coded categorical variables.
    print("\n=== Chi-square test + Cramer's V (ranked by association strength) ===")

    CHI_VARS = CAT_VARS + ["age"]
    n = len(df)
    chi_results = []
    for var in CHI_VARS:
        ct = pd.crosstab(df[var], df["hpv_vaccinated"])
        chi2, p, dof, expected = chi2_contingency(ct)
        k = min(ct.shape)
        cramers_v = np.sqrt(chi2 / (n * (k - 1)))
        chi_results.append({"variable": var, "chi2": chi2, "p_value": p, "cramers_v": cramers_v})

    chi_df = pd.DataFrame(chi_results).sort_values("cramers_v", ascending=False)
    chi_df["p_value_str"] = chi_df["p_value"].apply(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")
    print(chi_df[["variable", "chi2", "p_value_str", "cramers_v"]].to_string(index=False))
    chi_df.to_csv("eda_chisquare_cramersv.csv", index=False)
    print("Saved eda_chisquare_cramersv.csv")


    # ======================================================================
    # 7. VARIANCE INFLATION FACTOR (VIF) - multicollinearity check
    # ======================================================================
    # Pairwise correlation (Section 5) only catches two-variable relationships.
    # VIF catches multicollinearity arising from COMBINATIONS of variables
    # (e.g. wealth being well-predicted by residence + electricity + toilet_type
    # together, even if no single pairwise correlation looks alarming) --
    # most relevant for the logistic regression baseline, which is more
    # sensitive to this than the tree-based models.
    print("\n=== Variance Inflation Factor (VIF) ===")
    print("One-hot encoding categoricals (drop_first=True to avoid the dummy trap),")
    print("keeping hh_head_age/hh_size as continuous.\n")

    VIF_VARS = CAT_VARS + ["age", "hh_head_age", "hh_size"]
    cat_to_encode = [c for c in VIF_VARS if c not in ["hh_head_age", "hh_size"]]

    X = pd.get_dummies(df[VIF_VARS], columns=cat_to_encode, drop_first=True).astype(float)
    X = sm.add_constant(X)

    vif_df = pd.DataFrame()
    vif_df["feature"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_df = vif_df[vif_df["feature"] != "const"].sort_values("VIF", ascending=False)

    print(vif_df.to_string(index=False))
    print(f"\nMax VIF: {vif_df['VIF'].max():.2f}  "
          "(rule of thumb: >10 = serious concern, >5 = worth watching)")
    vif_df.to_csv("eda_vif.csv", index=False)
    print("Saved eda_vif.csv")


    # ======================================================================
    # 6. CONTINUOUS VARIABLE DISTRIBUTIONS
    # ======================================================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    viridis_mid = matplotlib.colormaps["viridis"](0.4)
    axes[0].hist(df["hh_head_age"], bins=30, color=viridis_mid, edgecolor="white")
    axes[0].set_title("Household head age", fontweight="bold")
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(df["hh_size"], bins=30, color=viridis_mid, edgecolor="white")
    axes[1].set_title("Household size", fontweight="bold")
    axes[1].set_xlabel("Number of members")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("eda_continuous.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved eda_continuous.png")

    print("\n=== EDA complete. All figures saved to:", DATA_DIR, "===")



def run_preprocessing_lasso():
    """
    PREPROCESSING + LASSO FEATURE SELECTION
    """
    print("\n" + "="*70)
    print("  PREPROCESSING + LASSO FEATURE SELECTION")
    print("="*70)

    warnings.filterwarnings("ignore")
    matplotlib.rcParams['font.size'] = 9

    # ======================================================================
    # 0. SETUP
    # ======================================================================
    # Portable data directory: assumes this script sits in scripts/ and
    # input/output files live in a sibling data/ folder (see README.md)
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.chdir(DATA_DIR)

    RANDOM_STATE = 42  # fixed throughout the whole ML pipeline for reproducibility

    df, meta = pyreadstat.read_dta("hpv_vaccination_clean.dta")
    print("Loaded:", df.shape)

    CAT_VARS = ["education", "wealth", "residence", "region", "religion", "insurance",
                "hh_head_sex", "media_cat", "literacy", "mobile_owner", "internet_use",
                "currently_working", "access_permission", "access_money", "access_distance",
                "water_source", "toilet_type", "electricity", "age"]
    CONT_VARS = ["hh_head_age", "hh_size"]


    # ======================================================================
    # 1. ENCODING
    # ======================================================================
    # One-hot encode all categoricals (drop_first=True avoids the dummy trap);
    # continuous variables (hh_head_age, hh_size) kept as-is until scaling.
    y = df["hpv_vaccinated"].astype(int)
    X_raw = df[CAT_VARS + CONT_VARS].copy()
    X = pd.get_dummies(X_raw, columns=CAT_VARS, drop_first=True).astype(float)

    print(f"Encoded feature count: {X.shape[1]} (from {len(CAT_VARS)} categorical + "
          f"{len(CONT_VARS)} continuous source variables)")


    # ======================================================================
    # 2. TRAIN/TEST SPLIT
    # ======================================================================
    # Held-out test set reserved for final model evaluation only, after the
    # full model comparison (Section: ML modeling, done separately). LASSO
    # feature selection below uses only the training set, with its own
    # internal stratified CV -- the test set is never touched here.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {X_train.shape[0]}  Test: {X_test.shape[0]}")
    print(f"Train outcome rate: {y_train.mean():.3f}  Test outcome rate: {y_test.mean():.3f}")

    X_train.to_csv("X_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)
    print("Saved X_train.csv, X_test.csv, y_train.csv, y_test.csv")


    # ======================================================================
    # 3. STANDARDIZATION
    # ======================================================================
    # Fit on training data only, applied to both train and test, to avoid
    # any leakage of test-set distribution information.
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),
                                   columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test),
                                  columns=X_test.columns, index=X_test.index)


    # ======================================================================
    # 4. LASSO FEATURE SELECTION (1-SE rule)
    # ======================================================================
    # Standard cross-validated LASSO (picking the C that maximizes mean CV
    # AUC) tends to select a fairly WEAK regularization strength, since even
    # negligible-effect features can marginally improve predictive AUC. For
    # genuine feature *selection* (as opposed to pure prediction), the
    # standard fix is the 1-SE rule: choose the simplest model (strongest
    # regularization / smallest C) whose CV performance is still within one
    # standard error of the best model's CV performance. This is the same
    # convention used by, e.g., R's glmnet cv.glmnet(..., s="lambda.1se").
    Cs = np.logspace(-3, 1, 30)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    mean_scores, se_scores = [], []
    for C in Cs:
        model = LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                    max_iter=5000, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring="roc_auc")
        mean_scores.append(scores.mean())
        se_scores.append(scores.std() / np.sqrt(len(scores)))

    mean_scores = np.array(mean_scores)
    se_scores = np.array(se_scores)

    best_idx = mean_scores.argmax()
    best_C, best_score, best_se = Cs[best_idx], mean_scores[best_idx], se_scores[best_idx]
    threshold = best_score - best_se

    eligible = np.where(mean_scores >= threshold)[0]
    se_idx = eligible[np.argmin(Cs[eligible])]
    se_C, se_score = Cs[se_idx], mean_scores[se_idx]

    print(f"\n=== LASSO regularization path ===")
    print(f"Best C (max CV AUC):  {best_C:.4f}  (AUC={best_score:.4f} +/- {best_se:.4f})")
    print(f"1-SE rule C:          {se_C:.4f}  (AUC={se_score:.4f})")
    print(f"Threshold (best-1SE): {threshold:.4f}")

    # --- Regularization path plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.log10(Cs), mean_scores, color="#31688e", linewidth=1.5)
    ax.fill_between(np.log10(Cs), mean_scores - se_scores, mean_scores + se_scores,
                     alpha=0.2, color="#31688e")
    ax.axvline(np.log10(best_C), linestyle="--", color="firebrick", linewidth=0.8,
               label=f"Best AUC (C={best_C:.3f})")
    ax.axvline(np.log10(se_C), linestyle="--", color="darkgreen", linewidth=0.8,
               label=f"1-SE rule (C={se_C:.3f})")
    ax.set_xlabel("log10(C)  [larger C = weaker regularization]")
    ax.set_ylabel("Mean CV AUC (5-fold)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("lasso_regularization_path.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved lasso_regularization_path.png")

    # --- Coefficient path: shows EVERY feature's coefficient trajectory as
    #     regularization strengthens, so you can see exactly where each one
    #     is shrunk to zero (rather than just the two summary AUC curves above) ---
    coef_path = []
    for C in Cs:
        m = LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                max_iter=5000, random_state=RANDOM_STATE)
        m.fit(X_train_scaled, y_train)
        coef_path.append(m.coef_[0])
    coef_path = np.array(coef_path)
    feature_names = X_train_scaled.columns.tolist()

    final_coefs = coef_path[-1, :]
    top_idx = np.argsort(np.abs(final_coefs))[::-1][:12]

    fig, ax = plt.subplots(figsize=(10, 7))
    for j in range(coef_path.shape[1]):
        if j in top_idx:
            ax.plot(np.log10(Cs), coef_path[:, j], linewidth=1.6, label=feature_names[j])
        else:
            ax.plot(np.log10(Cs), coef_path[:, j], linewidth=0.6, color="lightgray", zorder=1)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(np.log10(best_C), linestyle="--", color="firebrick", linewidth=0.9,
               label=f"Best AUC (C={best_C:.3f})")
    ax.axvline(np.log10(se_C), linestyle="--", color="darkgreen", linewidth=0.9,
               label=f"1-SE rule (C={se_C:.3f})")
    ax.set_xlabel("log10(C)  [larger C = weaker regularization]")
    ax.set_ylabel("Coefficient (standardized)")
    ax.legend(fontsize=6.5, loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig("lasso_coefficient_path.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved lasso_coefficient_path.png")

    # --- Fit final LASSO at the 1-SE C, extract selected/dropped features ---
    final_lasso = LogisticRegression(penalty="l1", solver="liblinear", C=se_C,
                                      max_iter=5000, random_state=RANDOM_STATE)
    final_lasso.fit(X_train_scaled, y_train)

    coefs = pd.Series(final_lasso.coef_[0], index=X_train_scaled.columns)
    selected = coefs[coefs != 0].sort_values(key=abs, ascending=False)
    dropped = coefs[coefs == 0]

    print(f"\nFeatures selected: {len(selected)} of {len(coefs)}")
    print(f"Features dropped:  {len(dropped)}")
    print("\n=== Selected features (sorted by |coefficient|) ===")
    print(selected.to_string())
    print("\n=== Dropped features ===")
    print(sorted(dropped.index.tolist()))

    selected.to_csv("lasso_selected_features.csv", header=["coefficient"])
    pd.Series(sorted(dropped.index.tolist())).to_csv("lasso_dropped_features.csv",
                                                        index=False, header=["dropped_feature"])
    print("\nSaved lasso_selected_features.csv, lasso_dropped_features.csv")

    # --- Coefficient plot ---
    fig, ax = plt.subplots(figsize=(7, 9))
    colors = ["#2ca25f" if c > 0 else "#de2d26" for c in selected.values]
    ax.barh(selected.index[::-1], selected.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LASSO coefficient (standardized)")
    plt.tight_layout()
    plt.savefig("lasso_coefficients.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved lasso_coefficients.png")

    print("\n=== Preprocessing + LASSO feature selection complete ===")
    print(f"Use the {len(selected)} selected features for the LR baseline;")
    print("full feature set remains available for the tree-based models and MLP.")



def run_table1_characteristics():
    """
    TABLE 1: SAMPLE CHARACTERISTICS
    """
    print("\n" + "="*70)
    print("  TABLE 1: SAMPLE CHARACTERISTICS")
    print("="*70)

    # Portable data directory: assumes this script sits in scripts/ and
    # input/output files live in a sibling data/ folder (see README.md)
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.chdir(DATA_DIR)

    df, meta = pyreadstat.read_dta("hpv_vaccination_clean.dta")
    n_total = len(df)
    print(f"N = {n_total}")


    def get_labels(var):
        vlabel_name = meta.variable_to_label.get(var)
        return meta.value_labels.get(vlabel_name, {})


    # ======================================================================
    # 1. VERIFY EVERY VARIABLE SUMS CORRECTLY TO N BEFORE BUILDING THE TABLE
    # ======================================================================
    cat_vars_check = ["age", "education", "wealth", "residence", "region", "religion",
                       "insurance", "hh_head_sex", "media_cat", "literacy", "mobile_owner",
                       "internet_use", "currently_working", "access_permission",
                       "access_money", "access_distance", "water_source", "toilet_type",
                       "electricity"]
    for var in cat_vars_check:
        total = df[var].value_counts().sum()
        assert total == n_total, f"{var} sums to {total}, expected {n_total}"
    print(f"All {len(cat_vars_check)} categorical variables verified to sum to N={n_total}")


    # ======================================================================
    # 2. BUILD THE TABLE
    # ======================================================================
    cat_vars_ordered = [
        ("age", "Age (years)", None),
        ("education", "Education", ["No education", "Primary", "Secondary", "Higher"]),
        ("wealth", "Wealth quintile", ["Poorest", "Poorer", "Middle", "Richer", "Richest"]),
        ("residence", "Residence", None),
        ("region", "Region", None),
        ("religion", "Religion", None),
        ("insurance", "Health insurance", None),
        ("hh_head_sex", "Household head sex", None),
        ("media_cat", "Mass media exposure", ["No exposure", "Low", "Moderate", "High"]),
        ("literacy", "Literacy", ["Cannot read at all", "Reads part of sentence",
                                   "Reads whole sentence", "Other/Unknown"]),
        ("mobile_owner", "Mobile phone ownership", None),
        ("internet_use", "Internet use", None),
        ("currently_working", "Currently working", None),
        ("access_permission", "Barrier: permission", None),
        ("access_money", "Barrier: money", None),
        ("access_distance", "Barrier: distance", None),
        ("water_source", "Water source", ["Piped", "Other improved", "Unimproved", "Other"]),
        ("toilet_type", "Toilet type", ["Flush/pour-flush", "Improved pit latrine",
                                         "Unimproved", "No facility/Other"]),
        ("electricity", "Electricity", None),
    ]

    rows = []
    for var, disp_name, order in cat_vars_ordered:
        vmap = get_labels(var)
        grp = df.groupby(var)["hpv_vaccinated"].agg(["count", "mean"]).reset_index()
        grp["label"] = grp[var].map(vmap) if vmap else grp[var].astype(int).astype(str)
        if order:
            grp["label"] = pd.Categorical(grp["label"], categories=order, ordered=True)
            grp = grp.sort_values("label")
        rows.append({"characteristic": disp_name, "category": "", "n": "", "pct": "", "vacc_pct": ""})
        for _, r in grp.iterrows():
            rows.append({
                "characteristic": "",
                "category": r["label"],
                "n": int(r["count"]),
                "pct": f"{100*r['count']/n_total:.1f}",
                "vacc_pct": f"{100*r['mean']:.1f}",
            })

    for var, disp_name in [("hh_head_age", "Household head age (years)"),
                            ("hh_size", "Household size")]:
        rows.append({
            "characteristic": disp_name, "category": "Mean (SD)",
            "n": n_total, "pct": "",
            "vacc_pct": f"{df[var].mean():.1f} ({df[var].std():.1f})",
        })

    table1 = pd.DataFrame(rows)
    table1.columns = ["Characteristic", "Category", "N", "%", "% vaccinated / Mean (SD)"]
    table1.to_csv("table1_sample_characteristics.csv", index=False)
    print("\nSaved table1_sample_characteristics.csv")
    print(table1.to_string(index=False))


    # ======================================================================
    # 3. LATEX VERSION (longtable, matching the cervical cancer paper's style)
    # ======================================================================
    latex_lines = [
        r"\begin{longtable}{lccc}",
        r"\caption{Sample characteristics and unweighted HPV vaccination uptake among girls aged 15--17}",
        r"\label{tab:descriptive} \\",
        r"\toprule",
        r"Characteristic & N & \% & \% vaccinated / Mean (SD) \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Characteristic & N & \% & \% vaccinated / Mean (SD) \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for _, r in table1.iterrows():
        if r["Category"] == "":
            latex_lines.append(rf"\multicolumn{{4}}{{l}}{{\textbf{{{r['Characteristic']}}}}} \\")
        else:
            cat = str(r["Category"]).replace("&", r"\&")
            latex_lines.append(rf"\hspace{{2mm}}{cat} & {r['N']} & {r['%']} & {r['% vaccinated / Mean (SD)']} \\")
    latex_lines.append(r"\end{longtable}")

    with open("table1_sample_characteristics.tex", "w") as f:
        f.write("\n".join(latex_lines))
    print("\nSaved table1_sample_characteristics.tex")

    print("\n=== Table 1 complete ===")



def run_model_comparison():
    """
    MODEL COMPARISON (UNTUNED)
    """
    print("\n" + "="*70)
    print("  MODEL COMPARISON (UNTUNED)")
    print("="*70)

    warnings.filterwarnings("ignore")
    matplotlib.rcParams['font.size'] = 9

    # ======================================================================
    # 0. SETUP
    # ======================================================================
    # Portable data directory: assumes this script sits in scripts/ and
    # input/output files live in a sibling data/ folder (see README.md)
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.chdir(DATA_DIR)
    RANDOM_STATE = 42

    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").iloc[:, 0]
    y_test = pd.read_csv("y_test.csv").iloc[:, 0]
    selected_features = pd.read_csv("lasso_selected_features.csv").iloc[:, 0].tolist()

    print(f"Using {len(selected_features)} LASSO-selected features, consistently across all models.")

    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    # Scaled versions -- needed for LR and MLP specifically; tree-based models
    # (RF, XGBoost, LightGBM, CatBoost) are scale-invariant and use the raw
    # (unscaled) selected features directly.
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_sel), columns=selected_features)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_sel), columns=selected_features)


    # ======================================================================
    # 1. DEFINE MODELS
    # ======================================================================
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                                 X_train_scaled, X_test_scaled),
        "Random Forest":       (RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
                                 X_train_sel, X_test_sel),
        "XGBoost":             (XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1),
                                 X_train_sel, X_test_sel),
        "LightGBM":            (LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
                                 X_train_sel, X_test_sel),
        "CatBoost":            (CatBoostClassifier(random_state=RANDOM_STATE, verbose=0),
                                 X_train_sel, X_test_sel),
        "MLP":                 (MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000,
                                               random_state=RANDOM_STATE, early_stopping=True,
                                               validation_fraction=0.15),
                                 X_train_scaled, X_test_scaled),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["roc_auc", "accuracy", "precision", "recall", "f1"]


    # ======================================================================
    # 2. STRATIFIED 5-FOLD CV ON TRAINING SET
    # ======================================================================
    print("\n=== Stratified 5-fold CV results (training set) ===")
    cv_summary = []
    for name, (model, Xtr, Xte) in models.items():
        cv_results = cross_validate(model, Xtr, y_train, cv=skf, scoring=scoring, n_jobs=1)
        row = {"model": name}
        for m in scoring:
            row[f"{m}_mean"] = cv_results[f"test_{m}"].mean()
            row[f"{m}_sd"] = cv_results[f"test_{m}"].std()
        cv_summary.append(row)
        print(f"\n{name}:")
        for m in scoring:
            print(f"  {m}: {row[f'{m}_mean']:.4f} +/- {row[f'{m}_sd']:.4f}")

    cv_df = pd.DataFrame(cv_summary)
    cv_df.to_csv("model_comparison_cv_results.csv", index=False)
    print("\nSaved model_comparison_cv_results.csv")


    # ======================================================================
    # 3. HELD-OUT TEST SET EVALUATION
    # ======================================================================
    print("\n\n=== Held-out test set results (fit on full training set) ===")
    test_summary = []
    roc_data = {}
    cm_data = {}
    for name, (model, Xtr, Xte) in models.items():
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]

        test_auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, test_auc)
        cm_data[name] = confusion_matrix(y_test, y_pred)

        row = {
            "model": name,
            "test_auc": test_auc,
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
        }
        test_summary.append(row)
        print(f"{name:22s} AUC={row['test_auc']:.4f}  Acc={row['test_accuracy']:.4f}  "
              f"Prec={row['test_precision']:.4f}  Rec={row['test_recall']:.4f}  F1={row['test_f1']:.4f}")

    test_df = pd.DataFrame(test_summary).sort_values("test_auc", ascending=False)
    test_df.to_csv("model_comparison_test_results.csv", index=False)
    print("\nSaved model_comparison_test_results.csv")


    # ======================================================================
    # 4. IDENTIFY BEST GRADIENT BOOSTING VARIANT
    # ======================================================================
    gb_variants = test_df[test_df["model"].isin(["XGBoost", "LightGBM", "CatBoost"])]
    best_gb = gb_variants.sort_values("test_auc", ascending=False).iloc[0]
    print(f"\n=== Best gradient boosting variant: {best_gb['model']} "
          f"(test AUC={best_gb['test_auc']:.4f}) ===")
    print("This is the GB variant that will represent 'gradient boosting' going forward.")


    # ======================================================================
    # 5. ROC CURVE COMPARISON PLOT
    # ======================================================================
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = matplotlib.colormaps["viridis"](np.linspace(0, 0.9, len(roc_data)))
    for (name, (fpr, tpr, roc_auc_val)), color in zip(roc_data.items(), colors):
        ax.plot(fpr, tpr, color=color, linewidth=1.6, label=f"{name} (AUC={roc_auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig("model_comparison_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved model_comparison_roc.png")


    # ======================================================================
    # 6. CONFUSION MATRICES (all models, shared color scale for comparability)
    # ======================================================================
    global_max = max(cm.max() for cm in cm_data.values())

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for ax, (name, cm) in zip(axes, cm_data.items()):
        im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=global_max)
        for i in range(2):
            for j in range(2):
                pct = 100 * cm[i, j] / cm.sum()
                color = "white" if cm[i, j] < global_max * 0.6 else "black"
                ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)", ha="center", va="center",
                         color=color, fontsize=9)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Not vacc.", "Vacc."], fontsize=8)
        ax.set_yticklabels(["Not vacc.", "Vacc."], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)
        ax.set_title(name, fontsize=9, fontweight="bold")

    fig.subplots_adjust(left=0.06, right=0.90, top=0.94, bottom=0.06, wspace=0.4, hspace=0.4)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Count")
    plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved confusion_matrices.png")

    print("\n=== Model comparison complete ===")



def run_lr_odds_ratios():
    """
    LOGISTIC REGRESSION: ODDS RATIOS + FOREST PLOT
    """
    print("\n" + "="*70)
    print("  LOGISTIC REGRESSION: ODDS RATIOS + FOREST PLOT")
    print("="*70)

    warnings.filterwarnings("ignore")
    matplotlib.rcParams['font.size'] = 9

    # Portable data directory: assumes this script sits in scripts/ and
    # input/output files live in a sibling data/ folder (see README.md)
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.chdir(DATA_DIR)

    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv").iloc[:, 0]
    selected_features = pd.read_csv("lasso_selected_features.csv").iloc[:, 0].tolist()

    X_sel = X_train[selected_features].copy()
    X_sel_const = sm.add_constant(X_sel)


    # ======================================================================
    # 1. FIT LOGISTIC REGRESSION (statsmodels, for proper SEs/CIs)
    # ======================================================================
    result = sm.Logit(y_train, X_sel_const).fit(disp=0)
    print(result.summary())

    params = result.params.drop("const")
    conf = result.conf_int().drop("const")
    pvals = result.pvalues.drop("const")

    or_table = pd.DataFrame({
        "OR": np.exp(params), "CI_lower": np.exp(conf[0]),
        "CI_upper": np.exp(conf[1]), "p_value": pvals
    })


    # ======================================================================
    # 2. HUMAN-READABLE LABELS
    # ======================================================================
    label_map = {
        "education_1.0": "Education: Primary (ref: None)",
        "education_2.0": "Education: Secondary (ref: None)",
        "region_2.0": "Region: Afar (ref: Tigray)",
        "region_4.0": "Region: Oromia (ref: Tigray)",
        "region_5.0": "Region: Somali (ref: Tigray)",
        "region_6.0": "Region: Benishangul-Gumuz (ref: Tigray)",
        "region_7.0": "Region: Central Ethiopia (ref: Tigray)",
        "region_8.0": "Region: Sidama (ref: Tigray)",
        "region_9.0": "Region: South West Ethiopia (ref: Tigray)",
        "region_10.0": "Region: South Ethiopia (ref: Tigray)",
        "region_13.0": "Region: Harari (ref: Tigray)",
        "literacy_2.0": "Literacy: Reads part of sentence (ref: Cannot read)",
        "literacy_3.0": "Literacy: Reads whole sentence (ref: Cannot read)",
        "literacy_4.0": "Literacy: Other/Unknown (ref: Cannot read)",
        "media_cat_1.0": "Media exposure: Low (ref: None)",
        "media_cat_2.0": "Media exposure: Moderate (ref: None)",
        "media_cat_3.0": "Media exposure: High (ref: None)",
        "wealth_2.0": "Wealth: Poorer (ref: Poorest)",
        "wealth_3.0": "Wealth: Middle (ref: Poorest)",
        "wealth_4.0": "Wealth: Richer (ref: Poorest)",
        "wealth_5.0": "Wealth: Richest (ref: Poorest)",
        "insurance_1.0": "Health insurance: Yes (ref: No)",
        "internet_use_1.0": "Internet use: Last 12 months (ref: Never)",
        "toilet_type_2.0": "Toilet: Improved pit latrine (ref: Flush)",
        "toilet_type_4.0": "Toilet: No facility/Other (ref: Flush)",
        "religion_3.0": "Religion: Muslim (ref: Orthodox)",
        "water_source_3.0": "Water source: Unimproved (ref: Piped)",
        "currently_working_1.0": "Currently working: Yes (ref: No)",
        "hh_size": "Household size (continuous)",
    }

    # Simplified labels for the forest PLOT specifically (reference categories
    # dropped -- standard forest-plot convention is to state them once in the
    # caption/footnote rather than repeat them on every axis label). The CSV
    # table above keeps the full labels, where the reference is needed to
    # correctly read each row's numeric OR.
    plot_label_map = {k: v.split(" (ref:")[0] for k, v in label_map.items()}
    or_table["label"] = or_table.index.map(label_map)
    missing = or_table[or_table["label"].isna()]
    if len(missing) > 0:
        raise ValueError(f"No label defined for: {missing.index.tolist()} -- "
                          "add these to label_map before proceeding.")


    # ======================================================================
    # 3. SAVE OR TABLE
    # ======================================================================
    or_table_out = or_table.sort_values("OR", ascending=False)
    # Plain-number p-value formatting (matches the <0.001 convention already
    # used in the chi-square/Cramer's V table) rather than scientific notation
    or_table_out["p_value_str"] = or_table_out["p_value"].apply(
        lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")
    or_table_out[["label", "OR", "CI_lower", "CI_upper", "p_value_str"]].to_csv(
        "lr_odds_ratios.csv", index=False, header=["label", "OR", "CI_lower", "CI_upper", "p_value"])
    print("\nSaved lr_odds_ratios.csv")
    print(or_table_out[["label", "OR", "CI_lower", "CI_upper", "p_value_str"]].to_string(index=False))


    # ======================================================================
    # 4. FOREST PLOT
    # ======================================================================
    or_table_plot = or_table.sort_values("OR", ascending=True)
    or_table_plot["plot_label"] = or_table_plot.index.map(plot_label_map)
    fig, ax = plt.subplots(figsize=(9, 10))
    norm = matplotlib.colors.Normalize(vmin=or_table_plot["OR"].min(), vmax=or_table_plot["OR"].max())
    colors = matplotlib.colormaps["viridis"](norm(or_table_plot["OR"]))

    for i, (idx, row) in enumerate(or_table_plot.iterrows()):
        ax.plot([row["CI_lower"], row["CI_upper"]], [i, i], color=colors[i], linewidth=1.8, zorder=1)
        ax.scatter(row["OR"], i, color=colors[i], s=45, zorder=2, edgecolor="black", linewidth=0.4)

    ax.axvline(1, linestyle="--", color="firebrick", linewidth=0.9)
    ax.set_yticks(np.arange(len(or_table_plot)))
    ax.set_yticklabels(or_table_plot["plot_label"], fontsize=7.5)
    ax.set_xscale("log")

    # Explicit tick positions with plain-number labels (not matplotlib's default
    # 10^0/10^1 scientific notation on a log axis), covering the actual OR range
    ticks = [0.3, 0.5, 1, 2, 3, 5]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.set_xlabel("Odds Ratio (95% CI, log scale)")
    plt.tight_layout()
    plt.savefig("lr_forest_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved lr_forest_plot.png")

    print("\n=== LR odds ratio analysis complete ===")



def run_hyperparameter_tuning():
    """
    HYPERPARAMETER TUNING
    """
    print("\n" + "="*70)
    print("  HYPERPARAMETER TUNING")
    print("="*70)

    warnings.filterwarnings("ignore")
    matplotlib.rcParams['font.size'] = 9

    # Portable data directory: assumes this script sits in scripts/ and
    # input/output files live in a sibling data/ folder (see README.md)
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.chdir(DATA_DIR)
    RANDOM_STATE = 42

    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").iloc[:, 0]
    y_test = pd.read_csv("y_test.csv").iloc[:, 0]
    selected_features = pd.read_csv("lasso_selected_features.csv").iloc[:, 0].tolist()

    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_sel), columns=selected_features)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_sel), columns=selected_features)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


    # ======================================================================
    # 1. HYPERPARAMETER GRIDS
    # ======================================================================
    param_grids = {
        "Logistic Regression": {
            "C": np.logspace(-3, 2, 50), "penalty": ["l1", "l2"], "solver": ["liblinear"],
        },
        "Random Forest": {
            "n_estimators": [100, 200, 300, 500], "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10, 20], "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", None],
        },
        "XGBoost": {
            "n_estimators": [50, 100, 200, 300], "max_depth": [2, 3, 4, 5, 6],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0],
        },
        "LightGBM": {
            "n_estimators": [50, 100, 200, 300], "max_depth": [2, 3, 4, 5, -1],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "num_leaves": [7, 15, 31, 63], "min_child_samples": [5, 10, 20, 30],
        },
        "CatBoost": {
            "iterations": [100, 200, 300], "depth": [3, 4, 5, 6, 7],
            "learning_rate": [0.01, 0.03, 0.05, 0.1], "l2_leaf_reg": [1, 3, 5, 7, 9],
        },
        "MLP": {
            "hidden_layer_sizes": [(32,), (64,), (32, 16), (64, 32), (100,), (64, 32, 16)],
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
            "learning_rate_init": [0.001, 0.01, 0.05],
        },
    }

    base_models = {
        "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), X_train_scaled, X_test_scaled),
        "Random Forest":       (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), X_train_sel, X_test_sel),
        "XGBoost":             (XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1), X_train_sel, X_test_sel),
        "LightGBM":            (LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1), X_train_sel, X_test_sel),
        "CatBoost":            (CatBoostClassifier(random_state=RANDOM_STATE, verbose=0), X_train_sel, X_test_sel),
        "MLP":                 (MLPClassifier(max_iter=2000, random_state=RANDOM_STATE, early_stopping=True,
                                               validation_fraction=0.15), X_train_scaled, X_test_scaled),
    }

    N_ITER = {"Logistic Regression": 30, "Random Forest": 30, "XGBoost": 30,
              "LightGBM": 30, "CatBoost": 20, "MLP": 20}


    # ======================================================================
    # 2. RUN TUNING FOR EACH MODEL
    # ======================================================================
    tuned_models = {}
    tuning_summary = []

    for name, (model, Xtr, Xte) in base_models.items():
        print(f"\nTuning {name}...")
        search = RandomizedSearchCV(
            model, param_grids[name], n_iter=N_ITER[name], cv=skf,
            scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=-1
        )
        search.fit(Xtr, y_train)
        tuned_models[name] = (search.best_estimator_, Xtr, Xte)
        tuning_summary.append({
            "model": name, "best_params": str(search.best_params_),
            "best_cv_auc": search.best_score_
        })
        print(f"  Best CV AUC: {search.best_score_:.4f}")
        print(f"  Best params: {search.best_params_}")

    tuning_df = pd.DataFrame(tuning_summary)
    tuning_df.to_csv("tuning_best_params.csv", index=False)
    print("\nSaved tuning_best_params.csv")


    # ======================================================================
    # 3. RE-EVALUATE TUNED MODELS ON THE HELD-OUT TEST SET
    # ======================================================================
    print("\n\n=== Tuned models: held-out test set results ===")
    test_summary = []
    roc_data = {}
    for name, (model, Xtr, Xte) in tuned_models.items():
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]
        test_auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, test_auc)

        row = {
            "model": name, "test_auc": test_auc,
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
        }
        test_summary.append(row)
        print(f"{name:22s} AUC={row['test_auc']:.4f}  Acc={row['test_accuracy']:.4f}  "
              f"Prec={row['test_precision']:.4f}  Rec={row['test_recall']:.4f}  F1={row['test_f1']:.4f}")

    test_df = pd.DataFrame(test_summary).sort_values("test_auc", ascending=False)
    test_df.to_csv("model_comparison_tuned_test_results.csv", index=False)
    print("\nSaved model_comparison_tuned_test_results.csv")


    # ======================================================================
    # 4. TUNED VS UNTUNED COMPARISON
    # ======================================================================
    untuned_df = pd.read_csv("model_comparison_test_results.csv")
    compare = untuned_df[["model", "test_auc"]].rename(columns={"test_auc": "untuned_auc"}).merge(
        test_df[["model", "test_auc"]].rename(columns={"test_auc": "tuned_auc"}), on="model"
    )
    compare["improvement"] = compare["tuned_auc"] - compare["untuned_auc"]
    compare = compare.sort_values("tuned_auc", ascending=False)
    compare.to_csv("tuning_improvement_comparison.csv", index=False)
    print("\n=== Tuned vs untuned AUC comparison ===")
    print(compare.to_string(index=False))
    print("\nSaved tuning_improvement_comparison.csv")


    # ======================================================================
    # 5. IDENTIFY NEW BEST MODEL AND BEST GB VARIANT
    # ======================================================================
    best_overall = test_df.iloc[0]
    print(f"\n=== Best overall model (tuned): {best_overall['model']} "
          f"(test AUC={best_overall['test_auc']:.4f}) ===")

    gb_variants = test_df[test_df["model"].isin(["XGBoost", "LightGBM", "CatBoost"])]
    best_gb = gb_variants.sort_values("test_auc", ascending=False).iloc[0]
    print(f"=== Best gradient boosting variant (tuned): {best_gb['model']} "
          f"(test AUC={best_gb['test_auc']:.4f}) ===")


    # ======================================================================
    # 6. ROC CURVE (TUNED MODELS)
    # ======================================================================
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = matplotlib.colormaps["viridis"](np.linspace(0, 0.9, len(roc_data)))
    for (name, (fpr, tpr, auc_val)), color in zip(roc_data.items(), colors):
        ax.plot(fpr, tpr, color=color, linewidth=1.6, label=f"{name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig("model_comparison_tuned_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved model_comparison_tuned_roc.png")

    # Save tuned models for the sensitivity analysis script
    with open("tuned_models.pkl", "wb") as f:
        pickle.dump({name: model for name, (model, _, _) in tuned_models.items()}, f)
    print("Saved tuned_models.pkl")

    print("\n=== Hyperparameter tuning complete ===")



def run_shap_analysis():
    """
    SHAP ANALYSIS
    """
    print("\n" + "="*70)
    print("  SHAP ANALYSIS")
    print("="*70)

    warnings.filterwarnings("ignore")
    matplotlib.rcParams['font.size'] = 9

    # Portable data directory: assumes this script sits in scripts/ and
    # input/output files live in a sibling data/ folder (see README.md)
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.chdir(DATA_DIR)
    RANDOM_STATE = 42

    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").iloc[:, 0]
    selected_features = pd.read_csv("lasso_selected_features.csv").iloc[:, 0].tolist()

    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    with open("tuned_models.pkl", "rb") as f:
        tuned_models = pickle.load(f)


    # ======================================================================
    # 1. USE THE TUNED MODELS (unscaled features for both -- XGBoost is
    #    scale-invariant anyway, and using unscaled features for LR keeps
    #    its SHAP values on the same footing as the odds-ratio analysis)
    # ======================================================================
    # LR: refit at its tuned hyperparameters on UNSCALED features (the
    # pickled version was tuned on scaled features, needed for fair
    # comparison against other models in that stage -- refitting here keeps
    # SHAP consistent with the OR/forest-plot analysis, which also uses
    # unscaled features)
    lr = LogisticRegression(**tuned_models["Logistic Regression"].get_params())
    lr.fit(X_train_sel, y_train)

    # XGBoost: already fit on unscaled features during tuning, use directly
    xgb = tuned_models["XGBoost"]


    # ======================================================================
    # 2. SHAP VALUES
    # ======================================================================
    # LinearExplainer for LR (exact for linear models); explicitly using the
    # full training set as background rather than SHAP's default 100-sample
    # subsample, since our dataset is small enough to afford this.
    masker = shap.maskers.Independent(X_train_sel, max_samples=X_train_sel.shape[0])
    explainer_lr = shap.LinearExplainer(lr, masker)
    shap_values_lr = explainer_lr.shap_values(X_test_sel)

    # TreeExplainer for XGBoost (exact for tree ensembles, no background needed)
    explainer_xgb = shap.TreeExplainer(xgb)
    shap_values_xgb = explainer_xgb.shap_values(X_test_sel)

    print("LR SHAP values shape:", shap_values_lr.shape)
    print("XGBoost SHAP values shape:", shap_values_xgb.shape)

    # Human-readable labels (same mapping used in the odds-ratio/forest-plot
    # analysis) for the summary plots, instead of raw dummy variable names
    label_map = {
        "education_1.0": "Education: Primary", "education_2.0": "Education: Secondary",
        "region_2.0": "Region: Afar", "region_4.0": "Region: Oromia",
        "region_5.0": "Region: Somali", "region_6.0": "Region: Benishangul-Gumuz",
        "region_7.0": "Region: Central Ethiopia", "region_8.0": "Region: Sidama",
        "region_9.0": "Region: South West Ethiopia", "region_10.0": "Region: South Ethiopia",
        "region_13.0": "Region: Harari",
        "literacy_2.0": "Literacy: Reads part of sentence",
        "literacy_3.0": "Literacy: Reads whole sentence",
        "literacy_4.0": "Literacy: Other/Unknown",
        "media_cat_1.0": "Media exposure: Low", "media_cat_2.0": "Media exposure: Moderate",
        "media_cat_3.0": "Media exposure: High",
        "wealth_2.0": "Wealth: Poorer", "wealth_3.0": "Wealth: Middle",
        "wealth_4.0": "Wealth: Richer", "wealth_5.0": "Wealth: Richest",
        "insurance_1.0": "Health insurance: Yes",
        "internet_use_1.0": "Internet use: Last 12 months",
        "toilet_type_2.0": "Toilet: Improved pit latrine",
        "toilet_type_4.0": "Toilet: No facility/Other",
        "religion_3.0": "Religion: Muslim",
        "water_source_3.0": "Water source: Unimproved",
        "currently_working_1.0": "Currently working: Yes",
        "hh_size": "Household size (continuous)",
    }
    display_names = [label_map[f] for f in selected_features]


    # ======================================================================
    # 3. SUMMARY (BEESWARM) PLOTS
    # ======================================================================
    fig = plt.figure(figsize=(10, 10))
    shap.summary_plot(shap_values_lr, X_test_sel, feature_names=display_names, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_lr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved shap_summary_lr.png")

    fig = plt.figure(figsize=(10, 10))
    shap.summary_plot(shap_values_xgb, X_test_sel, feature_names=display_names, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_xgb.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved shap_summary_xgb.png")


    # ======================================================================
    # 4. MEAN |SHAP| BAR PLOTS (simpler ranked-importance view)
    # ======================================================================
    def mean_abs_shap_plot(shap_values, feature_names, title_model, filename):
        mean_abs = np.abs(shap_values).mean(axis=0)
        ranking = pd.Series(mean_abs, index=feature_names).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, 9))
        norm = matplotlib.colors.Normalize(vmin=ranking.min(), vmax=ranking.max())
        colors = matplotlib.colormaps["viridis"](norm(ranking.values))
        ax.barh(ranking.index, ranking.values, color=colors)
        ax.set_xlabel("Mean |SHAP value|")
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {filename}")
        return ranking.sort_values(ascending=False)

    ranking_lr = mean_abs_shap_plot(shap_values_lr, display_names,
                                      "Logistic Regression", "shap_importance_lr.png")
    ranking_xgb = mean_abs_shap_plot(shap_values_xgb, display_names,
                                        "XGBoost", "shap_importance_xgb.png")


    # ======================================================================
    # 5. CROSS-MODEL COMPARISON TABLE
    # ======================================================================
    comparison = pd.DataFrame({
        "LR_mean_abs_shap": ranking_lr,
        "LR_rank": ranking_lr.rank(ascending=False).astype(int),
    }).join(pd.DataFrame({
        "XGBoost_mean_abs_shap": ranking_xgb,
        "XGBoost_rank": ranking_xgb.rank(ascending=False).astype(int),
    }))
    comparison["rank_difference"] = (comparison["LR_rank"] - comparison["XGBoost_rank"]).abs()
    comparison = comparison.sort_values("LR_rank")
    comparison.to_csv("shap_lr_vs_xgb_comparison.csv")
    print("\nSaved shap_lr_vs_xgb_comparison.csv")
    print("\n=== Top 10 features, LR vs XGBoost SHAP ranking ===")
    print(comparison.head(10).to_string())

    print("\n=== SHAP analysis complete ===")



def run_sensitivity_analysis():
    """
    SENSITIVITY ANALYSIS (10 REPEATED SPLITS)
    """
    print("\n" + "="*70)
    print("  SENSITIVITY ANALYSIS (10 REPEATED SPLITS)")
    print("="*70)

    warnings.filterwarnings("ignore")
    matplotlib.rcParams['font.size'] = 9

    # Portable data directory: assumes this script sits in scripts/ and
    # input/output files live in a sibling data/ folder (see README.md)
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.chdir(DATA_DIR)

    # ======================================================================
    # 1. REASSEMBLE THE FULL DATASET (so we can redraw genuinely fresh splits)
    # ======================================================================
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").iloc[:, 0]
    y_test = pd.read_csv("y_test.csv").iloc[:, 0]
    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)

    selected_features = pd.read_csv("lasso_selected_features.csv").iloc[:, 0].tolist()
    X_full_sel = X_full[selected_features]

    print(f"Full dataset: {len(X_full_sel)} rows")

    with open("tuned_models.pkl", "rb") as f:
        tuned_models = pickle.load(f)

    # Which models need scaled features (LR, MLP) vs raw (tree-based)
    needs_scaling = {"Logistic Regression": True, "Random Forest": False, "XGBoost": False,
                      "LightGBM": False, "CatBoost": False, "MLP": True}


    # ======================================================================
    # 2. RE-FIT EACH MODEL (AT ITS TUNED HYPERPARAMETERS) ACROSS 10 FRESH
    #    RANDOM SPLITS, TRACKING TEST AUC EACH TIME
    # ======================================================================
    N_REPEATS = 10
    results = {name: [] for name in tuned_models}

    for seed in range(N_REPEATS):
        Xtr, Xte, ytr, yte = train_test_split(
            X_full_sel, y_full, test_size=0.20, stratify=y_full, random_state=seed
        )

        scaler = StandardScaler()
        Xtr_scaled = pd.DataFrame(scaler.fit_transform(Xtr), columns=selected_features)
        Xte_scaled = pd.DataFrame(scaler.transform(Xte), columns=selected_features)

        for name, model in tuned_models.items():
            # Fresh clone at the same (already-tuned) hyperparameters, refit
            # on this seed's training split
            model_fresh = type(model)(**model.get_params())
            if needs_scaling[name]:
                model_fresh.fit(Xtr_scaled, ytr)
                y_prob = model_fresh.predict_proba(Xte_scaled)[:, 1]
            else:
                model_fresh.fit(Xtr, ytr)
                y_prob = model_fresh.predict_proba(Xte)[:, 1]
            auc = roc_auc_score(yte, y_prob)
            results[name].append(auc)

        print(f"Seed {seed} complete.")


    # ======================================================================
    # 3. SUMMARIZE STABILITY ACROSS SPLITS
    # ======================================================================
    summary = []
    for name, aucs in results.items():
        aucs = np.array(aucs)
        summary.append({
            "model": name, "mean_auc": aucs.mean(), "sd_auc": aucs.std(),
            "min_auc": aucs.min(), "max_auc": aucs.max(), "range": aucs.max() - aucs.min(),
        })
    summary_df = pd.DataFrame(summary).sort_values("mean_auc", ascending=False)
    summary_df.to_csv("sensitivity_analysis_summary.csv", index=False)
    print("\n=== Sensitivity analysis: AUC across 10 random splits ===")
    print(summary_df.to_string(index=False))
    print("\nSaved sensitivity_analysis_summary.csv")

    # How often was each model the single best performer across the 10 splits?
    raw_results = pd.DataFrame(results)
    raw_results.to_csv("sensitivity_analysis_raw.csv", index=False)
    win_counts = raw_results.idxmax(axis=1).value_counts()
    print("\n=== Number of splits (out of 10) each model performed best ===")
    print(win_counts.to_string())
    win_counts.to_csv("sensitivity_win_counts.csv", header=["times_best"])
    print("\nSaved sensitivity_analysis_raw.csv, sensitivity_win_counts.csv")


    # ======================================================================
    # 4. BOX PLOT OF AUC DISTRIBUTION ACROSS SPLITS
    # ======================================================================
    order = summary_df["model"].tolist()
    fig, ax = plt.subplots(figsize=(9, 6))
    box_data = [results[m] for m in order]
    bp = ax.boxplot(box_data, tick_labels=order, patch_artist=True, showmeans=True)

    colors = matplotlib.colormaps["viridis"](np.linspace(0, 0.9, len(order)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Test AUC (across 10 random splits)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("sensitivity_analysis_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved sensitivity_analysis_boxplot.png")

    print("\n=== Sensitivity analysis complete ===")



def run_calibration_analysis():
    """
    CALIBRATION ANALYSIS
    """
    print("\n" + "="*70)
    print("  CALIBRATION ANALYSIS")
    print("="*70)

    warnings.filterwarnings("ignore")
    matplotlib.rcParams['font.size'] = 9

    # Portable data directory: assumes this script sits in scripts/ and
    # input/output files live in a sibling data/ folder (see README.md)
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.chdir(DATA_DIR)

    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").iloc[:, 0]
    y_test = pd.read_csv("y_test.csv").iloc[:, 0]
    selected_features = pd.read_csv("lasso_selected_features.csv").iloc[:, 0].tolist()
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_sel), columns=selected_features)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_sel), columns=selected_features)

    with open("tuned_models.pkl", "rb") as f:
        tuned_models = pickle.load(f)

    needs_scaling = {"Logistic Regression": True, "Random Forest": False, "XGBoost": False,
                      "LightGBM": False, "CatBoost": False, "MLP": True}


    # ======================================================================
    # 1. BRIER SCORE FOR EACH MODEL
    # ======================================================================
    brier_scores = {}
    all_probs = {}
    for name, model in tuned_models.items():
        Xte = X_test_scaled if needs_scaling[name] else X_test_sel
        y_prob = model.predict_proba(Xte)[:, 1]
        brier_scores[name] = brier_score_loss(y_test, y_prob)
        all_probs[name] = y_prob

    brier_df = pd.DataFrame(list(brier_scores.items()), columns=["model", "brier_score"])
    brier_df = brier_df.sort_values("brier_score")
    brier_df.to_csv("calibration_brier_scores.csv", index=False)
    print("=== Brier scores (lower = better calibrated + discriminating) ===")
    print(brier_df.to_string(index=False))
    print("\nSaved calibration_brier_scores.csv")


    # ======================================================================
    # 2. CALIBRATION CURVES (5 bins, given N=569 test set -- finer binning
    #    was tested and found too noisy per-bin to be informative)
    # ======================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), height_ratios=[3, 1], sharex=True)
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")

    colors = matplotlib.colormaps["viridis"](np.linspace(0, 0.9, len(tuned_models)))
    for (name, y_prob), color in zip(all_probs.items(), colors):
        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=5, strategy="quantile")
        ax1.plot(mean_pred, frac_pos, marker="o", markersize=5, color=color, linewidth=1.6,
                  label=f"{name} (Brier={brier_scores[name]:.3f})")

    ax1.set_ylabel("Observed frequency")
    ax1.legend(fontsize=8, loc="upper left")

    # Companion histogram: distribution of predicted probabilities for the
    # final chosen model (Logistic Regression)
    ax2.hist(all_probs["Logistic Regression"], bins=20, color=colors[0], alpha=0.7)
    ax2.set_xlabel("Mean predicted probability")
    ax2.set_ylabel("Count\n(Logistic Regression)")

    plt.tight_layout()
    plt.savefig("calibration_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved calibration_curves.png")

    print("\n=== Calibration analysis complete ===")



def run_decision_curve_analysis():
    """
    DECISION CURVE ANALYSIS (DCA)
    """
    print("\n" + "="*70)
    print("  DECISION CURVE ANALYSIS (DCA)")
    print("="*70)

    warnings.filterwarnings("ignore")
    matplotlib.rcParams['font.size'] = 9

    # Portable data directory: assumes this script sits in scripts/ and
    # input/output files live in a sibling data/ folder (see README.md)
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.chdir(DATA_DIR)

    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").iloc[:, 0]
    y_test = pd.read_csv("y_test.csv").iloc[:, 0]
    selected_features = pd.read_csv("lasso_selected_features.csv").iloc[:, 0].tolist()
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_sel), columns=selected_features)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_sel), columns=selected_features)

    with open("tuned_models.pkl", "rb") as f:
        tuned_models = pickle.load(f)

    needs_scaling = {"Logistic Regression": True, "Random Forest": False, "XGBoost": False,
                      "LightGBM": False, "CatBoost": False, "MLP": True}


    def net_benefit(y_true, y_prob, threshold):
        n = len(y_true)
        pred_pos = y_prob >= threshold
        tp = np.sum(pred_pos & (y_true == 1))
        fp = np.sum(pred_pos & (y_true == 0))
        return (tp / n) - (fp / n) * (threshold / (1 - threshold))


    # ======================================================================
    # 1. COMPUTE NET BENEFIT ACROSS THE THRESHOLD RANGE FOR EVERY MODEL
    # ======================================================================
    thresholds = np.linspace(0.01, 0.80, 80)
    prevalence = y_test.mean()
    print(f"Test set prevalence: {prevalence:.4f}")

    nb_treat_all = np.array([prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds])
    nb_treat_none = np.zeros_like(thresholds)

    model_nb = {}
    for name, model in tuned_models.items():
        Xte = X_test_scaled if needs_scaling[name] else X_test_sel
        y_prob = model.predict_proba(Xte)[:, 1]
        model_nb[name] = np.array([net_benefit(y_test.values, y_prob, t) for t in thresholds])


    # ======================================================================
    # 2. DCA PLOT
    # ======================================================================
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(thresholds, nb_treat_all, color="black", linestyle="--", linewidth=1.2, label="Vaccinate all")
    ax.axhline(0, color="gray", linestyle=":", linewidth=1.2, label="Vaccinate none")

    colors = matplotlib.colormaps["viridis"](np.linspace(0, 0.9, len(tuned_models)))
    for (name, nb), color in zip(model_nb.items(), colors):
        ax.plot(thresholds, nb, color=color, linewidth=1.8, label=name)

    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_ylim(-0.1, 0.6)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig("decision_curve_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved decision_curve_analysis.png")


    # ======================================================================
    # 3. NET BENEFIT AT SELECTED, CLINICALLY-RELEVANT THRESHOLDS
    # ======================================================================
    key_thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]
    rows = []
    for kt in key_thresholds:
        idx = np.argmin(np.abs(thresholds - kt))
        row = {"threshold": kt, "Vaccinate all": nb_treat_all[idx], "Vaccinate none": 0.0}
        for name, nb in model_nb.items():
            row[name] = nb[idx]
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv("dca_net_benefit_summary.csv", index=False)
    print("\n=== Net benefit at key thresholds ===")
    print(summary_df.round(4).to_string(index=False))
    print("\nSaved dca_net_benefit_summary.csv")

    print("\n=== Decision curve analysis complete ===")



def run_confusion_matrices_tuned():
    """
    CONFUSION MATRICES (TUNED MODELS)
    """
    print("\n" + "="*70)
    print("  CONFUSION MATRICES (TUNED MODELS)")
    print("="*70)

    warnings.filterwarnings("ignore")
    matplotlib.rcParams['font.size'] = 9

    # Portable data directory: assumes this script sits in scripts/ and
    # input/output files live in a sibling data/ folder (see README.md)
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.chdir(DATA_DIR)

    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").iloc[:, 0]
    y_test = pd.read_csv("y_test.csv").iloc[:, 0]
    selected_features = pd.read_csv("lasso_selected_features.csv").iloc[:, 0].tolist()
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_sel), columns=selected_features)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_sel), columns=selected_features)

    with open("tuned_models.pkl", "rb") as f:
        tuned_models = pickle.load(f)

    needs_scaling = {"Logistic Regression": True, "Random Forest": False, "XGBoost": False,
                      "LightGBM": False, "CatBoost": False, "MLP": True}


    # ======================================================================
    # 1. COMPUTE CONFUSION MATRIX FOR EACH TUNED MODEL
    # ======================================================================
    cm_data = {}
    for name, model in tuned_models.items():
        Xte = X_test_scaled if needs_scaling[name] else X_test_sel
        y_pred = model.predict(Xte)
        cm_data[name] = confusion_matrix(y_test, y_pred)

    print("Confusion matrices computed for:", list(cm_data.keys()))


    # ======================================================================
    # 2. PLOT (shared color scale across all 6 panels)
    # ======================================================================
    global_max = max(cm.max() for cm in cm_data.values())

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for ax, (name, cm) in zip(axes, cm_data.items()):
        im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=global_max)
        for i in range(2):
            for j in range(2):
                pct = 100 * cm[i, j] / cm.sum()
                color = "white" if cm[i, j] < global_max * 0.6 else "black"
                ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)", ha="center", va="center",
                         color=color, fontsize=9)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Not vacc.", "Vacc."], fontsize=8)
        ax.set_yticklabels(["Not vacc.", "Vacc."], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)
        ax.set_title(name, fontsize=9, fontweight="bold")

    fig.subplots_adjust(left=0.06, right=0.90, top=0.94, bottom=0.06, wspace=0.4, hspace=0.4)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Count")
    plt.savefig("confusion_matrices_tuned.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved confusion_matrices_tuned.png")

    print("\n=== Confusion matrices (tuned models) complete ===")

# ============================================================================
# MAIN: run the full pipeline in order
# ============================================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    run_eda()
    run_preprocessing_lasso()
    run_table1_characteristics()
    run_model_comparison()
    run_lr_odds_ratios()
    run_hyperparameter_tuning()
    run_shap_analysis()
    run_sensitivity_analysis()
    run_calibration_analysis()
    run_decision_curve_analysis()
    run_confusion_matrices_tuned()

    print("\n" + "="*70)
    print("  FULL PIPELINE COMPLETE")
    print("="*70)
