"""
HPV Vaccination Uptake in Ethiopia -- Survey-Weighted Analysis and Explainable Machine Learning Pipeline
=========================================================================================================
Full analysis code:
"An Explainable Machine Learning Approach to Human Papillomavirus
Vaccination Uptake Among Adolescent Girls in Ethiopia"
2024--25 Ethiopia Demographic and Health Survey (DHS-8), girls aged 15-17.

This single file runs the complete pipeline end to end, in order:
   0. Build the survey-design file (weights, strata, clusters) from the DHS IR file
   1. Exploratory data analysis (unweighted bar panels, Cramer's V matrix, VIF)
   2. Survey-weighted prevalence, regional shares, "don't know" sensitivity, vaccination card
   3. Preprocessing + LASSO feature selection (1-SE rule) on the development split
   4. Table 1: sample characteristics (sample and survey-weighted)
   5. Untuned comparison of six ML algorithms (development split)
   6. Survey-weighted logistic regression: odds ratios, all 14 regions, wealth sensitivity, forest plot
   7. Hyperparameter tuning (RandomizedSearchCV) for all six models (development split)
   8. SHAP analysis (logistic regression + XGBoost)
   9. Nested repeated cross-validation of the complete pipeline (5x5), paired tests, LASSO stability
  10. Calibration (slope, intercept, deciles with 95% intervals) from nested out-of-fold predictions
  11. Decision curve analysis (event = unvaccinated; sample and survey-weighted)
  12. Confusion matrices for all tuned models (development split)

Data (not redistributed; request at https://dhsprogram.com/data/available-datasets.cfm):
  data/hpv_vaccination_clean.dta   cleaned analysis file (2,842 girls)
  data/ETIR8AFL.dta                EDHS 2024-25 individual recode (weights, PSU, strata)
The data folder is the folder called "data" next to this script (or the current folder's "data"
when run in a notebook). Override with the environment variable HPV_DATA_DIR; the IR file
location can be set with EDHS_IR_DTA.

Usage:
    pip install -r requirements.txt
    python hpv_vaccination_uptake_ml.py

Set RUN_NESTED_CV = False below to skip the slow nested cross-validation (steps 9-11 then reuse the
nested_cv_*.csv files already in data/). All figures (.png) and tables (.csv) are written to data/.
Reproducibility: logistic-regression and survey-weighted results reproduce exactly; tree-based models
can differ in the 2nd-3rd decimal across package versions and CPU thread counts.
"""

import os
import time
import warnings
import pickle
import pyreadstat
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from scipy.stats import chi2_contingency, loguniform, randint, uniform
from sklearn.model_selection import (train_test_split, StratifiedKFold, RepeatedStratifiedKFold,
                                      cross_val_score, cross_validate, RandomizedSearchCV)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_curve, auc, roc_auc_score, confusion_matrix,
                              accuracy_score, precision_score, recall_score,
                              f1_score, brier_score_loss, log_loss)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import shap

# ----------------------------------------------------------------------------
# Switches for the slow nested cross-validation (manuscript: 5 repeats x 20 draws)
# ----------------------------------------------------------------------------
RUN_NESTED_CV = True     # False = reuse nested_cv_*.csv already in data/
N_REPEATS = 5            # repeats of 5-fold outer CV (25 outer test folds)
N_SEARCH_DRAWS = 20      # random-search draws per model and outer fold

REGION_NAMES = {1: "Tigray", 2: "Afar", 3: "Amhara", 4: "Oromia", 5: "Somali",
                6: "Benishangul-Gumuz", 7: "Central Ethiopia", 8: "Sidama",
                9: "South West Ethiopia", 10: "South Ethiopia", 12: "Gambella",
                13: "Harari", 14: "Addis Ababa", 15: "Dire Dawa"}
ALL_CAT_VARS = ["education", "wealth", "residence", "region", "religion", "insurance",
                "hh_head_sex", "media_cat", "literacy", "mobile_owner", "internet_use",
                "currently_working", "access_permission", "access_money", "access_distance",
                "water_source", "toilet_type", "electricity", "age"]
CONT_VARS = ["hh_head_age", "hh_size"]


def get_data_dir():
    """data/ folder next to this script; current folder's data/ (or the folder itself) in a notebook."""
    if os.environ.get("HPV_DATA_DIR"):
        return os.path.abspath(os.environ["HPV_DATA_DIR"])
    try:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    except NameError:                                   # notebook / interactive session
        d = os.path.join(os.getcwd(), "data")
        return d if os.path.isdir(d) else os.getcwd()


# ============================================================================
# SURVEY-DESIGN TOOLS (Taylor linearization, with-replacement stratified cluster variance)
# ============================================================================
class Design:
    """Complex-survey design: weights, PSU (cluster) and strata. df = PSUs - strata."""
    def __init__(self, w, psu, strata):
        self.w = np.asarray(w, float); self.psu = np.asarray(psu); self.st = np.asarray(strata)
        self.key = pd.Series(list(zip(self.st, self.psu))).astype("category").cat.codes.values
        self.nstr = len(np.unique(self.st)); self.npsu = len(np.unique(self.key))
        self.df = self.npsu - self.nstr

    def total_cov(self, U):
        """Variance of the total of influence rows U (n x k)."""
        U = np.asarray(U, float).reshape(len(self.w), -1)
        frame = pd.DataFrame(U); frame["k"] = self.key; frame["s"] = self.st
        P = frame.groupby(["s", "k"]).sum().reset_index()
        V = np.zeros((U.shape[1],) * 2)
        for _, g in P.groupby("s"):
            A = g.drop(columns=["s", "k"]).values; n = len(A)
            if n < 2: continue
            A = A - A.mean(0); V += n / (n - 1) * A.T @ A
        return V

    def mean(self, y):
        """Weighted proportion and its design-based standard error."""
        y = np.asarray(y, float); p = (self.w * y).sum() / self.w.sum()
        u = self.w * (y - p) / self.w.sum()
        return p, np.sqrt(self.total_cov(u)[0, 0])


def svyglm_logit(X, y, d, names):
    """Survey-weighted logistic regression with design-based (Taylor) covariance and t(df) intervals."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    m = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=d.w / d.w.mean()).fit()
    b = m.params; p = m.predict(X); W = d.w * p * (1 - p)
    Ai = np.linalg.inv(X.T @ (W[:, None] * X))
    U = (d.w * (y - p))[:, None] * X
    V = Ai @ d.total_cov(U) @ Ai
    se = np.sqrt(np.diag(V)); pv = 2 * stats.t.sf(abs(b / se), d.df); q = stats.t.ppf(.975, d.df)
    out = pd.DataFrame({"term": names, "coef": b, "se": se, "OR": np.exp(b),
                        "lo": np.exp(b - q * se), "hi": np.exp(b + q * se), "p": pv})
    return out, V


def load_analysis_data():
    """Design file (weights/PSU/strata) + cleaned analysis file + one-hot feature matrix.
    Run from the data/ folder. Rows of the two files are checked to line up one-to-one."""
    ir = pd.read_csv("hpv_design.csv")
    df, meta = pyreadstat.read_dta("hpv_vaccination_clean.dta"); df = df.astype(float)
    assert (df.age.values == ir.v012.values).all() and (df.region.values == ir.v024.values).all(), \
        "hpv_design.csv does not line up with hpv_vaccination_clean.dta -- re-run run_build_design()"
    assert df.water_source.value_counts().sort_index().tolist() == [1094, 814, 801, 133], \
        "water_source coding differs from the expected recode (1 piped, 2 other improved, 3 unimproved, 4 other)"
    X = pd.get_dummies(df[CONT_VARS + ALL_CAT_VARS], columns=ALL_CAT_VARS, drop_first=True).astype(float)
    return ir, df, X


def run_build_design():
    """
    STEP 0. SURVEY-DESIGN FILE
    """
    print("\n" + "="*70)
    print("  SURVEY-DESIGN FILE (weights, strata, clusters)")
    print("="*70)

    DATA_DIR = get_data_dir()
    os.chdir(DATA_DIR)
    ir_path = os.environ.get("EDHS_IR_DTA", os.path.join(DATA_DIR, "ETIR8AFL.dta"))

    if not os.path.exists(ir_path):
        if os.path.exists("hpv_design.csv") and os.path.exists("hpv_ir_raw.csv"):
            print(f"IR file not found ({ir_path}); using the existing hpv_design.csv / hpv_ir_raw.csv.")
            return
        raise FileNotFoundError(f"Place the EDHS individual recode at {ir_path} "
                                "(or set EDHS_IR_DTA) to build the survey-design file.")

    # girls aged 15-17 who answered the HPV module (hpv03: 0 no, 1 yes, 8 don't know)
    cols = ["caseid", "v005", "v012", "v021", "v022", "v023", "v024", "v025", "v106", "v190",
            "hpv03", "hpv04", "hpv05", "hpv06"]
    ir = pd.read_stata(ir_path, columns=cols, convert_categoricals=False)
    g = ir[ir.hpv03.isin([0, 1, 8])].copy()
    print(f"HPV respondents: {len(g)}  (don't know = {(g.hpv03 == 8).sum()})")
    g.to_csv("hpv_ir_raw.csv", index=False)

    g = g[g.hpv03.isin([0, 1])].reset_index(drop=True)
    df, meta = pyreadstat.read_dta("hpv_vaccination_clean.dta")
    assert len(g) == len(df) == 2842
    assert (df.age.values == g.v012.values).all() and (df.region.values == g.v024.values).all()
    assert (df.hpv_vaccinated.values == g.hpv03.values).all()
    g["wt"] = g.v005 / 1e6                       # DHS sampling weight
    g.to_csv("hpv_design.csv", index=False)
    print("Alignment with the clean analysis file verified -> saved hpv_design.csv, hpv_ir_raw.csv")


def run_survey_prevalence():
    """
    SURVEY-WEIGHTED PREVALENCE AND SENSITIVITY ANALYSES
    """
    print("\n" + "="*70)
    print("  SURVEY-WEIGHTED PREVALENCE")
    print("="*70)

    warnings.filterwarnings("ignore")
    matplotlib.rcParams['font.size'] = 9
    DATA_DIR = get_data_dir()
    os.chdir(DATA_DIR)
    V = matplotlib.colormaps["viridis"]

    ir, df, X = load_analysis_data()
    y = ir.hpv03.values.astype(int)
    d = Design(ir.wt, ir.v021, ir.v022)
    q = stats.t.ppf(.975, d.df)

    # ======================================================================
    # 1. NATIONAL PREVALENCE AND DESIGN EFFECT
    # ======================================================================
    print(f"PSUs {d.npsu}, strata {d.nstr}, df {d.df}")
    p, se = d.mean(y)
    lg = np.log(p / (1 - p)); s_lg = se / (p * (1 - p))
    print(f"Unweighted {100*y.mean():.2f}%   Weighted {100*p:.2f}% (SE {100*se:.2f})")
    print(f"  95% CI (Wald)  {100*(p - q*se):.2f}-{100*(p + q*se):.2f}")
    print(f"  95% CI (logit) {100/(1+np.exp(-(lg - q*s_lg))):.2f}-{100/(1+np.exp(-(lg + q*s_lg))):.2f}")
    print(f"Design effect: {(se / np.sqrt(p * (1 - p) / len(y)))**2:.2f}")

    # ======================================================================
    # 2. REGIONAL PREVALENCE: SAMPLE SHARE VS WEIGHTED POPULATION SHARE
    # ======================================================================
    rows = []
    for r, name in REGION_NAMES.items():
        m = (ir.v024 == r).values; dd = Design(ir.wt[m], ir.v021[m], ir.v022[m])
        pp, ss = dd.mean(y[m])
        rows.append([name, m.sum(), 100*m.mean(), 100*ir.wt[m].sum()/ir.wt.sum(), 100*y[m].mean(), 100*pp, 100*ss])
    reg = pd.DataFrame(rows, columns=["region", "n", "sample_%", "weighted_pop_%", "unwtd_uptake", "wtd_uptake", "se"]).round(1)
    reg.to_csv("region_weighted_prevalence.csv", index=False)
    print("\n", reg.to_string(index=False))
    print("Saved region_weighted_prevalence.csv")

    P = reg.sort_values("wtd_uptake")
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 4.2), sharey=True, gridspec_kw={"width_ratios": [1.5, 1]})
    yy = np.arange(len(P)); ax = axs[0]
    ax.hlines(yy, P.wtd_uptake - 1.96*P.se, P.wtd_uptake + 1.96*P.se, color=V(0.25), lw=1.4)
    ax.plot(P.wtd_uptake, yy, "o", color=V(0.25), ms=4, label="Weighted (95% CI)")
    ax.plot(P.unwtd_uptake, yy, "s", mfc="none", mec=V(0.75), ms=4, label="Unweighted")
    ax.axvline(100*p, color="k", ls="--", lw=.7)
    ax.set_xlabel("HPV vaccination uptake (%)"); ax.set_yticks(yy); ax.set_yticklabels(P.region)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.text(100*p + 1, len(P) - .6, f"National {100*p:.1f}%", fontsize=7)
    ax = axs[1]
    ax.barh(yy - .2, P["sample_%"], .38, color=V(0.55), label="Share of sample")
    ax.barh(yy + .2, P["weighted_pop_%"], .38, color=V(0.9), label="Weighted population share")
    ax.set_xlabel("Share (%)"); ax.legend(frameon=False, fontsize=7, loc="lower right")
    plt.tight_layout(); plt.savefig("region_weighted.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print("Saved region_weighted.png")

    # ======================================================================
    # 3. SENSITIVITY: "DON'T KNOW" ANSWERS CLASSIFIED AS NOT VACCINATED
    # ======================================================================
    if os.path.exists("hpv_ir_raw.csv"):
        raw = pd.read_csv("hpv_ir_raw.csv"); raw["wt"] = raw.v005 / 1e6
        yk = (raw.hpv03 == 1).astype(float).values
        dk = Design(raw.wt, raw.v021, raw.v022); pk, sk = dk.mean(yk); qk = stats.t.ppf(.975, dk.df)
        print(f"\nDon't know as not vaccinated (n={len(raw)}): weighted {100*pk:.2f}% "
              f"({100*(pk - qk*sk):.2f}-{100*(pk + qk*sk):.2f}); unweighted {100*yk.mean():.2f}%")
        k = (raw.hpv03 == 8)
        print(f"  Don't know: n={k.sum()}, weighted share {100*raw.wt[k].sum()/raw.wt.sum():.2f}%")

        # share of vaccinated girls reporting that a vaccination card was ever held (hpv04)
        if "hpv04" in raw.columns:
            g = raw[raw.hpv03 == 1]; dg = Design(g.wt, g.v021, g.v022)
            for v in [1, 2, 3]:
                pv_, sv_ = dg.mean((g.hpv04 == v).astype(float).values)
                print(f"  hpv04 == {v}: {100*pv_:.1f}% (SE {100*sv_:.1f})")

    print("\n=== Survey-weighted prevalence complete ===")


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
    DATA_DIR = get_data_dir()
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
    # 5. ASSOCIATION MATRIX (Cramer's V; replaces the Pearson correlation heatmap)
    # ======================================================================
    # Pearson correlations treat the nominal codes (region, religion, ...) as
    # numbers; Cramer's V is the appropriate pairwise association for
    # categorical variables (0 = none, 1 = perfect).
    def cramers_v(a, b):
        ct = pd.crosstab(a, b)
        chi2 = chi2_contingency(ct, correction=False)[0]
        k = min(ct.shape) - 1
        return np.sqrt(chi2 / (ct.values.sum() * k)) if k > 0 else 0.0

    V_VARS = [("Education", "education"), ("Wealth", "wealth"), ("Residence", "residence"),
              ("Region", "region"), ("Religion", "religion"), ("Insurance", "insurance"),
              ("Media", "media_cat"), ("Literacy", "literacy"), ("Internet", "internet_use"),
              ("Water", "water_source"), ("Toilet", "toilet_type"), ("Electricity", "electricity"),
              ("Working", "currently_working"), ("Vaccinated", "hpv_vaccinated")]
    labels = [v[0] for v in V_VARS]
    M = np.array([[1.0 if a == b else cramers_v(df[va], df[vb])
                   for b, (_, vb) in enumerate(V_VARS)] for a, (_, va) in enumerate(V_VARS)])
    pd.DataFrame(M, index=labels, columns=labels).round(3).to_csv("cramers_v_matrix.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] < 0.6 else "black")
    plt.colorbar(im, fraction=0.046, label="Cramer's V")
    plt.tight_layout()
    plt.savefig("cramers_v_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved cramers_v_matrix.csv, cramers_v_matrix.png")


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
    # Pairwise association (Section 5) only catches two-variable relationships.
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
    # 8. CONTINUOUS VARIABLE DISTRIBUTIONS
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
    DATA_DIR = get_data_dir()
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
    TABLE 1: SAMPLE CHARACTERISTICS (sample and survey-weighted)
    """
    print("\n" + "="*70)
    print("  TABLE 1: SAMPLE CHARACTERISTICS")
    print("="*70)

    DATA_DIR = get_data_dir()
    os.chdir(DATA_DIR)

    df, meta = pyreadstat.read_dta("hpv_vaccination_clean.dta")
    n_total = len(df)
    print(f"N = {n_total}")

    # Survey design (weights, PSU, strata) for the weighted columns
    ir = pd.read_csv("hpv_design.csv")
    assert (df.age.values == ir.v012.values).all() and (df.region.values == ir.v024.values).all(), \
        "hpv_design.csv does not line up with hpv_vaccination_clean.dta -- re-run run_build_design()"
    y_all = df["hpv_vaccinated"].values.astype(float)
    d_all = Design(ir.wt, ir.v021, ir.v022)
    q = stats.t.ppf(.975, d_all.df)
    W = ir.wt.values


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

    rows, num_rows = [], []
    for var, disp_name, order in cat_vars_ordered:
        vmap = get_labels(var)
        grp = df.groupby(var)["hpv_vaccinated"].agg(["count", "mean"]).reset_index()
        grp["label"] = grp[var].map(vmap) if vmap else grp[var].astype(int).astype(str)
        if order:
            grp["label"] = pd.Categorical(grp["label"], categories=order, ordered=True)
            grp = grp.sort_values("label")
        rows.append({"characteristic": disp_name, "category": "", "n": "", "pct": "",
                     "vacc_pct": "", "wpct": "", "wvacc": ""})
        for _, r in grp.iterrows():
            m = (df[var].values == r[var])
            dd = Design(W[m], ir.v021.values[m], ir.v022.values[m])
            p, se = dd.mean(y_all[m])
            lg = np.log(p / (1 - p)); s = se / (p * (1 - p))        # logit-scale 95% CI
            lo, hi = 1 / (1 + np.exp(-(lg - q * s))), 1 / (1 + np.exp(-(lg + q * s)))
            wshare = 100 * W[m].sum() / W.sum()
            rows.append({
                "characteristic": "",
                "category": r["label"],
                "n": int(r["count"]),
                "pct": f"{100*r['count']/n_total:.1f}",
                "vacc_pct": f"{100*r['mean']:.1f}",
                "wpct": f"{wshare:.1f}",
                "wvacc": f"{100*p:.1f} ({100*lo:.1f}-{100*hi:.1f})",
            })
            num_rows.append(dict(variable=disp_name, level=str(r["label"]), n=int(r["count"]),
                                 pct_sample=100 * r["count"] / n_total, pct_weighted=wshare,
                                 unw_uptake=100 * r["mean"], wtd_uptake=100 * p, lo=100 * lo, hi=100 * hi))

    for var, disp_name in [("hh_head_age", "Household head age (years)"),
                            ("hh_size", "Household size")]:
        x = df[var].values.astype(float)
        wm = np.average(x, weights=W); wsd = np.sqrt(np.average((x - wm) ** 2, weights=W))
        rows.append({
            "characteristic": disp_name, "category": "Mean (SD)",
            "n": n_total, "pct": "",
            "vacc_pct": f"{x.mean():.1f} ({x.std(ddof=1):.1f})",
            "wpct": "", "wvacc": f"{wm:.1f} ({wsd:.1f})",
        })

    table1 = pd.DataFrame(rows)
    table1.columns = ["Characteristic", "Category", "N", "%", "% vaccinated / Mean (SD)",
                      "Weighted %", "Weighted % vaccinated (95% CI) / Mean (SD)"]
    table1.to_csv("table1_sample_characteristics.csv", index=False)
    pd.DataFrame(num_rows).round(1).to_csv("descriptive_weighted.csv", index=False)
    print("\nSaved table1_sample_characteristics.csv, descriptive_weighted.csv")
    print(table1.to_string(index=False))


    # ======================================================================
    # 3. LATEX VERSION (longtable, matching the cervical cancer paper's style)
    # ======================================================================
    hdr = (r"Characteristic & N & \% & \% vacc. / Mean (SD) & Weighted \% "
           r"& Weighted \% vacc. (95\% CI) / Mean (SD) \\")
    latex_lines = [
        r"\begin{longtable}{lccccc}",
        r"\caption{Sample characteristics and HPV vaccination uptake among girls aged 15--17 (unweighted and survey-weighted)}",
        r"\label{tab:descriptive} \\",
        r"\toprule",
        hdr,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        hdr,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for _, r in table1.iterrows():
        if r["Category"] == "":
            latex_lines.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{{r['Characteristic']}}}}} \\")
        else:
            cat = str(r["Category"]).replace("&", r"\&")
            latex_lines.append(rf"\hspace{{2mm}}{cat} & {r['N']} & {r['%']} & {r['% vaccinated / Mean (SD)']} "
                               rf"& {r['Weighted %']} & {r['Weighted % vaccinated (95% CI) / Mean (SD)']} \\")
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
    DATA_DIR = get_data_dir()
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

    DATA_DIR = get_data_dir()
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



    # ======================================================================
    # 5. SURVEY-WEIGHTED ANALYSIS (full sample, design-based variance)
    # ======================================================================
    # Sections 1-4 above (development training split, model-based CIs) are kept
    # for comparison with the first submission. Population-level inference
    # below uses all 2,842 girls with the DHS weights, strata and clusters
    # (Taylor linearization, t-intervals on PSUs - strata degrees of freedom),
    # with the same 29 LASSO-selected covariates.
    ir, dfa, Xa = load_analysis_data()
    ya = ir.hpv03.values.astype(int)
    d = Design(ir.wt, ir.v021, ir.v022)
    q = stats.t.ppf(.975, d.df)
    pd.set_option("display.width", 250)
    miss = [c for c in selected_features if c not in Xa.columns]
    assert not miss, f"selected features missing from the design matrix: {miss}"

    # --- 5a. 29-feature model: unweighted, unweighted + cluster-robust, survey-weighted ---
    Xs = sm.add_constant(Xa[selected_features].values.astype(float))
    names = ["const"] + selected_features
    m0 = sm.Logit(ya, Xs).fit(disp=0)
    m1 = sm.Logit(ya, Xs).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d.key})
    sv, _ = svyglm_logit(Xs, ya, d, names)
    res = pd.DataFrame({"term": names,
                        "OR_unw": np.exp(m0.params), "lo_unw": np.exp(m0.conf_int()[:, 0]), "hi_unw": np.exp(m0.conf_int()[:, 1]),
                        "lo_cl": np.exp(m1.conf_int()[:, 0]), "hi_cl": np.exp(m1.conf_int()[:, 1]),
                        "OR_svy": sv.OR, "lo_svy": sv.lo, "hi_svy": sv.hi, "p_svy": sv.p})
    res.to_csv("lr29_unw_cluster_svy.csv", index=False)
    print("\n=== 29-feature model: unweighted vs cluster-robust vs survey-weighted ===")
    print(res.round(2).to_string(index=False))
    print("Saved lr29_unw_cluster_svy.csv")

    # --- 5b. All 14 regions: sum-to-zero (average-region) coding and Oromia reference ---
    nonreg = [c for c in selected_features if not c.startswith("region_")]
    codes = list(REGION_NAMES)
    R = np.stack([(ir.v024.values == c).astype(float) for c in codes], 1)

    def fit_regions(weighted, ref):
        """ref='effect': sum-to-zero coding (each region vs the average region);
        ref=<region code>: contrasts against that region."""
        if ref == "effect":
            Rm = R[:, :-1] - R[:, [-1]]; rn = codes[:-1]
        else:
            keep = [i for i, c in enumerate(codes) if c != ref]; Rm = R[:, keep]; rn = [codes[i] for i in keep]
        Xm = np.column_stack([np.ones(len(ya))] + [Xa[c].values for c in nonreg] + [Rm])
        nm = ["const"] + nonreg + ["reg:" + REGION_NAMES[c] for c in rn]
        if weighted:
            out, V = svyglm_logit(Xm, ya, d, nm); b = out.coef.values; qq = q
        else:
            m = sm.Logit(ya, Xm).fit(disp=0); out = None; V = m.cov_params(); b = m.params; qq = 1.96
        return b, V, nm, qq, out

    def region_table(weighted, ref):
        b, V, nm, qq, _ = fit_regions(weighted, ref)
        ri = [i for i, n in enumerate(nm) if n.startswith("reg:")]
        L = np.zeros((len(codes), len(nm)))
        if ref == "effect":
            for j, i in enumerate(ri): L[j, i] = 1
            L[-1, ri] = -1
        else:
            j = 0
            for k, c in enumerate(codes):
                if c == ref: continue
                L[k, ri[j]] = 1; j += 1
        rows = []
        for k, c in enumerate(codes):
            if not L[k].any():
                rows.append([REGION_NAMES[c], 1.0, np.nan, np.nan]); continue
            e = L[k] @ b; se = np.sqrt(L[k] @ V @ L[k])
            rows.append([REGION_NAMES[c], np.exp(e), np.exp(e - qq * se), np.exp(e + qq * se)])
        return pd.DataFrame(rows, columns=["region", "OR", "lo", "hi"]).round(2)

    a = region_table(True, "effect").rename(columns={"OR": "OR_vs_average", "lo": "lo_avg", "hi": "hi_avg"})
    o = region_table(True, 4).rename(columns={"OR": "OR_vs_Oromia", "lo": "lo_or", "hi": "hi_or"})
    u = region_table(False, "effect").rename(columns={"OR": "OR_vs_average_unwtd", "lo": "lo_u", "hi": "hi_u"})
    reg_all = a.merge(o, on="region").merge(u, on="region")
    reg_all.to_csv("region_all14_weighted.csv", index=False)
    print("\n=== All 14 regions (survey-weighted; average-region and Oromia-reference contrasts) ===")
    print(reg_all.to_string(index=False))
    print("Saved region_all14_weighted.csv")

    # --- 5c. Wealth sensitivity: do water/toilet/education/literacy/region absorb the wealth gradient? ---
    def wealth_or(drop):
        cols = [c for c in selected_features if not any(c.startswith(p) for p in drop)]
        Xm = np.column_stack([np.ones(len(ya))] + [Xa[c].values for c in cols])
        out, _ = svyglm_logit(Xm, ya, d, ["const"] + cols)
        return out[out.term.str.startswith("wealth")]

    SPECS = [("Full 29-feature model", []),
             ("Drop water + toilet", ["water", "toilet"]),
             ("Drop education, literacy, region", ["education", "literacy", "region"]),
             ("Wealth with household/ social covariates only",
              ["education", "literacy", "region", "media", "insurance", "hh_size", "water",
               "toilet", "internet", "religion", "currently"])]
    rows = []
    for spec, drop in SPECS:
        for r in wealth_or(drop).itertuples():
            rows.append(dict(model=spec, term=r.term, OR=r.OR, lo=r.lo, hi=r.hi, p=r.p))
    wsens = pd.DataFrame(rows)
    wsens.round(3).to_csv("wealth_sensitivity_weighted.csv", index=False)
    print("\n=== Wealth ORs (reference: poorest), survey-weighted ===")
    print(wsens.round(2).to_string(index=False))
    print("Saved wealth_sensitivity_weighted.csv")

    # --- 5d. Final design-based model (all 14 regions, sum-to-zero) + forest plot ---
    _, _, nm, _, out = fit_regions(True, "effect")
    b, V, _, _, _ = fit_regions(True, "effect")
    ri = [i for i, n in enumerate(nm) if n.startswith("reg:")]
    rows = []
    for k, c in enumerate(codes):
        Lk = np.zeros(len(nm))
        if k < len(codes) - 1: Lk[ri[k]] = 1
        else: Lk[ri] = -1
        e = Lk @ b; se = np.sqrt(Lk @ V @ Lk)
        rows.append(dict(term="Region: " + REGION_NAMES[c], OR=np.exp(e), lo=np.exp(e - q * se), hi=np.exp(e + q * se)))
    REG = pd.DataFrame(rows)

    lab = {"literacy_3.0": "Literacy: whole sentence", "literacy_2.0": "Literacy: part of sentence",
           "literacy_4.0": "Literacy: other/unknown", "education_1.0": "Education: primary",
           "education_2.0": "Education: secondary", "media_cat_1.0": "Media exposure: low",
           "media_cat_2.0": "Media exposure: moderate", "media_cat_3.0": "Media exposure: high",
           "wealth_2.0": "Wealth: poorer", "wealth_3.0": "Wealth: middle", "wealth_4.0": "Wealth: richer",
           "wealth_5.0": "Wealth: richest", "insurance_1.0": "Health insurance: yes",
           "hh_size": "Household size (per member)", "water_source_3.0": "Water: unimproved",
           "toilet_type_2.0": "Toilet: improved pit latrine", "toilet_type_4.0": "Toilet: none/other",
           "religion_3.0": "Religion: Muslim", "internet_use_1.0": "Internet use: last 12 months",
           "currently_working_1.0": "Currently working"}
    NR = out[out.term.isin(nonreg)].copy()
    unlabelled = [t for t in NR.term if t not in lab]
    assert not unlabelled, f"No label defined for: {unlabelled} -- add them to lab."
    NR["term"] = NR.term.map(lab)
    FM = pd.concat([NR[["term", "OR", "lo", "hi", "p"]], REG], ignore_index=True)
    FM.round(3).to_csv("final_svy_model_or.csv", index=False)
    print("\n=== Final survey-weighted model (OR, 95% design-based CI) ===")
    print(FM.round(2).to_string(index=False))
    print("Saved final_svy_model_or.csv")

    order = ["Education: primary", "Education: secondary", "Literacy: part of sentence",
             "Literacy: whole sentence", "Literacy: other/unknown", "Media exposure: low",
             "Media exposure: moderate", "Media exposure: high", "Health insurance: yes",
             "Wealth: poorer", "Wealth: middle", "Wealth: richer", "Wealth: richest",
             "Household size (per member)", "Water: unimproved", "Toilet: improved pit latrine",
             "Toilet: none/other", "Religion: Muslim", "Internet use: last 12 months", "Currently working"]
    order = [t for t in order if t in set(FM.term)]
    A = FM.set_index("term")
    rows_f = [(t, A.loc[t]) for t in order] + [(t, A.loc[t]) for t in REG.sort_values("OR").term]
    Vc = matplotlib.colormaps["viridis"]
    fig, ax = plt.subplots(figsize=(6.2, 8.4))
    for i, (t, r) in enumerate(rows_f[::-1]):
        col = Vc(0.15) if not t.startswith("Region") else Vc(0.62)
        ax.plot([r.lo, r.hi], [i, i], color=col, lw=1.6); ax.plot(r.OR, i, "o", color=col, ms=4)
    ax.set_yticks(range(len(rows_f))); ax.set_yticklabels([t for t, _ in rows_f[::-1]], fontsize=7.5)
    ax.axvline(1, color="k", lw=.7, ls="--"); ax.set_xscale("log")
    ax.set_xticks([.1, .25, .5, 1, 2, 4, 8, 16]); ax.set_xticklabels(["0.1", "0.25", "0.5", "1", "2", "4", "8", "16"])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Adjusted odds ratio (95% design-based CI, log scale)")
    ax.axhline(len(REG) - .5, color="grey", lw=.5)
    plt.tight_layout()
    plt.savefig("svy_forest_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved svy_forest_plot.png")

    print("\n=== Odds ratio analysis (development split + survey-weighted) complete ===")



def run_hyperparameter_tuning():
    """
    HYPERPARAMETER TUNING
    """
    print("\n" + "="*70)
    print("  HYPERPARAMETER TUNING")
    print("="*70)

    warnings.filterwarnings("ignore")
    matplotlib.rcParams['font.size'] = 9

    DATA_DIR = get_data_dir()
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

    DATA_DIR = get_data_dir()
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




def run_confusion_matrices_tuned():
    """
    CONFUSION MATRICES (TUNED MODELS)
    """
    print("\n" + "="*70)
    print("  CONFUSION MATRICES (TUNED MODELS)")
    print("="*70)

    warnings.filterwarnings("ignore")
    matplotlib.rcParams['font.size'] = 9

    DATA_DIR = get_data_dir()
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


class CatBoostNative(ClassifierMixin, BaseEstimator):
    """scikit-learn wrapper so CatBoost can use native categorical features inside RandomizedSearchCV."""
    def __init__(self, iterations=300, learning_rate=0.05, depth=4, l2_leaf_reg=3, seed=0):
        self.iterations = iterations; self.learning_rate = learning_rate
        self.depth = depth; self.l2_leaf_reg = l2_leaf_reg; self.seed = seed

    def fit(self, X, y):
        cats = [c for c in X.columns if X[c].dtype == object]
        self.m_ = CatBoostClassifier(iterations=self.iterations, learning_rate=self.learning_rate,
                                     depth=self.depth, l2_leaf_reg=self.l2_leaf_reg, random_seed=self.seed,
                                     verbose=0, thread_count=1, cat_features=cats).fit(X, y)
        self.classes_ = self.m_.classes_
        return self

    def predict_proba(self, X): return self.m_.predict_proba(X)
    def predict(self, X): return self.m_.predict(X)


def run_nested_cv():
    """
    NESTED REPEATED CROSS-VALIDATION OF THE COMPLETE PIPELINE
    """
    print("\n" + "="*70)
    print("  NESTED REPEATED CROSS-VALIDATION (5 x 5)")
    print("="*70)

    warnings.filterwarnings("ignore")
    DATA_DIR = get_data_dir()
    os.chdir(DATA_DIR)

    # ======================================================================
    # 0. SETUP
    # ======================================================================
    # Every outer training fold repeats the WHOLE pipeline - LASSO feature
    # selection (1-SE rule) and RandomizedSearchCV tuning (3-fold inner CV) -
    # so no information from the outer test fold reaches model building.
    # The same 25 outer splits are used for all eight models, which makes
    # the paired AUC comparisons in run_nested_cv_summary() valid.
    ir, df, X = load_analysis_data()
    y = ir.hpv03.values.astype(int)
    cat_cols = [c for c in ALL_CAT_VARS]
    Xc = df[cat_cols + CONT_VARS].copy()
    for c in cat_cols:
        Xc[c] = Xc[c].astype(int).astype(str)          # native categoricals for CatBoost
    bench_cols = [c for c in X.columns if c.startswith(("education_", "literacy_", "region_"))]
    cv_inner = StratifiedKFold(3, shuffle=True, random_state=1)
    print(f"Repeats: {N_REPEATS}   random-search draws per model and fold: {N_SEARCH_DRAWS}")


    def tune(est, space, Xtr, ytr, seed):
        s = RandomizedSearchCV(est, space, n_iter=N_SEARCH_DRAWS, scoring="roc_auc", cv=cv_inner,
                               random_state=seed, n_jobs=2, refit=True)
        s.fit(Xtr, ytr)
        return s.best_estimator_

    def lasso_select(Xtr, ytr, seed):
        """LASSO with the 1-SE rule on the outer training fold only."""
        Z = StandardScaler().fit_transform(Xtr)
        Cs = np.logspace(-3, 0, 15); aucs = []
        for C in Cs:
            a = []
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(Z, ytr):
                m = LogisticRegression(penalty="l1", solver="liblinear", C=C).fit(Z[tr], ytr[tr])
                a.append(roc_auc_score(ytr[va], m.decision_function(Z[va])))
            aucs.append(a)
        aucs = np.array(aucs); mean = aucs.mean(1); se = aucs.std(1, ddof=1) / np.sqrt(aucs.shape[1])
        b = mean.argmax()
        C1 = Cs[np.where(mean >= mean[b] - se[b])[0].min()]        # most regularised C within 1 SE of the best
        m = LogisticRegression(penalty="l1", solver="liblinear", C=C1).fit(Z, ytr)
        return np.where(m.coef_[0] != 0)[0]

    def calib(yt, p):
        """Calibration slope and calibration-in-the-large (intercept with the slope fixed at 1)."""
        p = np.clip(p, 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p))
        slope = sm.Logit(yt, sm.add_constant(lp)).fit(disp=0).params[1]
        icpt = sm.GLM(yt, np.ones_like(lp), offset=lp, family=sm.families.Binomial()).fit().params[0]
        return slope, icpt

    lr_space = {"logisticregression__C": loguniform(1e-3, 1e2)}


    # ======================================================================
    # 1. OUTER LOOP: 5-FOLD CV REPEATED N_REPEATS TIMES
    # ======================================================================
    rows, preds = [], []
    sel_freq = np.zeros(X.shape[1]); nsplit = 0
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=2026)
    t0 = time.time()
    for k, (tr, te) in enumerate(rskf.split(X, y)):
        rep, fold = divmod(k, 5)
        Xtr, Xte = X.values[tr], X.values[te]; ytr, yte = y[tr], y[te]; out = {}

        # 1. Benchmark: education + literacy + region only
        bi = [list(X.columns).index(c) for c in bench_cols]
        m = tune(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), lr_space, Xtr[:, bi], ytr, k)
        out["Benchmark LR (educ, literacy, region)"] = m.predict_proba(Xte[:, bi])[:, 1]

        # 2. LASSO (selected inside the fold) -> logistic regression
        si = lasso_select(Xtr, ytr, k); sel_freq[si] += 1; nsplit += 1
        m = tune(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), lr_space, Xtr[:, si], ytr, k)
        out["LASSO-selected LR"] = m.predict_proba(Xte[:, si])[:, 1]

        # 3. Penalised logistic regression on all features
        m = tune(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), lr_space, Xtr, ytr, k)
        out["LR (all features)"] = m.predict_proba(Xte)[:, 1]

        # 4. Random forest
        m = tune(RandomForestClassifier(n_estimators=300, random_state=k, n_jobs=1),
                 {"max_depth": randint(3, 16), "min_samples_leaf": randint(1, 20), "max_features": uniform(0.1, 0.5)},
                 Xtr, ytr, k)
        out["Random forest"] = m.predict_proba(Xte)[:, 1]

        # 5. XGBoost
        m = tune(XGBClassifier(n_jobs=1, random_state=k, verbosity=0, tree_method="hist"),
                 {"n_estimators": randint(100, 600), "learning_rate": loguniform(0.01, 0.2),
                  "max_depth": randint(2, 7), "subsample": uniform(0.6, 0.4),
                  "colsample_bytree": uniform(0.5, 0.5), "min_child_weight": randint(1, 10),
                  "reg_lambda": loguniform(0.1, 20)}, Xtr, ytr, k)
        out["XGBoost"] = m.predict_proba(Xte)[:, 1]

        # 6. LightGBM
        m = tune(LGBMClassifier(n_jobs=1, random_state=k, verbose=-1),
                 {"n_estimators": randint(100, 600), "learning_rate": loguniform(0.01, 0.2),
                  "num_leaves": randint(4, 40), "min_child_samples": randint(5, 50),
                  "subsample": uniform(0.6, 0.4), "subsample_freq": [1],
                  "colsample_bytree": uniform(0.5, 0.5), "reg_lambda": loguniform(0.1, 20)}, Xtr, ytr, k)
        out["LightGBM"] = m.predict_proba(Xte)[:, 1]

        # 7. CatBoost with native categorical features
        m = tune(CatBoostNative(seed=k),
                 {"iterations": randint(150, 500), "learning_rate": loguniform(0.02, 0.2),
                  "depth": randint(3, 7), "l2_leaf_reg": loguniform(1, 20)}, Xc.iloc[tr], ytr, k)
        out["CatBoost (native categoricals)"] = m.predict_proba(Xc.iloc[te])[:, 1]

        # 8. Neural network (MLP)
        m = tune(make_pipeline(StandardScaler(), MLPClassifier(max_iter=400, early_stopping=True, random_state=k)),
                 {"mlpclassifier__hidden_layer_sizes": [(16,), (32,), (64,), (32, 16)],
                  "mlpclassifier__alpha": loguniform(1e-4, 1e0),
                  "mlpclassifier__learning_rate_init": loguniform(1e-3, 3e-2)}, Xtr, ytr, k)
        out["MLP"] = m.predict_proba(Xte)[:, 1]

        for name, p in out.items():
            sl, ic = calib(yte, p)
            rows.append(dict(split=k, repeat=rep, fold=fold, model=name, auc=roc_auc_score(yte, p),
                             brier=brier_score_loss(yte, p), logloss=log_loss(yte, p),
                             cal_slope=sl, cal_int=ic))
            preds.append(pd.DataFrame({"split": k, "model": name, "idx": te, "y": yte, "p": p}))
        print(f"split {k+1}/{5*N_REPEATS} done  {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv("nested_cv_metrics.csv", index=False)

    pd.concat(preds).to_csv("nested_cv_oof_preds.csv", index=False)
    pd.DataFrame({"feature": X.columns, "selection_freq": sel_freq / nsplit}) \
        .sort_values("selection_freq", ascending=False).to_csv("lasso_selection_stability.csv", index=False)
    print("Saved nested_cv_metrics.csv, nested_cv_oof_preds.csv, lasso_selection_stability.csv")
    print(f"\n=== Nested cross-validation complete ({time.time()-t0:.0f} s) ===")


def run_nested_cv_summary():
    """
    NESTED CV SUMMARY: AUC WITH CORRECTED CIs, PAIRED TESTS, LASSO STABILITY
    """
    print("\n" + "="*70)
    print("  NESTED CV SUMMARY")
    print("="*70)

    warnings.filterwarnings("ignore")
    matplotlib.rcParams['font.size'] = 9
    DATA_DIR = get_data_dir()
    os.chdir(DATA_DIR)
    V = matplotlib.colormaps["viridis"]

    m = pd.read_csv("nested_cv_metrics.csv")
    n_splits = m.split.nunique(); N = 2842
    n_te = N / 5; n_tr = N - n_te
    rho = n_te / n_tr          # Nadeau-Bengio correction: repeated CV folds overlap, so the naive SE is too small

    def corr_t(d):
        """Mean, 95% CI and p-value with the Nadeau-Bengio corrected variance (1/k + n_test/n_train)."""
        mu = d.mean(); var = d.var(ddof=1); se = np.sqrt((1 / len(d) + rho) * var)
        t = mu / se; p = 2 * stats.t.sf(abs(t), len(d) - 1); qq = stats.t.ppf(.975, len(d) - 1)
        return mu, mu - qq * se, mu + qq * se, p

    # ======================================================================
    # 1. SUMMARY TABLE
    # ======================================================================
    rows = []
    for name, g in m.groupby("model"):
        a = g.auc.values; mu, lo, hi, _ = corr_t(a - 0.5)
        rows.append(dict(model=name, AUC=a.mean(), AUC_sd=a.std(ddof=1), AUC_lo=lo + .5, AUC_hi=hi + .5,
                         Brier=g.brier.mean(), cal_slope=g.cal_slope.mean(), cal_int=g.cal_int.mean()))
    S = pd.DataFrame(rows).sort_values("AUC", ascending=False).round(3)
    S.to_csv("nested_cv_summary.csv", index=False)
    print(S.to_string(index=False))
    print("Saved nested_cv_summary.csv")

    # ======================================================================
    # 2. PAIRED AUC DIFFERENCES (same outer splits for every model)
    # ======================================================================
    P = m.pivot(index="split", columns="model", values="auc")
    print(f"\nPaired AUC differences (model minus reference), Nadeau-Bengio corrected, {n_splits} splits")
    prow = []
    for ref in ["LASSO-selected LR", "Benchmark LR (educ, literacy, region)"]:
        print("\nvs", ref)
        for c in P.columns:
            if c == ref: continue
            mu, lo, hi, p = corr_t((P[c] - P[ref]).values)
            prow.append(dict(reference=ref, model=c, diff=mu, lo=lo, hi=hi, p=p))
            print(f"  {c:38s} diff {mu:+.3f} ({lo:+.3f},{hi:+.3f}) p={p:.3f}")
    pd.DataFrame(prow).round(4).to_csv("nested_cv_paired_tests.csv", index=False)
    best = P.mean().idxmax()
    print(f"\nBest mean AUC: {best}.  Number of splits in which each model is top:")
    print(P.idxmax(axis=1).value_counts().to_string())

    # ======================================================================
    # 3. LASSO SELECTION STABILITY
    # ======================================================================
    print("\n=== LASSO selection frequency across outer training folds ===")
    print(pd.read_csv("lasso_selection_stability.csv").head(32).round(2).to_string(index=False))

    # ======================================================================
    # 4. FIGURE: OUT-OF-FOLD AUC PER MODEL
    # ======================================================================
    order_m = S.sort_values("AUC").model.tolist()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, name in enumerate(order_m):
        a = m[m.model == name].auc.values; s = S[S.model == name].iloc[0]
        ax.scatter(a, np.full(len(a), i) + np.random.RandomState(i).uniform(-.18, .18, len(a)),
                   s=8, color=V(.8), alpha=.5)
        ax.plot([s.AUC_lo, s.AUC_hi], [i, i], color=V(.1), lw=2.2); ax.plot(s.AUC, i, "o", color=V(.1), ms=5)
    ax.set_yticks(range(len(order_m))); ax.set_yticklabels(order_m, fontsize=8)
    ax.set_xlabel(f"Out-of-fold AUC ({n_splits} outer folds; bar = 95% CI, resampling-corrected)")
    plt.tight_layout()
    plt.savefig("nested_cv_auc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved nested_cv_auc.png, nested_cv_paired_tests.csv")

    print("\n=== Nested CV summary complete ===")


def run_calibration_analysis():
    """
    CALIBRATION (nested out-of-fold predictions): SLOPE, INTERCEPT, DECILES
    """
    print("\n" + "="*70)
    print("  CALIBRATION ANALYSIS")
    print("="*70)

    warnings.filterwarnings("ignore")
    matplotlib.rcParams['font.size'] = 9
    DATA_DIR = get_data_dir()
    os.chdir(DATA_DIR)
    V = matplotlib.colormaps["viridis"]

    ir, df, X = load_analysis_data()
    y = ir.hpv03.values.astype(int)
    pr = pd.read_csv("nested_cv_oof_preds.csv"); pr["rep"] = pr.split // 5

    # ======================================================================
    # 1. CALIBRATION SLOPE AND INTERCEPT, BOOTSTRAP 95% CI
    # ======================================================================
    # Predictions are pooled across the five outer test folds of repeat 0, so
    # every girl has exactly one out-of-fold prediction. A slope of 1 and an
    # intercept of 0 indicate perfect calibration; slope < 1 = over-confident.
    def slope_int(yy, p):
        p = np.clip(p, 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p))
        s = sm.Logit(yy, sm.add_constant(lp)).fit(disp=0).params[1]
        i = sm.GLM(yy, np.ones_like(lp), offset=lp, family=sm.families.Binomial()).fit().params[0]
        return s, i

    rng = np.random.RandomState(7); rows = []
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.7), sharex=True, sharey=True)
    for ax, (mod, label) in zip(axs, [("LASSO-selected LR", "Logistic regression (LASSO-selected)"),
                                      ("XGBoost", "XGBoost")]):
        g = pr[(pr.model == mod) & (pr.rep == 0)].sort_values("idx"); p = g.p.values
        assert len(g) == len(y) and (g.idx.values == np.arange(len(y))).all()
        s0, i0 = slope_int(y, p); bs = []
        for _ in range(300):
            ix = rng.randint(0, len(y), len(y)); bs.append(slope_int(y[ix], p[ix]))
        bs = np.array(bs)
        rows.append(dict(model=label, slope=s0, slope_lo=np.percentile(bs[:, 0], 2.5), slope_hi=np.percentile(bs[:, 0], 97.5),
                         intercept=i0, int_lo=np.percentile(bs[:, 1], 2.5), int_hi=np.percentile(bs[:, 1], 97.5)))

        # ==================================================================
        # 2. DECILE CALIBRATION PLOT (Wilson 95% intervals)
        # ==================================================================
        bins = pd.qcut(p, 10, labels=False); xs, ys, lo, hi = [], [], [], []
        for k in range(10):
            mm = bins == k; n = mm.sum(); ph = y[mm].mean(); z = 1.96
            c = (ph + z*z / (2*n)) / (1 + z*z / n); h = z * np.sqrt(ph*(1-ph)/n + z*z / (4*n*n)) / (1 + z*z / n)
            xs.append(p[mm].mean()); ys.append(ph); lo.append(c - h); hi.append(c + h)
        ax.plot([0, 1], [0, 1], color="grey", lw=.8, ls="--")
        ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)],
                    fmt="o", color=V(.2), ms=4, capsize=2, lw=1)
        ax.set_xlim(.1, .9); ax.set_ylim(.1, .9); ax.set_title(label, fontsize=8.5)
        ax.set_xlabel("Mean predicted probability (deciles)")
    axs[0].set_ylabel("Observed proportion vaccinated")
    plt.tight_layout()
    plt.savefig("calibration_oof.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    C = pd.DataFrame(rows).round(3); C.to_csv("calibration_pooled.csv", index=False)
    print(C.to_string(index=False))
    print("\nSaved calibration_pooled.csv, calibration_oof.png")

    print("\n=== Calibration analysis complete ===")


def run_decision_curve_analysis():
    """
    DECISION CURVE ANALYSIS (event = unvaccinated; sample and survey-weighted)
    """
    print("\n" + "="*70)
    print("  DECISION CURVE ANALYSIS")
    print("="*70)

    warnings.filterwarnings("ignore")
    matplotlib.rcParams['font.size'] = 9
    DATA_DIR = get_data_dir()
    os.chdir(DATA_DIR)
    V = matplotlib.colormaps["viridis"]

    ir, df, X = load_analysis_data()
    y = ir.hpv03.values.astype(int); wt = ir.wt.values
    pr = pd.read_csv("nested_cv_oof_preds.csv"); pr["rep"] = pr.split // 5

    # ======================================================================
    # 1. NET BENEFIT ACROSS THRESHOLDS
    # ======================================================================
    # The intervention is outreach to girls predicted to be UNVACCINATED, so
    # the event is "unvaccinated" (1 - y) and the risk is 1 - P(vaccinated).
    # Net benefit = TP/N - FP/N * t/(1-t); the survey-weighted version
    # replaces counts with weighted counts. Net benefit is averaged over the
    # repeats of the nested cross-validation.
    ths = np.round(np.arange(.10, .81, .05), 2)

    def net_benefit(ev, risk, w, t):
        treat = risk >= t
        tp = (w * (treat & (ev == 1))).sum() / w.sum(); fp = (w * (treat & (ev == 0))).sum() / w.sum()
        return tp - fp * t / (1 - t)

    keep = {"LASSO-selected LR": "Logistic regression (LASSO-selected)", "XGBoost": "XGBoost",
            "CatBoost (native categoricals)": "CatBoost",
            "Benchmark LR (educ, literacy, region)": "Benchmark (education, literacy, region)"}
    ev = 1 - y; rows = []
    for wtd in (False, True):
        w = wt if wtd else np.ones(len(y))
        for t in ths:
            r = dict(weighted=wtd, threshold=t,
                     treat_all=(w * ev).sum() / w.sum() - (w * (1 - ev)).sum() / w.sum() * t / (1 - t),
                     treat_none=0.0)
            for mod in keep:
                v = []
                for rep, g in pr[pr.model == mod].groupby("rep"):
                    g = g.sort_values("idx"); assert len(g) == len(y) and (g.idx.values == np.arange(len(y))).all()
                    v.append(net_benefit(ev, 1 - g.p.values, w, t))
                r[mod] = np.mean(v)
            rows.append(r)
    D = pd.DataFrame(rows); D.round(4).to_csv("dca_unvaccinated.csv", index=False)
    print("Saved dca_unvaccinated.csv")

    # ======================================================================
    # 2. DCA PLOT
    # ======================================================================
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    for ax, wtd, title in zip(axs, (False, True), ("Sample (unweighted)", "Survey-weighted")):
        g = D[D.weighted == wtd]
        ax.plot(g.threshold, g.treat_all, color="grey", lw=1.2, label="Contact all")
        ax.axhline(0, color="k", lw=.7, ls=":", label="Contact none")
        for i, (mod, label) in enumerate(keep.items()):
            ax.plot(g.threshold, g[mod], lw=1.5, color=V([.1, .45, .72, .92][i]),
                    ls="-" if i < 3 else "--", label=label)
        ax.set_xlabel("Threshold probability of being unvaccinated"); ax.set_title(title, fontsize=9)
        ax.set_ylim(-.05, .5)
    axs[0].set_ylabel("Net benefit"); axs[1].legend(frameon=False, fontsize=7)
    plt.tight_layout()
    plt.savefig("dca_unvaccinated.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved dca_unvaccinated.png")

    # ======================================================================
    # 3. NET BENEFIT AT SELECTED THRESHOLDS
    # ======================================================================
    print(D[D.threshold.isin([.2, .3, .4, .5, .6, .7])].round(3).to_string(index=False))
    print(f"\nUnvaccinated prevalence: unweighted {ev.mean():.3f}, weighted {(wt*ev).sum()/wt.sum():.3f}")

    print("\n=== Decision curve analysis complete ===")


# ============================================================================
# MAIN: run the full pipeline in order
# ============================================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    run_build_design()
    run_eda()
    run_survey_prevalence()
    run_preprocessing_lasso()
    run_table1_characteristics()
    run_model_comparison()
    run_lr_odds_ratios()
    run_hyperparameter_tuning()
    run_shap_analysis()
    if RUN_NESTED_CV:
        run_nested_cv()
    run_nested_cv_summary()
    run_calibration_analysis()
    run_decision_curve_analysis()
    run_confusion_matrices_tuned()

    print("\n" + "="*70)
    print("  FULL PIPELINE COMPLETE")
    print("="*70)
