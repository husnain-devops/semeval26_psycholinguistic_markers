#!/usr/bin/env python3
"""Merge features, run tests and simple models, compute SHAP for RandomForest if available."""
import os
from pathlib import Path
import logging

SEED = 42

BASE = Path(__file__).resolve().parents[1]
FEAT = BASE / "features"
FIG = BASE / "figures"
MODELS = BASE / "models"
PROC = BASE / "data_processed"
LOG = BASE / "reports"

FEAT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
# Create subdirectories for different plot types
FIG_MEAN = FIG / 'mean_comparison'
FIG_MEAN.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
run_log = LOG / "run_log.txt"

def append_log(msg):
    with open(run_log, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    logging.info(msg)

def main():
    append_log('[stat_tests] START')
    # load basic features
    try:
        df = pd.read_parquet(PROC / 'data_processed.parquet')
    except Exception:
        try:
            df = pd.read_csv(PROC / 'data_clean.csv')
        except Exception:
            append_log('[stat_tests] no base data found')
            return
    # try to merge lexical and discourse and sentiment
    parts = [FEAT / 'lexical_complexity.csv', FEAT / 'discourse_markers.csv', FEAT / 'sentiment_emotion.csv', FEAT / 'token_pos_counts.csv']
    dfs = [df]
    for p in parts:
        if p.exists():
            try:
                dfs.append(pd.read_csv(p))
            except Exception as e:
                append_log(f"could not read {p}: {e}")
    # naive merge on _id if present
    merged = dfs[0]
    for sub in dfs[1:]:
        if '_id' in merged.columns and '_id' in sub.columns:
            merged = merged.merge(sub, on='_id', how='left')
    # choose numeric features
    numcols = merged.select_dtypes(include=[float,int]).columns.tolist()
    append_log(f"[stat_tests] numeric cols: {numcols[:20]}")
    # t-test yes vs no for first few features
    try:
        from scipy.stats import ttest_ind, mannwhitneyu
        yes = merged[merged['conspiracy']=='yes']
        no = merged[merged['conspiracy']=='no']
        tests = []
        for c in numcols:
            try:
                a = yes[c].dropna()
                b = no[c].dropna()
                if len(a)>2 and len(b)>2:
                    tstat, p = ttest_ind(a,b, nan_policy='omit')
                    tests.append({'feature':c,'tstat':float(tstat),'p':float(p)})
            except Exception:
                continue
        pd.DataFrame(tests).to_csv(FEAT / 'univariate_tests.csv', index=False)
        append_log(f"[stat_tests] saved univariate_tests.csv")
        
        # Plot univariate test results
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            tests_df = pd.DataFrame(tests)
            if len(tests_df) > 0:
                # Sort by absolute t-statistic
                tests_df = tests_df.sort_values('tstat', key=abs, ascending=False).head(20)
                
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                
                # Plot t-statistics
                axes[0].barh(tests_df['feature'], tests_df['tstat'])
                axes[0].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                axes[0].set_xlabel('T-statistic')
                axes[0].set_title('Top 20 Features: T-statistics (yes vs no)')
                axes[0].grid(axis='x', alpha=0.3)
                
                # Plot p-values (log scale)
                axes[1].barh(tests_df['feature'], tests_df['p'])
                axes[1].axvline(x=0.05, color='red', linestyle='--', linewidth=0.8, label='p=0.05')
                axes[1].set_xlabel('P-value')
                axes[1].set_title('Top 20 Features: P-values (yes vs no)')
                axes[1].set_xscale('log')
                axes[1].legend()
                axes[1].grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(FIG_MEAN / 'univariate_tests.png', dpi=150, bbox_inches='tight')
                plt.close()
                append_log(f"[stat_tests] saved univariate_tests.png")
        except Exception as e:
            append_log(f"[stat_tests] plotting failed: {e}")
    except Exception as e:
        append_log(f"stat tests skipped: {e}")

    # quick models
    try:
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        X = merged[numcols].fillna(0)
        y = merged['conspiracy'].apply(lambda s: 1 if s=='yes' else 0 if s=='no' else 2)
        # binary yes vs no only
        mask = y.isin([0,1])
        Xb = X[mask]
        yb = y[mask]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, random_state=SEED))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        scores = cross_val_score(clf, Xb, yb, cv=cv, scoring='f1', n_jobs=1)
        append_log(f"LogReg F1 (5-fold): mean={scores.mean():.3f}")
        # RandomForest
        rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
        rf.fit(Xb.fillna(0), yb)
        import joblib
        joblib.dump(rf, MODELS / 'rf_model.joblib')
        append_log('RandomForest trained and saved')
        # SHAP
        try:
            import shap
            explainer = shap.TreeExplainer(rf)
            shap_vals = explainer.shap_values(Xb.fillna(0))
            # get mean abs shap for binary class 1
            import numpy as np
            mean_abs = np.abs(shap_vals[1]).mean(axis=0)
            idx = np.argsort(mean_abs)[::-1][:20]
            feat_names = [numcols[i] for i in idx]
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,6))
            shap.summary_plot(shap_vals, Xb.fillna(0), show=False)
            plt.tight_layout()
            plt.savefig(FIG_MEAN / 'shap_summary.png')
            pd.DataFrame({'feature':feat_names,'importance':mean_abs[idx]}).to_csv(FIG.parent / 'shap_top20.csv', index=False)
            append_log('SHAP summary saved')
        except Exception as e:
            append_log(f"SHAP skipped or failed: {e}")
    except Exception as e:
        append_log(f"modeling skipped: {e}")

    append_log('[stat_tests] END')

if __name__ == '__main__':
    main()
