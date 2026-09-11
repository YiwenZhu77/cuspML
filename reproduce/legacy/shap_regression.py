"""Recompute regression SHAP for the equatorward-MLAT model (paper cross-check).

WHAT:   Trains the exact paper XGBoost model (74 feats, random split seed 42) on the
        archived catalog, then computes TreeExplainer SHAP on 2,000 test-set samples
        and prints the top mean|SHAP| features. Replaces the stale r029 (occurrence
        classifier) SHAP with the regression model's own attributions.
RESULT: mean|SHAP| in MLAT degrees for the top features -> the numbers cited at
        main.tex:226/305/341.
RUN:    CUSPML_DATA=/glade/work/yizhu/cuspML/reproduce/data \
        /glade/u/home/yizhu/work/miniconda3/envs/py3.10/bin/python shap_regression.py
DEPS:   reuses reproduce.load_frame + XGB config; shap 0.49, xgboost, pandas, numpy.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reproduce import load_frame, XGB
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import shap

dfc, feats, targets = load_frame()
X = dfc[feats].to_numpy(float)
y = dfc["abs_eq_mlat"].to_numpy(float)
N = len(dfc)
itr, ite = train_test_split(np.arange(N), test_size=0.2, random_state=42)
m = XGBRegressor(**XGB).fit(X[itr], y[itr])

# SHAP on 2,000 test samples (deterministic subsample, seed 42)
rng = np.random.RandomState(42)
sub = rng.choice(ite, size=min(2000, len(ite)), replace=False)
expl = shap.TreeExplainer(m)
sv = expl.shap_values(X[sub])
mean_abs = np.abs(sv).mean(axis=0)
order = np.argsort(mean_abs)[::-1]

print(f"model: {N} crossings, {len(feats)} features, test-MAE = "
      f"{np.abs(y[ite]-m.predict(X[ite])).mean():.4f} deg; SHAP on n={len(sub)}")
print("rank feature                         mean|SHAP| (deg)")
top = []
for r, i in enumerate(order[:12], 1):
    print(f"{r:>3}  {feats[i]:<32} {mean_abs[i]:.4f}")
    top.append({"feature": feats[i], "mean_abs_shap": round(float(mean_abs[i]), 4)})

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "src", "kernels", "cuspmap_mvp", "bundles", "r029b_shap_regression.json")
out = os.path.abspath(out)
json.dump({"model": "equatorward_mlat_xgb", "split": "random_seed42", "n_shap": int(len(sub)),
           "test_mae_deg": round(float(np.abs(y[ite]-m.predict(X[ite])).mean()), 4),
           "top_features": top}, open(out, "w"), indent=1)
print(f"saved {out}")
