"""Per-drug response prediction: elastic net vs XGBoost, K-fold CV.

For each drug with enough screened lines:
  * outer K-fold CV -> out-of-fold predictions -> Pearson r, Spearman rho, RMSE
  * elastic net tunes alpha/l1_ratio with an inner CV on each training fold
  * features are standardized inside each fold (no leakage)
Outputs results/models/cv_metrics.csv and oof_predictions.parquet.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _enet(seed):
    # eps=1e-2 ends the alpha path at alpha_max/100: the weaker penalties are the
    # slowest to fit and were never selected in profiling. With tol=1e-3 and 15
    # alphas this is ~5-12x faster with held-out r unchanged (|delta r| <= 0.005).
    kw = dict(l1_ratio=[0.1, 0.5, 0.9], eps=1e-2, tol=1e-3, cv=3, max_iter=3000,
              n_jobs=-1, random_state=seed)
    try:                       # scikit-learn >= 1.7: alphas=<int>
        return ElasticNetCV(alphas=15, **kw)
    except TypeError:          # older scikit-learn
        return ElasticNetCV(n_alphas=15, **kw)


def make_models(seed):
    return {
        "elastic_net": lambda: make_pipeline(StandardScaler(), _enet(seed)),
        "xgboost": lambda: XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.3, min_child_weight=5, reg_lambda=1.0,
            n_jobs=-1, random_state=seed, verbosity=0),
    }


def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    mcfg = cfg["modeling"]
    proc = Path(cfg["out_dir"]) / "processed"
    out = Path(cfg["out_dir"]) / "models"
    out.mkdir(parents=True, exist_ok=True)

    X_all = pd.read_parquet(proc / "features.parquet")
    resp = pd.read_parquet(proc / "response.parquet")

    counts = resp.groupby("DRUG_NAME")["ModelID"].nunique()
    drugs = counts[counts >= mcfg["min_cell_lines_per_drug"]].sort_values(ascending=False).index.tolist()
    if mcfg.get("max_drugs"):
        drugs = drugs[: mcfg["max_drugs"]]
    print(f"modeling {len(drugs)} drugs")

    models = make_models(mcfg["seed"])
    kf = KFold(n_splits=mcfg["cv_folds"], shuffle=True, random_state=mcfg["seed"])
    metrics, oof_rows = [], []

    for i, drug in enumerate(drugs, 1):
        d = resp[resp["DRUG_NAME"] == drug].groupby("ModelID")["response"].mean()
        X = X_all.loc[d.index].values
        y = d.values
        meta = resp[resp["DRUG_NAME"] == drug].iloc[0]
        for name, factory in models.items():
            pred = np.zeros_like(y, dtype=float)
            for tr, te in kf.split(X):
                m = factory()
                m.fit(X[tr], y[tr])
                pred[te] = m.predict(X[te])
            r, _ = stats.pearsonr(y, pred)
            rho, _ = stats.spearmanr(y, pred)
            rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
            metrics.append({"drug": drug, "target": meta["target"], "pathway": meta["pathway"],
                            "model": name, "n_lines": len(y), "pearson_r": r,
                            "spearman_rho": rho, "rmse": rmse})
            oof_rows.append(pd.DataFrame({"drug": drug, "model": name, "ModelID": d.index,
                                          "observed": y, "predicted": pred}))
        last = [m for m in metrics if m["drug"] == drug]
        print(f"[{i}/{len(drugs)}] {drug:30s} " +
              "  ".join(f"{m['model']} r={m['pearson_r']:.2f}" for m in last))

    met = pd.DataFrame(metrics)
    met.to_csv(out / "cv_metrics.csv", index=False)
    pd.concat(oof_rows).to_parquet(out / "oof_predictions.parquet", index=False)

    summary = met.groupby("model")[["pearson_r", "spearman_rho", "rmse"]].median()
    summary.to_csv(out / "cv_summary.csv")
    print("\nmedian across drugs:\n", summary.round(3))
    by_path = (met[met.model == "xgboost"].groupby("pathway")["pearson_r"]
               .median().sort_values(ascending=False))
    by_path.to_csv(out / "xgboost_r_by_pathway.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    main(ap.parse_args().config)
