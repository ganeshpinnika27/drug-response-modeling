"""Figures + SHAP biomarker attribution.

  fig1_model_comparison.png    per-drug Pearson r, elastic net vs XGBoost
  fig2_association_volcano.png mutation-drug volcano, unadjusted vs tissue-adjusted
  fig3_shap_<drug>.png         SHAP summary for the best-predicted drugs
  shap_top_features.csv        top SHAP features per explained drug
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml
from xgboost import XGBRegressor


def _label_points(ax, pts, xcol, pcol):
    """Label points with text stacked so labels on each side don't overlap."""
    ymax = -np.log10(max(pts[pcol].min(), 1e-300)) if len(pts) else 1
    gap = 0.06 * ymax
    for side, sub in ((-1, pts[pts[xcol] < 0]), (1, pts[pts[xcol] >= 0])):
        last = np.inf
        for _, r in sub.sort_values(pcol).iterrows():
            yp = -np.log10(max(r[pcol], 1e-300))
            yt = min(yp, last - gap)
            last = yt
            ax.annotate(f"{r['gene']}:{r['drug']}", (r[xcol], yp),
                        xytext=(r[xcol] + 0.25 * side, yt), fontsize=6,
                        ha="left" if side > 0 else "right", va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="grey"))


def main(cfg_path: str, n_explain: int):
    cfg = yaml.safe_load(open(cfg_path))
    root = Path(cfg["out_dir"])
    figs = root / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    met = pd.read_csv(root / "models" / "cv_metrics.csv")
    assoc = pd.read_csv(root / "associations" / "mutation_drug_associations.csv")
    X_all = pd.read_parquet(root / "processed" / "features.parquet")
    resp = pd.read_parquet(root / "processed" / "response.parquet")

    # Fig 1: model comparison
    wide = met.pivot(index="drug", columns="model", values="pearson_r").dropna()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(wide["elastic_net"], wide["xgboost"], s=12, alpha=0.6)
    lim = [min(wide.min().min(), 0), max(wide.max().max(), 0.8)]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("Elastic net  (CV Pearson r)")
    ax.set_ylabel("XGBoost  (CV Pearson r)")
    ax.set_title(f"Per-drug prediction accuracy (n={len(wide)} drugs)")
    fig.tight_layout(); fig.savefig(figs / "fig1_model_comparison.png", dpi=200); plt.close(fig)

    # Fig 2: volcano, unadjusted vs tissue-adjusted
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    panels = [("delta_mean", "p", "fdr", "Unadjusted (Welch t-test)"),
              ("delta_adj", "p_adj", "fdr_adj", "Tissue-adjusted (response ~ mutation + tissue)")]
    for ax, (dcol, pcol, fcol, title) in zip(axes, panels):
        a = assoc.dropna(subset=[pcol])
        y = -np.log10(a[pcol].clip(lower=1e-300))
        sig = a[fcol] < 0.05
        ax.scatter(a[dcol], y, s=6, c=np.where(sig, "tab:red", "lightgrey"))
        _label_points(ax, a[sig].nsmallest(10, pcol), dcol, pcol)
        ax.set_xlim(a[dcol].min() - 1.6, a[dcol].max() + 1.4)   # room for labels
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("Δ response (mutant − WT)   ← more sensitive")
        ax.set_title(f"{title}\n{int(sig.sum())} hits at FDR < 5%", fontsize=10)
    axes[0].set_ylabel("−log10 p")
    fig.tight_layout(); fig.savefig(figs / "fig2_association_volcano.png", dpi=200); plt.close(fig)

    # SHAP on best-predicted drugs
    seed = cfg["modeling"]["seed"]
    best = (met[met.model == "xgboost"].sort_values("pearson_r", ascending=False)
            .head(n_explain)["drug"].tolist())
    # Expression-biomarker controls are explained even if not among the best
    expr_ctl = [(d, g) for d, g in cfg.get("expression_controls", [])
                if d in set(resp["DRUG_NAME"])]
    best += [d for d, _ in expr_ctl if d not in best]
    rows, ctl_rows = [], []
    for drug in best:
        d = resp[resp["DRUG_NAME"] == drug].groupby("ModelID")["response"].mean()
        X = X_all.loc[d.index]
        m = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
                         colsample_bytree=0.3, min_child_weight=5, n_jobs=-1,
                         random_state=seed, verbosity=0).fit(X, d.values)
        sv = shap.TreeExplainer(m).shap_values(X)
        imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
        for rank, (feat, val) in enumerate(imp.head(10).items(), 1):
            rows.append({"drug": drug, "rank": rank, "feature": feat, "mean_abs_shap": val})
        for d, gene in expr_ctl:
            feat = f"expr__{gene}"
            if d == drug and feat in imp.index:
                ctl_rows.append({"drug": drug, "feature": feat,
                                 "shap_rank": int(imp.index.get_loc(feat)) + 1,
                                 "n_features": len(imp), "mean_abs_shap": imp[feat]})
        plt.figure()
        shap.summary_plot(sv, X, max_display=12, show=False)
        plt.title(drug)
        plt.tight_layout(); plt.savefig(figs / f"fig3_shap_{drug.replace('/', '_').replace(' ', '_')}.png", dpi=200)
        plt.close("all")
    pd.DataFrame(rows).to_csv(root / "models" / "shap_top_features.csv", index=False)
    if ctl_rows:
        ctl = pd.DataFrame(ctl_rows)
        ctl.to_csv(root / "models" / "expression_controls.csv", index=False)
        print("expression controls (SHAP rank):\n", ctl.to_string(index=False))
    print(f"figures written to {figs}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--n-explain", type=int, default=6)
    a = ap.parse_args()
    main(a.config, a.n_explain)
