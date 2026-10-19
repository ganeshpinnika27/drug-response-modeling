"""External validation of GDSC results in the PRISM Repurposing screen.

PRISM (Broad, 20Q2 secondary screen) measured many of the same drugs with a
different assay (pooled barcoded cell lines) and includes cell lines that
were never screened in GDSC. Drugs are matched by normalized name; the PRISM
response is the dose-response AUC (lower = more sensitive), preferring the
MTS010 screen where a drug-line pair was measured more than once, as the
PRISM readme recommends. Only lines that passed STR profiling are used.

For each drug in both screens:
  1. screen agreement: Spearman rho between measured GDSC LN IC50 and PRISM
     AUC on shared cell lines (the ceiling for any model)
  2. shared lines: rho between GDSC out-of-fold predictions and PRISM AUC
  3. new lines: models trained on all GDSC lines predict PRISM lines never
     screened in GDSC; rho between predictions and PRISM AUC
Biomarker replication: each tissue-adjusted GDSC association (FDR < 5%) for a
matched drug is re-tested in PRISM with the same model, response ~ mutation +
tissue. All GDSC tests that were not significant are re-tested too, as a
baseline for how often associations replicate by chance.

Outputs in results/prism/: drug_validation.csv, association_replication.csv,
and figures/fig4_prism_validation.png
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from associations import ols_mutation_effect          # noqa: E402
from prepare_data import clean_gene                    # noqa: E402
from train_models import make_models                   # noqa: E402

SCREEN_PREFERENCE = {"MTS010": 0, "HTS002": 1, "MTS006": 2, "MTS005": 3}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def load_prism(path):
    p = pd.read_csv(path, usecols=["depmap_id", "name", "screen_id", "auc", "passed_str_profiling"],
                    low_memory=False)
    p = p[(p["passed_str_profiling"] == True) & p["auc"].notna() & p["name"].notna()].copy()  # noqa: E712
    p["key"] = p["name"].map(norm)
    p["rank"] = p["screen_id"].map(SCREEN_PREFERENCE).fillna(9)
    best = p.groupby(["key", "depmap_id"])["rank"].transform("min")
    p = p[p["rank"] == best]
    return p.groupby(["key", "depmap_id"])["auc"].mean()


def build_features(cfg, ids, columns):
    """DepMap features for extra cell lines, with the GDSC feature columns."""
    expr_genes = [c[6:] for c in columns if c.startswith("expr__")]
    mut_genes = [c[5:] for c in columns if c.startswith("mut__")]
    out = []
    for path, genes, prefix, binarize in [(cfg["depmap"]["expression"], expr_genes, "expr__", False),
                                          (cfg["depmap"]["mutations"], mut_genes, "mut__", True)]:
        m = pd.read_csv(path, index_col=0)
        m = m.select_dtypes(include=[np.number])
        m.columns = [clean_gene(c) for c in m.columns]
        m = m.loc[:, ~m.columns.duplicated()]
        m = m[m.index.isin(ids)].reindex(columns=genes)
        if binarize:
            m = (m > 0).astype(int)
        m.columns = [prefix + g for g in genes]
        out.append(m)
    feats = out[0].join(out[1], how="inner")
    missing = int(feats.isna().all().sum())
    if missing:
        print(f"note: {missing} feature columns absent for the extra lines; filled with 0")
    return feats.reindex(columns=columns).fillna(0)


def spearman(a, b):
    if len(a) < 10:
        return np.nan, np.nan
    r = stats.spearmanr(a, b)
    return r.statistic, r.pvalue


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    if "prism" not in cfg:
        print("no prism section in config; skipping PRISM validation")
        return
    root = Path(cfg["out_dir"])
    out = root / "prism"; out.mkdir(parents=True, exist_ok=True)
    figs = root / "figures"; figs.mkdir(parents=True, exist_ok=True)

    X_gdsc = pd.read_parquet(root / "processed" / "features.parquet")
    resp = pd.read_parquet(root / "processed" / "response.parquet")
    met = pd.read_csv(root / "models" / "cv_metrics.csv")
    oof = pd.read_parquet(root / "models" / "oof_predictions.parquet")
    assoc = pd.read_csv(root / "associations" / "mutation_drug_associations.csv")
    prism = load_prism(cfg["prism"]["dose_response"])

    gdsc_names = {norm(d): d for d in resp["DRUG_NAME"].unique()}
    shared_keys = sorted(set(gdsc_names) & set(prism.index.get_level_values("key")))
    prism_lines = set(prism.index.get_level_values("depmap_id"))
    gdsc_lines = set(resp["ModelID"])
    new_ids = prism_lines - gdsc_lines
    X_new = build_features(cfg, new_ids, list(X_gdsc.columns))
    print(f"drugs in both screens: {len(shared_keys)}; PRISM lines not in GDSC with DepMap "
          f"features: {len(X_new)}")

    models = make_models(cfg["modeling"]["seed"])
    rows = []
    for i, key in enumerate(shared_keys, 1):
        drug = gdsc_names[key]
        y_gdsc = resp[resp["DRUG_NAME"] == drug].groupby("ModelID")["response"].mean()
        y_prism = prism.loc[key]
        shared = y_gdsc.index.intersection(y_prism.index)
        row = {"drug": drug, "n_gdsc": len(y_gdsc), "n_shared": len(shared)}
        row["rho_measured"], row["p_measured"] = spearman(y_gdsc[shared], y_prism[shared])

        o = oof[oof["drug"] == drug]
        for name in models:
            pred = o[o["model"] == name].set_index("ModelID")["predicted"]
            s = pred.index.intersection(y_prism.index)
            row[f"rho_shared_{name}"], _ = spearman(pred[s], y_prism[s])

        new = X_new.index.intersection(y_prism.index)
        row["n_new"] = len(new)
        if len(new) >= 10:
            X_tr = X_gdsc.loc[y_gdsc.index].values
            for name, factory in models.items():
                m = factory().fit(X_tr, y_gdsc.values)
                row[f"rho_new_{name}"], row[f"p_new_{name}"] = spearman(
                    m.predict(X_new.loc[new].values), y_prism[new])
        cv = met[(met["drug"] == drug)].set_index("model")["pearson_r"]
        for name in models:
            row[f"gdsc_cv_r_{name}"] = cv.get(name, np.nan)
        rows.append(row)
        print(f"[{i}/{len(shared_keys)}] {drug:28s} measured rho={row['rho_measured']:.2f}  "
              f"new-line rho (enet)={row.get('rho_new_elastic_net', np.nan):.2f}", flush=True)

    val = pd.DataFrame(rows)
    val.to_csv(out / "drug_validation.csv", index=False)
    cols = ["rho_measured", "rho_shared_elastic_net", "rho_shared_xgboost",
            "rho_new_elastic_net", "rho_new_xgboost"]
    print("\nmedian Spearman rho across drugs:\n", val[cols].median().round(3).to_string())
    sig_new = (val["p_new_elastic_net"] < 0.05) & (val["rho_new_elastic_net"] > 0)
    print(f"drugs with significant positive new-line rho (elastic net): {int(sig_new.sum())} of {len(val)}")

    # --- biomarker replication -------------------------------------------
    lineage = pd.read_csv(cfg["depmap"]["model"]).set_index("ModelID")["OncotreeLineage"]
    mut_all = build_features(cfg, prism_lines, [c for c in X_gdsc.columns if c.startswith("mut__")])
    min_n = cfg.get("associations", {}).get("min_group_lines", 10)
    min_tissue = cfg.get("associations", {}).get("min_tissue_lines", 10)
    rep = []
    tested = assoc[assoc["drug"].map(norm).isin(shared_keys) & assoc["p_adj"].notna()]
    for (drug, gene), a in tested.groupby(["drug", "gene"]):
        y = prism.loc[norm(drug)]
        ids = y.index.intersection(mut_all.index)
        if f"mut__{gene}" not in mut_all.columns:
            continue
        is_mut = mut_all.loc[ids, f"mut__{gene}"].values == 1
        if is_mut.sum() < min_n or (~is_mut).sum() < min_n:
            continue
        tis = lineage.reindex(ids).fillna("Unknown")
        tis = tis.where(tis.map(tis.value_counts()) >= min_tissue, "Other")
        dummies = pd.get_dummies(tis, drop_first=True).values.astype(float)
        beta, p = ols_mutation_effect(y[ids].values, is_mut, dummies)
        g = a.iloc[0]
        rep.append({"drug": drug, "gene": gene, "gdsc_delta_adj": g["delta_adj"],
                    "gdsc_fdr_adj": g["fdr_adj"], "gdsc_significant": g["fdr_adj"] < 0.05,
                    "prism_n_mut": int(is_mut.sum()), "prism_delta_auc": beta, "prism_p": p})
    rep = pd.DataFrame(rep)
    rep["same_direction"] = np.sign(rep["gdsc_delta_adj"]) == np.sign(rep["prism_delta_auc"])
    rep["replicated"] = rep["same_direction"] & (rep["prism_p"] < 0.05)
    rep.to_csv(out / "association_replication.csv", index=False)
    for flag, label in [(True, "GDSC significant (FDR<5%)"), (False, "GDSC not significant")]:
        s = rep[rep["gdsc_significant"] == flag]
        print(f"{label}: {len(s)} testable in PRISM; same direction {s['same_direction'].mean():.0%}; "
              f"replicated (same direction, p<0.05) {s['replicated'].mean():.0%}")

    # --- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    labels = ["Measured\nGDSC vs PRISM", "Model, shared lines\n(elastic net, OOF)",
              "Model, new lines\n(elastic net)", "Model, new lines\n(XGBoost)"]
    data = [val[c].dropna() for c in ["rho_measured", "rho_shared_elastic_net",
                                      "rho_new_elastic_net", "rho_new_xgboost"]]
    axes[0].boxplot(data)
    axes[0].set_xticks(range(1, 5)); axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].axhline(0, color="grey", lw=0.6)
    axes[0].set_ylabel("Spearman rho with PRISM AUC")
    axes[0].set_title(f"Agreement with PRISM ({len(val)} drugs)", fontsize=10)
    v = val.dropna(subset=["rho_measured", "rho_new_elastic_net"])
    axes[1].scatter(v["rho_measured"], v["rho_new_elastic_net"], s=12, alpha=0.7)
    lim = [min(v[["rho_measured", "rho_new_elastic_net"]].min().min(), 0) - 0.05, 1]
    axes[1].plot(lim, lim, "k--", lw=0.8)
    axes[1].set_xlabel("Measured agreement, GDSC vs PRISM (shared lines)")
    axes[1].set_ylabel("Model agreement on new lines")
    axes[1].set_title("Prediction on new lines vs screen agreement", fontsize=10)
    fig.tight_layout(); fig.savefig(figs / "fig4_prism_validation.png", dpi=200); plt.close(fig)
    print(f"written to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    main(ap.parse_args().config)
