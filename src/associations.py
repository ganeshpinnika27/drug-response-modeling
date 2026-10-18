"""GDSC-style mutation -> drug response association scan.

For every (drug name, hotspot-mutated gene) pair:
  * unadjusted: Welch t-test of response in mutant vs wild-type lines, with
    effect size (Cohen's d, delta mean)
  * tissue-adjusted: OLS  response ~ mutation + tissue  (tissue = DepMap
    OncotreeLineage), testing the mutation coefficient. Lineages with few
    screened lines are pooled into "Other". This keeps mutations that are
    enriched in a tissue (e.g. EZH2 in lymphoma, KRAS in pancreas) from
    standing in for that tissue's overall drug sensitivity.
Benjamini-Hochberg FDR is applied to each set of tests separately.
Negative delta = mutant lines are MORE sensitive (lower IC50).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(n)
    res[order] = np.clip(q, 0, 1)
    out[ok] = res
    return out


def ols_mutation_effect(y: np.ndarray, is_mut: np.ndarray, tissue_dummies: np.ndarray):
    """Coefficient, p-value for `is_mut` in y ~ 1 + is_mut + tissue dummies."""
    X = np.column_stack([np.ones(len(y)), is_mut.astype(float), tissue_dummies])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return np.nan, np.nan      # mutation confounded with tissue: not estimable
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    df = len(y) - X.shape[1]
    sigma2 = np.sum((y - X @ beta) ** 2) / df
    se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1])
    t = beta[1] / se
    return beta[1], 2 * stats.t.sf(abs(t), df)


def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    proc = Path(cfg["out_dir"]) / "processed"
    out = Path(cfg["out_dir"]) / "associations"
    out.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(proc / "features.parquet")
    resp = pd.read_parquet(proc / "response.parquet")
    mut = feats[[c for c in feats.columns if c.startswith("mut__")]]
    lineage = (pd.read_csv(cfg["depmap"]["model"]).set_index("ModelID")["OncotreeLineage"]
                 .reindex(feats.index).fillna("Unknown"))

    acfg = cfg.get("associations", {})
    min_n = acfg.get("min_group_lines", 10)
    min_tissue = acfg.get("min_tissue_lines", 10)
    rows = []
    # One test per drug name, averaging a line's response across DRUG_IDs when a
    # drug was screened more than once -- the same unit train_models.py uses.
    for drug, d in resp.groupby("DRUG_NAME"):
        drug_ids = ";".join(str(i) for i in sorted(d["DRUG_ID"].unique()))
        y = d.groupby("ModelID")["response"].mean()
        m = mut.loc[mut.index.intersection(y.index)]
        y = y.loc[m.index]
        tis = lineage.loc[m.index]
        counts = tis.value_counts()
        tis = tis.where(tis.map(counts) >= min_tissue, "Other")
        dummies = pd.get_dummies(tis, drop_first=True).values.astype(float)
        for gene_col in m.columns:
            is_mut = m[gene_col].values == 1
            n_mut = int(is_mut.sum())
            if n_mut < min_n or (len(y) - n_mut) < min_n:
                continue
            a, b = y.values[is_mut], y.values[~is_mut]
            t, p = stats.ttest_ind(a, b, equal_var=False)
            pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            beta, p_adj = ols_mutation_effect(y.values, is_mut, dummies)
            rows.append({
                "DRUG_IDs": drug_ids, "drug": drug, "gene": gene_col[5:],
                "n_mut": n_mut, "n_wt": len(b),
                "delta_mean": a.mean() - b.mean(),
                "cohens_d": (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan,
                "p": p,
                "delta_adj": beta, "p_adj": p_adj,
            })

    res = pd.DataFrame(rows)
    res["fdr"] = bh_fdr(res["p"].values)
    res["fdr_adj"] = bh_fdr(res["p_adj"].values)
    res = res.sort_values("p_adj")
    res.to_csv(out / "mutation_drug_associations.csv", index=False)

    def n_hits(fdr, delta):
        s = res[fdr] < 0.05
        return int((s & (res[delta] < 0)).sum()), int((s & (res[delta] > 0)).sum())
    print(f"tests: {len(res)}  (tissue-adjusted estimable: {res['p_adj'].notna().sum()})")
    print("FDR<5%% hits  sensitizing / resistance:  unadjusted %d / %d   tissue-adjusted %d / %d"
          % (*n_hits("fdr", "delta_mean"), *n_hits("fdr_adj", "delta_adj")))
    sig = res[(res["fdr_adj"] < 0.05) & (res["delta_adj"] < 0)]
    print(sig.head(15)[["drug", "gene", "n_mut", "delta_adj", "fdr_adj", "fdr"]].to_string(index=False))

    # Positive-control check
    pcs = pd.DataFrame(cfg.get("positive_controls", []), columns=["drug", "gene"])
    if len(pcs):
        chk = pcs.merge(res, on=["drug", "gene"], how="left")
        chk.to_csv(out / "positive_controls.csv", index=False)
        print("\npositive controls:")
        print(chk[["drug", "gene", "n_mut", "delta_mean", "fdr", "delta_adj", "fdr_adj"]]
              .to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    main(ap.parse_args().config)
