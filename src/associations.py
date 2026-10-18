"""GDSC-style mutation -> drug response association scan.

For every (drug name, hotspot-mutated gene) pair: Welch t-test of response in
mutant vs wild-type lines, effect size (Cohen's d, delta mean), and
Benjamini-Hochberg FDR across all tests. Negative delta = mutant lines are
MORE sensitive (lower IC50).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    proc = Path(cfg["out_dir"]) / "processed"
    out = Path(cfg["out_dir"]) / "associations"
    out.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(proc / "features.parquet")
    resp = pd.read_parquet(proc / "response.parquet")
    mut = feats[[c for c in feats.columns if c.startswith("mut__")]]

    rows = []
    # One test per drug name, averaging a line's response across DRUG_IDs when a
    # drug was screened more than once -- the same unit train_models.py uses.
    for drug, d in resp.groupby("DRUG_NAME"):
        drug_ids = ";".join(str(i) for i in sorted(d["DRUG_ID"].unique()))
        y = d.groupby("ModelID")["response"].mean()
        m = mut.loc[mut.index.intersection(y.index)]
        y = y.loc[m.index]
        for gene_col in m.columns:
            is_mut = m[gene_col].values == 1
            n_mut = int(is_mut.sum())
            if n_mut < 3 or (len(y) - n_mut) < 3:
                continue
            a, b = y.values[is_mut], y.values[~is_mut]
            t, p = stats.ttest_ind(a, b, equal_var=False)
            pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            rows.append({
                "DRUG_IDs": drug_ids, "drug": drug, "gene": gene_col[5:],
                "n_mut": n_mut, "n_wt": len(b),
                "delta_mean": a.mean() - b.mean(),
                "cohens_d": (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan,
                "p": p,
            })

    res = pd.DataFrame(rows)
    res["fdr"] = bh_fdr(res["p"].values)
    res = res.sort_values("p")
    res.to_csv(out / "mutation_drug_associations.csv", index=False)

    sig = res[(res["fdr"] < 0.05) & (res["delta_mean"] < 0)]
    print(f"tests: {len(res)}   sensitizing hits at FDR<5%: {len(sig)}")
    print(sig.head(15)[["drug", "gene", "n_mut", "delta_mean", "cohens_d", "fdr"]].to_string(index=False))

    # Positive-control check
    pcs = pd.DataFrame(cfg.get("positive_controls", []), columns=["drug", "gene"])
    if len(pcs):
        chk = pcs.merge(res, on=["drug", "gene"], how="left")
        chk.to_csv(out / "positive_controls.csv", index=False)
        print("\npositive controls:")
        print(chk[["drug", "gene", "n_mut", "delta_mean", "fdr"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    main(ap.parse_args().config)
