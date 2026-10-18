"""Merge GDSC2 drug response with DepMap expression + hotspot mutations.

Outputs (in results/processed/):
  response.parquet   long table: ModelID, DRUG_ID, DRUG_NAME, target, pathway, response
  features.parquet   wide table indexed by ModelID: expr__<GENE>, mut__<GENE>
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def clean_gene(col: str) -> str:
    # DepMap columns look like "BRAF (673)"; keep the symbol
    return re.sub(r"\s*\(\d+\)$", "", col)


def load_depmap_matrix(path: str, model_ids: set) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    # Some releases include extra ID columns; keep numeric gene columns only
    df = df.select_dtypes(include=[np.number])
    df.columns = [clean_gene(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    return df[df.index.isin(model_ids)]


def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    out = Path(cfg["out_dir"]) / "processed"
    out.mkdir(parents=True, exist_ok=True)

    # --- GDSC2 response ---------------------------------------------------
    gdsc_path = cfg["gdsc"]["dose_response"]
    gdsc = pd.read_excel(gdsc_path) if gdsc_path.endswith("xlsx") else pd.read_csv(gdsc_path)
    resp_col = cfg["modeling"]["response"]
    gdsc = gdsc[["SANGER_MODEL_ID", "DRUG_ID", "DRUG_NAME", "PUTATIVE_TARGET",
                 "PATHWAY_NAME", resp_col]].dropna(subset=[resp_col])

    # --- DepMap model map (Sanger ID -> DepMap ACH ID) --------------------
    model = pd.read_csv(cfg["depmap"]["model"])
    idmap = model[["ModelID", "SangerModelID"]].dropna()
    gdsc = gdsc.merge(idmap, left_on="SANGER_MODEL_ID", right_on="SangerModelID", how="inner")
    gdsc = gdsc.rename(columns={resp_col: "response",
                                "PUTATIVE_TARGET": "target",
                                "PATHWAY_NAME": "pathway"})
    # Same drug name can appear under >1 DRUG_ID (different screens); average per line
    gdsc = (gdsc.groupby(["ModelID", "DRUG_ID", "DRUG_NAME", "target", "pathway"], dropna=False)
                ["response"].mean().reset_index())

    ids = set(gdsc["ModelID"])

    # --- Features ----------------------------------------------------------
    expr = load_depmap_matrix(cfg["depmap"]["expression"], ids)
    n_top = cfg["features"]["n_top_variable_genes"]
    top = list(expr.var().sort_values(ascending=False).index[:n_top])
    # Always keep the drugs' annotated target genes, even if not highly variable
    targets = {t.strip() for s in gdsc["target"].dropna() for t in re.split(r"[,;/]", str(s))}
    keep_targets = [g for g in expr.columns if g in targets and g not in top]
    expr = expr.loc[:, top + keep_targets]
    print(f"added {len(keep_targets)} drug-target genes outside the top-variance set")
    expr.columns = ["expr__" + c for c in expr.columns]

    mut = load_depmap_matrix(cfg["depmap"]["mutations"], ids)
    mut = (mut > 0).astype(int)
    mut = mut.loc[:, mut.sum() >= cfg["features"]["min_mutated_lines"]]
    mut.columns = ["mut__" + c for c in mut.columns]

    feats = expr.join(mut, how="inner").fillna(0)
    gdsc = gdsc[gdsc["ModelID"].isin(feats.index)]

    feats.to_parquet(out / "features.parquet")
    gdsc.to_parquet(out / "response.parquet", index=False)

    print(f"cell lines with features+response: {feats.shape[0]}")
    print(f"features: {expr.shape[1]} expression, {mut.shape[1]} hotspot mutation genes")
    print(f"drugs: {gdsc['DRUG_NAME'].nunique()}  response rows: {len(gdsc)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    main(ap.parse_args().config)
