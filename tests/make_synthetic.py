"""Generate small synthetic inputs in the real GDSC2/DepMap formats with
planted signals (BRAF-mutant lines sensitive to Dabrafenib; ERBB2 expression
drives Lapatinib). Used only to smoke-test the pipeline; NOT for results."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
out = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/synthetic")
out.mkdir(parents=True, exist_ok=True)

n_lines, n_genes = 400, 300
ach = [f"ACH-{i:06d}" for i in range(n_lines)]
sid = [f"SIDM{i:05d}" for i in range(n_lines)]
pd.DataFrame({"ModelID": ach, "SangerModelID": sid}).to_csv(out / "Model.csv", index=False)

genes = [f"GENE{i}" for i in range(n_genes - 3)] + ["ERBB2", "EGFR", "MDM2"]
expr = pd.DataFrame(rng.normal(3, 1, (n_lines, n_genes)), index=ach,
                    columns=[f"{g} ({1000+i})" for i, g in enumerate(genes)])
expr.index.name = ""
expr.to_csv(out / "OmicsExpressionProteinCodingGenesTPMLogp1.csv")

mut_genes = ["BRAF", "KRAS", "TP53", "PIK3CA", "EGFR"]
freqs = [0.12, 0.2, 0.45, 0.15, 0.05]
mut = pd.DataFrame({f"{g} ({2000+i})": rng.binomial(1, f, n_lines)
                    for i, (g, f) in enumerate(zip(mut_genes, freqs))}, index=ach)
mut.to_csv(out / "OmicsSomaticMutationsMatrixHotspot.csv")

braf = mut.iloc[:, 0].values
tp53 = mut.iloc[:, 2].values
erbb2 = expr["ERBB2 (1297)"].values
rows = []
drugs = {
    "Dabrafenib": (1, "BRAF", "ERK MAPK signaling", lambda: 3 - 2.5 * braf),
    "Lapatinib": (2, "ERBB2, EGFR", "EGFR signaling", lambda: 4 - 1.2 * (erbb2 - 3)),
    "Nutlin-3a (-)": (3, "MDM2", "p53 pathway", lambda: 2 + 1.8 * tp53),
    "Cisplatin": (4, "DNA crosslinker", "DNA replication", lambda: np.full(n_lines, 3.0)),
}
for name, (did, tgt, path, f) in drugs.items():
    y = f() + rng.normal(0, 0.7, n_lines)
    for s, v in zip(sid, y):
        rows.append({"SANGER_MODEL_ID": s, "DRUG_ID": did, "DRUG_NAME": name,
                     "PUTATIVE_TARGET": tgt, "PATHWAY_NAME": path, "LN_IC50": v, "AUC": np.nan})
pd.DataFrame(rows).to_csv(out / "GDSC2_fitted_dose_response.csv", index=False)

cfg = f"""data_dir: {out}
out_dir: {out}/results
gdsc: {{dose_response: {out}/GDSC2_fitted_dose_response.csv}}
depmap:
  model: {out}/Model.csv
  expression: {out}/OmicsExpressionProteinCodingGenesTPMLogp1.csv
  mutations: {out}/OmicsSomaticMutationsMatrixHotspot.csv
features: {{n_top_variable_genes: 200, min_mutated_lines: 10}}
modeling: {{response: LN_IC50, min_cell_lines_per_drug: 100, cv_folds: 5, seed: 42, max_drugs: null}}
positive_controls:
  - [Dabrafenib, BRAF]
  - [Nutlin-3a (-), TP53]
"""
(out / "config.yaml").write_text(cfg)
print(f"synthetic data + config in {out}")
