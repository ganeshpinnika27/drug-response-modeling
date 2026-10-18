# Pharmacogenomic Drug Response Modeling (GDSC2 × DepMap)

Predict cancer cell line drug sensitivity from transcriptomic and mutation
features, and find response biomarkers. Uses only public data.

**Questions**
1. Which hotspot mutations make cell lines significantly more sensitive or
   resistant to a drug? (GDSC-style association scan with FDR control.)
2. How well can drug response (LN IC50) be predicted from expression and
   mutation data? How does a sparse linear model (elastic net) compare with
   gradient boosting (XGBoost)?
3. Which features drive the predictions? (SHAP attribution, with known
   biomarkers used as positive controls.)

## Data
| Source | File | What it gives |
|---|---|---|
| GDSC release 8.5 (Sanger) | `GDSC2_fitted_dose_response.xlsx` | LN IC50 / AUC for ~280 drugs × ~970 lines |
| DepMap 24Q4 Public (Broad) | `Model.csv` | maps Sanger IDs (SIDM…) to DepMap IDs (ACH-…) |
| DepMap 24Q4 Public | `OmicsExpressionProteinCodingGenesTPMLogp1.csv` | log2(TPM+1) RNA-seq |
| DepMap 24Q4 Public | `OmicsSomaticMutationsMatrixHotspot.csv` | hotspot mutation calls per gene |

```bash
bash scripts/00_download.sh      # ~530 MB; GDSC from Sanger, DepMap from figshare (MD5-checked)
```

DepMap, Broad (2024). DepMap 24Q4 Public. figshare+. Dataset.
https://doi.org/10.25452/figshare.plus.27993248

## Run
```bash
conda env create -f environment.yml && conda activate drugresp
./run_all.sh --test              # 1-minute smoke test on synthetic data
./run_all.sh                     # real data (2 h 49 min for 270 drugs on a MacBook Air, Apple Silicon)
```
For a fast first pass, set `modeling.max_drugs: 20` in `config/config.yaml`.
On SCINet, wrap `run_all.sh` in an `sbatch` script with `--cpus-per-task=16`.

## Method
* **Features:** the top 2,000 most variable genes plus every annotated drug
  target gene, and binary hotspot mutation status for genes mutated in at
  least 10 lines. Variance filtering is unsupervised, so it doesn't leak the
  response into feature selection.
* **Associations:** for every drug × gene pair with at least 10 mutant and 10
  wild-type lines, the primary test is a tissue-adjusted linear model,
  `response ~ mutation + tissue` (DepMap OncotreeLineage; lineages with fewer
  than 10 screened lines pooled as "Other"), testing the mutation coefficient.
  An unadjusted Welch t-test is kept for comparison. Benjamini–Hochberg FDR is
  applied to each. A negative Δ means the mutant lines are more sensitive.
  Duplicate screens of a drug are averaged per cell line.
* **Prediction:** 5-fold outer CV with out-of-fold predictions. Elastic net
  tunes α (15 values from α_max down to α_max/100) and l1_ratio (0.1, 0.5,
  0.9) by inner 3-fold CV on each training fold, and scaling
  is fit inside each fold. Metrics are Pearson r, Spearman ρ and RMSE.
* **Explanation:** TreeExplainer SHAP on XGBoost refit to the full data for
  the best-predicted drugs. Lapatinib is also explained as an expression
  control: its biomarker is ERBB2 amplification, which shows up as high ERBB2
  expression rather than as a hotspot mutation.

## Outputs (`results/`)
| File | Content |
|---|---|
| `associations/mutation_drug_associations.csv` | all drug × mutation tests (unadjusted `p`/`fdr`, tissue-adjusted `p_adj`/`fdr_adj`) |
| `associations/positive_controls.csv` | e.g. BRAF→Dabrafenib, TP53→Nutlin-3a (resistance) |
| `models/expression_controls.csv` | SHAP rank of ERBB2 expression in the Lapatinib model |
| `models/cv_metrics.csv`, `cv_summary.csv` | per-drug and median CV performance |
| `models/xgboost_r_by_pathway.csv` | predictability by target pathway |
| `models/shap_top_features.csv` | top SHAP features per drug |
| `figures/fig1–3*.png` | model comparison, volcano plots (unadjusted vs tissue-adjusted), SHAP summaries |

## Results
GDSC2 (release 8.5) × DepMap 24Q4. 967 of 969 GDSC2 cell lines map to a
DepMap ID; 713 also have RNA-seq and hotspot mutation calls. Features: 2,112
expression genes (2,000 most variable + 112 drug targets) and 21 hotspot genes.

**Prediction.** 270 drugs had ≥ 300 screened lines (411–713 per drug, median 659).

| Model | Median CV Pearson r (IQR) | Median Spearman ρ |
|---|---|---|
| Elastic net | 0.554 (0.477–0.607) | 0.531 |
| XGBoost | 0.546 (0.479–0.610) | 0.529 |

- The two models are statistically indistinguishable: XGBoost is better on
  132 of 270 drugs (Wilcoxon signed-rank p = 0.56). Gradient boosting adds
  nothing over a sparse linear model here (fig1).
- 184 of 270 drugs reach r > 0.5. Best predicted: Nutlin-3a (0.75), AZD5991
  (MCL1, 0.73), Venetoclax (BCL2, 0.72), Irinotecan (TOP1, 0.70).
- By target pathway (XGBoost median r): p53 pathway 0.66, DNA replication 0.64
  and cell cycle 0.61 are most predictable; PI3K/MTOR (0.48) and WNT (0.45)
  are least.

**Associations.** 4,683 drug × mutation tests.

| | FDR < 5% hits | sensitizing | resistance |
|---|---|---|---|
| Unadjusted t-test | 537 | 131 | 406 |
| Tissue-adjusted | **50** | **40** | **10** |

- Without adjustment, tissue of origin dominates. 70 sensitizing hits were
  EZH2, whose 10 mutant lines are 80% lymphoid (lymphoma lines respond to
  many drugs), and 214 hits were KRAS, whose mutant lines are 75% lung,
  pancreas and bowel. After adjustment: EZH2 70 → 0, KRAS 214 → 12.
- The 40 adjusted sensitizing hits are mostly known pathway biology:
  - BRAF (13 hits): 12 are BRAF, MEK or ERK inhibitors (e.g. Dabrafenib,
    Trametinib, Ulixertinib).
  - PIK3CA (10): 4 PI3K inhibitors (Alpelisib, Taselisib, Pictilisib,
    Buparlisib) and 6 AKT inhibitors.
  - PTEN (6): 5 AKT inhibitors (Afuresertib, AZD5363, Ipatasertib,
    Uprosertib, GSK2110183B).
  - NRAS (5) and KRAS (4): MEK/ERK inhibitors only.
- KRAS mutation predicts resistance to EGFR/HER2 and other RTK inhibitors
  (Osimertinib, Lapatinib, Afatinib, Cediranib, Axitinib, Foretinib),
  consistent with KRAS signaling downstream of the receptor.
- Positive controls (tissue-adjusted): BRAF → Dabrafenib Δ = −3.03, FDR 9e-30;
  BRAF → Trametinib Δ = −1.39, FDR 0.004; BRAF → SB590885 Δ = −0.97,
  FDR 3e-4; TP53 → Nutlin-3a *resistance* Δ = +1.07, FDR 1e-16 (MDM2
  inhibition needs functional p53). EGFR → Erlotinib was not testable (8
  mutant lines).

**SHAP.** Top features recover known biomarkers:
- Irinotecan: SLFN11 expression, by a wide margin (mean |SHAP| 0.44 vs 0.10
  for the next feature), the established TOP1-inhibitor sensitivity marker.
- Venetoclax: expression of its target BCL2 ranks first.
- Nutlin-3a: MDM2 expression, TP53 mutation and CDKN2A expression are the top 3.
- Lapatinib (expression control): ERBB2 expression ranks 5th of 2,133
  features (XGBoost r = 0.54).

## Limitations / next steps
- Cell lines are an imperfect stand-in for patient tumors. Tissue adjustment
  removes between-lineage confounding but not confounders within a lineage
  (e.g. subtype).
- A few adjusted hits have no obvious mechanism (e.g. CTNNB1 → Nutlin-3a
  sensitivity, 16 mutant lines) and would need independent validation.
- SHAP is computed on models refit to all lines, so it explains what the model
  uses, not held-out performance.
- In profiling, elastic net picked l1_ratio = 0.1 (the most ridge-like value
  in the grid) for 2 of 3 drugs tested. Selected hyperparameters aren't saved
  per drug yet; saving them and widening the grid is a possible refinement.
- Possible extension: connectivity-map (LINC L1000) scoring for drug
  repurposing.
