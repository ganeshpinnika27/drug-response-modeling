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
./run_all.sh                     # real data (≈30–90 min for all drugs on 8 cores)
```
For a fast first pass, set `modeling.max_drugs: 20` in `config/config.yaml`.
On SCINet, wrap `run_all.sh` in an `sbatch` script with `--cpus-per-task=16`.

## Method
* **Features:** the top 2,000 most variable genes plus every annotated drug
  target gene, and binary hotspot mutation status for genes mutated in at
  least 10 lines. Variance filtering is unsupervised, so it doesn't leak the
  response into feature selection.
* **Associations:** Welch t-test of response in mutant vs wild-type lines for
  every drug × gene pair, with Benjamini–Hochberg FDR. A negative Δ means the
  mutant lines are more sensitive.
* **Prediction:** 5-fold outer CV with out-of-fold predictions. Elastic net
  tunes α and l1_ratio by inner 3-fold CV on each training fold, and scaling
  is fit inside each fold. Metrics are Pearson r, Spearman ρ and RMSE.
* **Explanation:** TreeExplainer SHAP on XGBoost refit to the full data for
  the best-predicted drugs.

## Outputs (`results/`)
| File | Content |
|---|---|
| `associations/mutation_drug_associations.csv` | all drug × mutation tests |
| `associations/positive_controls.csv` | e.g. BRAF→Dabrafenib, TP53→Nutlin-3a |
| `models/cv_metrics.csv`, `cv_summary.csv` | per-drug and median CV performance |
| `models/xgboost_r_by_pathway.csv` | predictability by target pathway |
| `models/shap_top_features.csv` | top SHAP features per drug |
| `figures/fig1–3*.png` | model comparison, volcano plot, SHAP summaries |

## Results
*Fill in after running on real data. Don't copy the synthetic test numbers.*

- Drugs modeled: __ ; cell lines: __
- Median CV Pearson r: elastic net __ , XGBoost __
- Best-predicted pathways: __
- Significant sensitizing associations (FDR < 5%): __ ; positive controls recovered: __
- Notable SHAP findings: __

## Limitations / next steps
- Cell lines are an imperfect stand-in for patient tumors. Tissue of origin
  confounds many associations, so add tissue as a covariate or run
  tissue-stratified tests.
- Possible extension: connectivity-map (LINC L1000) scoring for drug
  repurposing.
