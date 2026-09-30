# AI_A1_G14 Potato Cooperative Decision Pipeline

Reproducible command-line pipeline for the **Musanze HarvestLink Cooperative** (SWE 3513 Assignment 1). The pipeline estimates harvest weight, flags consignments that need dispatch attention, and groups collection points with similar operating profiles.

## Group information

- Group number: **G14**
- Group verification code: **AI-G14**
- Group leader: **Teta Erssie**
- Members and roles:
  - Member 1, Data and UX lead: Joel Eliezer Youto Mongar Jr
  - Member 2, Regression engineer: Niyibigira Gad
  - Member 3, Classification engineer: Uwiringiyimana Marie Claire
  - Member 4, Clustering and QA engineer: Mbabazi Sandrine
  - Member 5, Reproducibility and release lead: Teta Erssie
- GitHub repository URL: *add the public clone URL after pushing*
- Final commit hash: *filled after the last commit*
- Dataset SHA-256 fingerprint: *printed by `run_all.py` and saved in `artifacts/data_report.json`*

## Tested environment

- Python **3.10+** (this package was executed on Python 3.14.5)
- Operating system: macOS or any environment that can create a virtualenv
- Required packages: see `requirements.txt` (NumPy, pandas, scikit-learn, matplotlib, seaborn, joblib)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the full pipeline

From the project root (`AI_A1_G14/`):

```bash
python run_all.py --data data/AI_A1_G14.csv --output artifacts/ --group AI-G14
```

The command must print the group code and the dataset SHA-256 fingerprint. It overwrites files in `artifacts/` and `models/`.

## Predict one record

Valid record:

```bash
python predict.py --record '{"plot_area_ha":1.2,"rainfall_mm":81,"soil_ph":5.7,"seed_kg":210,"distance_km":14,"arrival_hour":9}'
```

Malformed or incomplete records are rejected with a JSON error, for example:

```bash
python predict.py --record '{"plot_area_ha":1.2}'
```

## Expected outputs

After a successful `run_all.py` run:

```
artifacts/data_report.json
artifacts/regression_metrics.json
artifacts/regression_loss.png
artifacts/classification_metrics.json
artifacts/confusion_matrix.png
artifacts/clustering_metrics.json
artifacts/clusters.csv
artifacts/cluster_plot.png
models/regression.joblib
models/classifier.joblib
models/cluster.joblib
models/meta.json
```

`predict.py` prints JSON with:

- `regression_prediction_kg`
- `classification_prediction`
- `classification_probability`
- `cluster_label`
- `group_code`
- `model_version`

## What each module does

| Path | Role |
|---|---|
| `src/data.py` | Schema checks, missing/duplicate report, NumPy feature matrix, SHA-256 |
| `src/regression.py` | NumPy linear regression and batch gradient descent |
| `src/classification.py` | Logistic regression for `dispatch_attention` |
| `src/clustering.py` | Standardized K-means, silhouette for k = 2..5 |
| `predict.py` | Single-record scoring and input validation |

Fixed random seed: **42** (`src/config.py`). Supervised scalers are fitted on the training split only. Clustering uses input features only; `record_id`, `actual_yield_kg`, and `dispatch_attention` are never clustering inputs.

## Known limitations

- The lecturer-issued file currently in `data/AI_A1_G14.csv` has five rows. Test MAE, F1, and silhouette values on this file are unstable. The assessor’s hidden CSV with the same schema is the real check that outputs are computed, not hard-coded.
- Linear regression assumes a linear relationship in the scaled feature space.
- Logistic regression reports a probability, not a guarantee that a consignment is unsafe.
- Clusters are unsupervised groupings. They are not verified farm types.
- `record_id` is an identifier and is never used as a model feature.

## Evidence

- `evidence/AI_USE.md` — generative AI disclosure
- `evidence/REGRESSION_DERIVATION.md` — gradient-descent notes
- `evidence/TEST_LOG.pdf` — clean-run record
- `AI_A1_G14_UIUX.pdf` — staff dashboard design (separate Moodle upload)
- `AI_A1_G14_CONTRIBUTIONS.pdf` — signed contribution mapping (separate Moodle upload)

Do not submit `venv/`, `.venv/`, `__pycache__/`, or notebook checkpoints.
