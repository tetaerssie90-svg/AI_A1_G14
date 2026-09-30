# Generative AI disclosure (AI_A1_G14)

The group used Cursor (Grok 4.6 agent) while building this package. AI did not replace testing, and every member must still be able to explain or change their own section in the live check.

## Tool

- Name: Cursor agent (Grok 4.6)
- Purpose: implement the remaining pipeline after a partial data-report script existed, generate the UI/UX and contributions PDFs, and write setup documentation.

## Important prompts / requests

- Read `SWE_3513_Assignment_1_Artificial_Intelligence by MClement.pdf` and report what was already finished.
- Follow the assignment workflow and complete the remaining technical product: data validation, NumPy gradient descent, classification, clustering, `predict.py`, README, evidence files, and PDFs.

## Files affected

- `run_all.py`, `predict.py`, `requirements.txt`, `README.md`
- `src/config.py`, `src/data.py`, `src/regression.py`, `src/classification.py`, `src/clustering.py`, `src/utils.py`
- `AI_A1_G14_UIUX.pdf`, `AI_A1_G14_CONTRIBUTIONS.pdf`
- `evidence/AI_USE.md`, `evidence/REGRESSION_DERIVATION.md`, `evidence/TEST_LOG.pdf`

## How the group verified the output

1. Installed `requirements.txt` in a clean virtual environment.
2. Ran `python run_all.py --data data/AI_A1_G14.csv --output artifacts/ --group AI-G14`.
3. Confirmed the printed SHA-256 matches `artifacts/data_report.json`.
4. Confirmed every required JSON, CSV, PNG, and model file was written by that command.
5. Ran `predict.py` with the assignment’s valid JSON record.
6. Ran `predict.py` with a deliberately incomplete record and confirmed a validation error.
7. Checked that regression uses NumPy only, that supervised scalers are fitted on train data, and that clustering excludes targets and `record_id`.

Members remain responsible for the live verification change (seed, learning rate, threshold, or k range) in `src/config.py`.
