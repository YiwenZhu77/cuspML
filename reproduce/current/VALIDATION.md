# Validation of the revised manuscript reproduction

Validated on 2026-09-11 using Python 3.10.16 in a fresh virtual environment without system site packages. Exact dependency versions are recorded in `requirements-lock.txt`; input hashes and machine-readable results are in `validation.json`.

All models were fitted from the released processed crossing table. Strict verification passed for **53 prediction sets**, including **34 grouped test folds**, all four targets, the empirical and machine-learning baselines, both history-window experiments and the hemisphere balance control. All **60 distributed input files** passed SHA-256 checks. The final model uses 74 ordered features, 29,935 training crossings and 9,733 temporal test crossings. The independent-window 60-minute anchor reuses the same primary prediction.

The primary MAE was **1.1095330013 degrees**, versus **1.7486284771 degrees** for the frozen Newell calibration: **36.5483854% lower MAE**. The calibration table is distributed explicitly; no test labels enter its fit.

Tree SHAP on 2,000 test crossings, gain importance, and all 150 partial-dependence grid predictions were independently recomputed and matched. All **10 manuscript and Supporting Information figures** were generated from the newly trained results and visually inspected. The five regression tests also passed.

The full initial training job took **12 min 51 s** on a Casper CPU compute node, excluding queue time. GBR accounted for about 290 seconds. Ridge alone was subsequently refitted with one BLAS thread, producing predictions bitwise identical to its frozen reference. The released code contains that setting. Complete reproduction uses **six threads for MLP**; an independent four-thread check changed its optimization result and failed strict comparison. These settings were fixed rather than increasing tolerances. Individual predictions retain absolute tolerance `2e-4`; MAE, RMSE and correlation retain absolute tolerance `2e-5`.

This validates modeling and figures **from processed inputs**. It does not rerun raw DMSP decoding, precipitation label identification, or a new download of the mutable upstream OMNI archive. The previous Zenodo package remains a historical snapshot; this release does not update that record.
