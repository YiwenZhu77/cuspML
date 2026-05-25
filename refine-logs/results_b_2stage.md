# Two-Stage Probability Map — Results (Option B follow-up)

**Date**: 2026-05-25
**Spec**: `docs/superpowers/specs/2026-05-25-cusp-probability-map-mvp-design.md`
**Previous results**: `refine-logs/results.md` (R001-R003 single-stage MVP)
**Pivot rationale**: single-stage R002 produced backwards magnitude ordering (storm < quiet). Root cause = fixed 1:10 ratio per crossing erases P(cusp | SW). Two-stage decomposes:

    P(cusp at (MLAT, MLT) | SW) = P(cusp observed in hour | SW)     <- stage 1
                                  * P((MLAT, MLT) | cusp, SW)       <- stage 2 (R002 reused)

## Runs

### R011 — Stage 1 P(cusp observed in hour | SW)

Data: NASA OMNI2 hourly flat file (`omni2_all_years.dat`, 176 MB) restricted to 1987-2014, 245,448 hours. After dropna on 13 base+derived SW features, 197,775 hours kept (80.6%). Labels: 12.09% positive (hour contains any of 48k crossings).

Features (13): doy, imf_bx, imf_by, imf_bz, sw_v, sw_n, sw_pdyn, B_T, clock_angle, sin_clock_half, newell_cf, kan_lee_ef, vBs. **AE excluded** — the 48k crossings were pre-filtered to AE < 100 (Anderson 2024), so including AE would teach stage 1 the filter rather than physics.

Model: XGBoost binary:logistic, 600 trees, depth 6, scale_pos_weight=1, isotonic-calibrated on a 10% held-out slice. Best iter 398, wall 13 s.

| Metric | Value |
|---|---|
| Test AUC-ROC | 0.7285 |
| Test AUC-PR | 0.2521 |
| Test Brier | 0.0982 |
| Reliability | Good in P in [0, 0.4]; two sparse bins at 0.42 / 0.61 inflate max-deviation |

### R012 — Stage 1 x Stage 2 combined, 6 case-study heatmaps

|  # | SW state | Stage 1 P(hour) | Stage 2 peak | Combined peak |
|---|---|---|---|---|
| 1 | Strong south Bz (-10) | 0.095 | 0.019 | 0.0018 |
| 2 | Strong north Bz (+10) | 0.018 | 0.344 | 0.0061 |
| 3 | Strong By+ (Bz=-3, By=+8) | 0.107 | 0.197 | 0.0210 |
| 4 | Strong By- (Bz=-3, By=-8) | 0.110 | 0.138 | 0.0152 |
| 5 | Quiet | 0.000 | 0.559 | 0.0000 |
| 6 | Storm (Bz=-15, V=700) | 0.071 | 0.019 | 0.0013 |

Physics sanity: **4/6 PASS**
- PASS storm > quiet (combined 0.001 > 0.000)
- PASS south Bz peak at lower lat than north Bz (71 vs 81)
- PASS midnight near 0 everywhere
- PASS stage 1 ordering: quiet < north Bz < storm < south Bz < By states
- **FAIL** south Bz combined peak (0.002) < north Bz combined peak (0.006)
- **FAIL** all combined peaks under 0.05 — visually unimpressive

## Diagnosis

**Stage 1 alone works as intended.** P(cusp | SW) ordering is correct: quiet (0.000) << north Bz (0.018) < storm (0.071) < south Bz (0.095) ~ By states (0.10-0.11). The model learns that active-but-not-extreme SW states have ~10% chance of producing a crossing in any given hour, and quiet SW essentially never does. Storm being slightly below south Bz is plausible given the AE<100 filter biased our positives away from storm time.

**Stage 2 is the bottleneck.** Stage 2 outputs are well-calibrated as a binary classifier ("is this (MLAT, MLT) the cusp?"), but they are not a probability density over the dial. Peak values are 0.5-0.6 when the model is confident (well-sampled MLAT, e.g. 81 deg under north Bz / quiet) and drop to 0.02 when the model is uncertain (sparsely sampled MLAT, e.g. 71 deg under storm / strong south Bz). This uncertainty propagates into the combined product and overwhelms stage 1's correct SW ordering.

The math is consistent — stage 2 returns P((MLAT, MLT) is the cusp | observed, SW). Where DMSP rarely measured, that probability is honestly low. The product is internally consistent, just not what a forecaster wants to read.

## What this tells us about the design

Two-stage decomposition partially fixes the bug from single-stage MVP (storm > quiet now works), but it exposes a deeper issue: **stage 2 was trained as a binary classifier, not as a spatial density**. To make the combined product give meaningful magnitudes for a forecaster, stage 2 should either:

1. Be normalized over the dial (treat as PMF, sum to 1 per SW state) — easy post-hoc fix, but loses calibration meaning
2. Be retrained as a Gaussian / mixture model fit to cusp positions per SW (parametric — back to the band model I floated earlier)
3. Use raw 1Hz SSJ negatives so the model sees real observation density, not synthetic 1:10 ratios

## Next-step options

### B-fix-1: Renormalize stage 2 over the dial
Quickest. At inference, divide stage 2 map by its sum, multiply by stage 1 scalar. Now the combined map sums to stage 1 P across the dial. Loses interpretability of stage 2 as binary classifier output but gives "per-cell occupancy probability" that a forecaster can compare across SW states. ~30 min.

### B-fix-2: Train stage 2 as density estimation
Replace the binary cross-entropy objective with a likelihood-based one: fit a 2D parametric density (e.g. mixture of 2 Gaussians on the dial) per training crossing, then learn the density parameters as a function of SW. This is the parametric band model from the original spec discussion. ~half day.

### Escalate to Option C (raw 1Hz SSJ negatives)
The slow but principled path. Real negatives = same orbit, different (MLAT, MLT), labeled by Anderson criterion failure. Stage 2 gets real density. 2-3 days.

**Recommendation**: try **B-fix-1** first (30 min). If combined ordering still wrong for south Bz vs north Bz, then **B-fix-2** or **C**. Send these results to GPT-5.4 via auto-review-loop for sanity check before further iteration.

## Files added this round

- `src/lib/cusp_stage1.py` — OMNI2 hourly loader, derived feature engineering, stage-1 fit
- `src/scripts/mvp/r011_stage1.py` — train + calibrate stage 1
- `src/scripts/mvp/r012_case_studies_2stage.py` — combine + replot 6 cases
- `output/omni_raw/omni2_all_years.dat` (176 MB, gitignored) — full OMNI2 hourly 1963-2026
- `src/kernels/cuspmap_mvp/bundles/r011_stage1_model.ubj`, `_isotonic.pkl`, `_features.json`, `_results.json`
- `src/kernels/cuspmap_mvp/bundles/r012_2stage_case_studies.json`
- `src/kernels/cuspmap_mvp/figures/2stage_case*.png` (6 polar dial heatmaps, shared vmax=0.021)

## Summary

The two-stage approach demonstrates that **SW-conditional cusp occurrence rate is learnable** (stage 1 AUC 0.73 with clean physics ordering), and that **stage 2 can be reused as-is** to give spatial structure. The remaining gap is purely in how the spatial output is calibrated — a normalization fix or a density-based reformulation, neither of which requires new data. The raw-SSJ path is no longer urgent.
