# Cusp Probability Map MVP — Final Results

**Date**: 2026-05-25
**Codex review**: Round 3 score **9/10**, verdict **Ready** to start paper writing
**Commits**: c8cca77 (R013 first commit), uncommitted R009 + R014 + R015 + R016 + R017 pending one final commit

## Pipeline (final)

```
SW state at time T (74 features from OMNI + derived)
       |
       |----> stage 1 [R015 opp-restricted]
       |       P(DMSP-detectable cusp crossing in hour | SW)
       |       trained on 147,521 opportunity hours (DMSP active within +/-24 h),
       |       16.2% positive rate, XGBoost binary:logistic + isotonic
       |       -> scalar s1
       |
       \----> stage 2 [R002, paper-1 hyperparams]
               P((MLAT, MLT) is the cusp | observed, SW)
               trained on 2.18M expanded rows from 39,668 crossings,
               5 pos + 50 strat. neg per crossing, polar Cartesian (x, y),
               XGBoost binary:logistic + isotonic
               -> 2D grid s2(MLAT, MLT)
                     |
                     v
       combined(MLAT, MLT, SW) = s1 * cell_area_weighted_PMF(s2)
       Sums to s1 over the dial.
```

## Key metrics

| Component | Metric | Value | Notes |
|---|---|---|---|
| Stage 2 — random split (R002) | AUC-ROC | 0.9283 | Brier 0.0562 |
| Stage 2 — temporal split (R009) | AUC-ROC | 0.9234 | drop 0.005 from random, PASS gate |
| Stage 1 — opportunity-restricted (R015, +/-24h) | AUC | 0.7249, AP 0.32, Brier 0.123 | 16.2% positive rate |
| Stage 1 — window sweep (R017) | logp_improvement at +/-6,12,24,48 h | +2.09, +1.99, +1.95, +1.96 nats | stable across choices |
| Combined product — 6 case studies (R016) | physics sanity | **6/6 PASS** | south Bz peak > north Bz, storm > quiet, lat ordering correct, midnight ~ 0 |
| Combined product — end-to-end on 500 real held-out crossings (R016) | median 2D peak distance | 3.32 deg | p90 8.30 deg |
| Combined product — true cell rank | top-10 / top-1% | 40.4% / 59.2% | of 1920 cells |
| Combined product — calibrated logp(true cell) | improvement over uniform | +1.91 nats | 6.7x random |
| Hemisphere strat (R002) | N vs S AUC | 0.93 vs 0.91 | single model OK |
| Shuffled-SW novelty gap (R003) | AUC-ROC / AUC-PR | 0.085 / 0.20 | spatial prior strong, SW signal real on PR |

## Case study results (R016, north hemisphere)

| # | SW state | s1 P | Combined peak P | Peak MLAT | Peak MLT |
|---|---|---|---|---|---|
| 1 | Strong south Bz (-10) | 0.098 | 0.0031 | 71 | 11.5 |
| 2 | Strong north Bz (+10) | 0.038 | 0.0017 | 80 | 11.5 |
| 3 | By+ (Bz=-3, By=+8) | 0.175 | 0.0079 | 74 | 11.5 |
| 4 | By- (Bz=-3, By=-8) | 0.175 | 0.0091 | 74 | 11.5 |
| 5 | Quiet | 0.000 | 0.0000 | — | — |
| 6 | Storm (Bz=-15, V=700) | 0.096 | 0.0032 | 71 | 12.0 |

Physics ordering passes 6/6: storm > quiet, south Bz > north Bz, south Bz at lower lat than north Bz (71 vs 80), By states top of distribution (consistent with established cusp magnetopause-reconnection physics where strong-shear IMF orientations produce broader/more frequent cusp), midnight near zero everywhere.

## Honest open issues for the paper

1. **Stage 1 target is "DMSP-detectable cusp crossing observation", not "pure cusp existence"**. AE<100 filter is inherited from Anderson 2024 / paper 1. Frame the claim accordingly; do not claim "P(cusp | SW)" without that caveat.
2. **Opportunity window proxy** (+/-24 h around any crossing) is a heuristic, not a TLE-based orbit availability mask. R017 sweep shows results are stable across +/-6/12/24/48 h, so the proxy is defensible. Reviewer-friendly defense: window-sensitivity table in supplementary.
3. **Per-MLT coverage limit**: nightside (MLT 0-4, 20-24) has zero training positives because of DMSP sun-synchronous geometry. Combined product correctly returns ~0 there. Plan: add a coverage/failure figure (R010 originally) showing supported vs unsupported dial sectors.
4. **Shuffled-SW gap is 0.085 on AUC-ROC** (target was 0.10). AUC-PR gap is 0.20 — the SW signal is real on the imbalanced metric. Frame as: "SW reshapes and reweights a strong spatial prior", not "SW dominates the forecast".
5. **Single-stage R002 produced backwards magnitudes** (storm 0.02, quiet 0.56) until the two-stage decomposition fixed it. This is the central narrative of the paper: the design choice was forced by what the data could and could not encode.

## What is ready to write

Per codex round 3:
- **Method**: two-stage architecture (decomposition rationale), stage 1 OMNI-hour occurrence model, stage 2 reused from paper 1 with spatial encoding, renormalization step
- **Validation**: temporal split (R009), end-to-end held-out evaluation (R014/R016), window sensitivity (R017)
- **Figures**: case studies (already in `src/kernels/cuspmap_mvp/figures/2stage_v2_*.png`), reliability diagrams (R002, R015), per-MLT coverage map (still TODO), AUC bar chart for shuffled-SW control
- **Discussion**: framing should emphasize that the SW-conditioned product is an observation-process forecast layered on a spatial prior, and that the spatial prior tracks DMSP sampling geometry. Honest framing wins over operational over-claim.

## Files (commit pending)

New code:
- `src/scripts/mvp/r009_temporal_split.py` (temporal split for stage 2)
- `src/scripts/mvp/r013_normalized_2stage.py` (renormalized combined, area-weighted)
- `src/scripts/mvp/r014_endtoend_eval.py` (held-out crossings logp + peak distance)
- `src/scripts/mvp/r015_stage1_opportunity.py` (opp-restricted stage 1)
- `src/scripts/mvp/r016_combined_v2.py` (final combined product)
- `src/scripts/mvp/r017_window_sweep.py` (+/-6/12/24/48 sensitivity)

New result files (bundles/):
- `r009_temporal_results.json`
- `r013_normalized_2stage.json`
- `r014_endtoend.json`
- `r015_stage1_opp_*.{ubj,pkl,json}`
- `r016_final_combined.json`
- `r017_window_sweep.json`

New figures (figures/):
- `2stage_renorm_case*.png` (R013)
- `2stage_v2_case*.png` (R016, final)

## Codex review log

3 rounds:
- Round 1: 7/10 Almost. Flagged (a) stage 1 target proxy, (b) no end-to-end validation, (c) renorm endpoint bias.
- Round 2: 8.5/10 Almost. After R009 + R014 + R013 v2 fixed (b) and (c). Single blocker remained: (a) stage 1 needs opportunity control.
- Round 3: 9/10 Ready. After R015 + R016 + R017 addressed (a) with proxy + sensitivity sweep. Pre-submission polish only: per-MLT coverage figure.

Full debate transcript saved in `AUTO_REVIEW.md` (this skill writes it automatically).

## Next step

Codex says ready. Recommended next ARIS skill: `/paper-plan` to draft the paper 2 outline from these results, then `/paper-figure` for publication-quality plots, then `/paper-write` for the LaTeX draft. Or stop here for the user to review the final state before committing to paper writing.
