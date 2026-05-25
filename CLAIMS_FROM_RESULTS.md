# Claims from MVP Results — Paper 2 (Cusp Probability Map)

**Date**: 2026-05-25
**Source data**: `refine-logs/results_final.md`, `refine-logs/auto_review_paper2.md`
**Judgment**: Codex MCP via auto-review-loop thread `019e60ea-4809-7893-aee8-ad6907f88184`, round 4 (claim eval)
**Status**: 4 fully supported claims, 4 partially supported (with concrete pre-submission additions)
**Next step**: hand to `/paper-plan` for paper outline

---

## C1. Two-stage decomposition fixes the magnitude-inversion failure of the single-stage classifier

- **Supported**: yes
- **Confidence**: high
- **Evidence**: Single-stage R002 case studies had quiet peak P=0.559 and storm/south-Bz peak P=0.019 (physically inverted). Two-stage R016: quiet 0.000, storm 0.0032, south-Bz 0.0031 (correct ordering).
  - Source files: `bundles/r002_case_studies.json`, `bundles/r016_final_combined.json`
- **What it does not show**: Two-stage isn't the only possible fix; the final combined magnitudes are not globally calibrated probabilities (per-cell PMF over 1920 cells).
- **Suggested paper phrasing**: "A direct single-stage classifier produced physically inverted map magnitudes in synthetic test cases; separating occurrence from spatial shape removed that failure mode."
- **Action items**: none

## C2. Stage 2 spatial classifier generalizes across solar cycles

- **Supported**: yes
- **Confidence**: high
- **Evidence**: Random split AUC 0.9283 / Brier 0.0562. Temporal split (train<2008, test 2008-2014 incl. solar min) AUC 0.9234 / Brier 0.0579. AUC drop 0.0049.
  - Source files: `bundles/r002_baseline_results.json`, `bundles/r009_temporal_results.json`
- **What it does not show**: One temporal split, not a full year-by-year stability scan; Southern Hemisphere temporal performance is weaker and data-limited.
- **Suggested paper phrasing**: "The spatial classifier retained nearly identical performance under a train-before-2008, test 2008-2014 split spanning solar-minimum conditions, indicating strong temporal robustness."
- **Action items**: optional leave-one-year-out scan for supplementary

## C3. SW-conditioned cusp-observation probability can be learned from upstream OMNI data

- **Supported**: partial
- **Confidence**: medium
- **Evidence**: Stage 1 with opportunity-restricted negatives AUC 0.7249, AP 0.32, Brier 0.123. Window sensitivity (+/-{6, 12, 24, 48} h) stable at AUC 0.68-0.74 and combined logp improvement +1.95 to +2.09 nats over uniform.
  - Source files: `bundles/r015_stage1_opp_results.json`, `bundles/r017_window_sweep.json`
- **What it does not show**: Target is "DMSP-detectable cusp crossing occurrence" filtered by AE<100, not pure physical cusp existence. The opportunity proxy is a +/-24h heuristic, not TLE-based orbit availability.
- **Suggested paper phrasing**: "A solar-wind-only stage-1 model can learn the probability of a DMSP-detectable cusp crossing within an opportunity-restricted hour from upstream OMNI data."
- **Action items**: include the window-sensitivity table in supplementary; state explicitly in methods that stage 1 predicts observed crossing occurrence, not physical cusp existence

## C4. Combined product gives meaningful spatial localization on real held-out crossings

- **Supported**: yes
- **Confidence**: medium
- **Evidence**: 500 held-out real crossings (test split same as R002): median 2D peak distance 3.32 deg, p90 8.30 deg, true cell in top-1% of 1920-cell map 59.2%, combined logp(true cell) improvement over uniform +1.91 nats.
  - Source files: `bundles/r014_endtoend.json`, `bundles/r016_final_combined.json`
- **What it does not show**: Conditional-on-positive-crossings evaluation only. Stage-2-only remains better on true-cell logp and argmax distance.
- **Suggested paper phrasing**: "On held-out real crossings, the combined map assigned substantial probability mass near the observed crossing location, with median localization error of 3.3 deg."
- **Action items**: add one unconditional hourly forecast eval on opportunity-restricted positive AND negative hours before submission

## C5. Combined product reproduces established cusp-physics ordering

- **Supported**: yes
- **Confidence**: medium
- **Evidence**: 6/6 synthetic case sanity checks: storm > quiet, south Bz > north Bz, south Bz at lower lat (71 deg) than north Bz (80 deg), midnight near zero, By states top of distribution.
  - Source file: `bundles/r016_final_combined.json`
- **What it does not show**: Qualitative sanity checks on synthetic SW states with simplified history features. Not quantitative validation against an independent physical truth source.
- **Suggested paper phrasing**: "Synthetic driver states produced probability maps qualitatively consistent with established cusp behavior, including equatorward displacement under southward IMF and suppression under quiet conditions."
- **Action items**: none

## C6. Single XGBoost handles both hemispheres

- **Supported**: partial
- **Confidence**: medium
- **Evidence**: Random split N AUC 0.9302 vs S AUC 0.9127, difference 0.018. Temporal split shows S AUC dropping to 0.8713 on limited test rows.
  - Source files: `bundles/r002_baseline_results.json`, `bundles/r009_temporal_results.json`
- **What it does not show**: Southern Hemisphere data support is much thinner; the small AUC gap masks a larger generalization gap.
- **Suggested paper phrasing**: "A single hemispherically pooled model was adequate for the MVP, although Southern Hemisphere validation remains data-limited."
- **Action items**: pooled vs hemisphere-specific comparison under temporal or LOYO eval for supplementary

## C7. SW signal contributes real predictive skill beyond spatial prior

- **Supported**: partial
- **Confidence**: medium
- **Evidence**: Shuffled-SW control (R003) reduced stage-2 AUC from 0.9283 to 0.8431 and AP from 0.5090 to 0.3051.
  - Source file: `bundles/r003_shuffled_sw_results.json`
- **What it does not show**: ROC gap 0.085 missed the pre-set 0.10 target. Spatial prior (x_polar, y_polar, hemi_code) alone is very strong. Control was run on stage 2, not the end-to-end product.
- **Suggested paper phrasing**: "Solar-wind features provide measurable incremental skill beyond the strong dayside spatial prior, with the largest gain evident in precision-recall rather than ROC alone."
- **Action items**: add end-to-end shuffled-SW control on true-cell logp, not just stage-2 ROC

## C8. Product honestly identifies unsupported MLT sectors

- **Supported**: partial
- **Confidence**: medium
- **Evidence**: Per-MLT bin AUC: only 8-12 (0.88) and 12-16 (0.90) have sufficient positives. 0-4, 4-8, 16-20, 20-24 have 0 positives in training; combined product correctly returns ~0 there.
  - Source file: `bundles/r002_baseline_results.json` (per_mlt_auc field)
- **What it does not show**: No explicit support mask in figures yet. Low probabilities in unsupported sectors can be visually misread as physical absence rather than data unavailability.
- **Suggested paper phrasing**: "Training and evaluation support is confined to the well-sampled dayside MLT sectors; nightside and shoulder sectors should be treated as unsupported rather than physically excluded."
- **Action items**: coverage/failure figure (originally R010); add explicit support-mask overlay in main case-study figures

---

## Summary

| Claim | Supported | Confidence | Action items before submission |
|---|---|---|---|
| C1 single-stage failure mode fix | yes | high | none |
| C2 temporal generalization | yes | high | optional LOYO |
| C3 stage 1 SW->occurrence learnable | partial | medium | clarify target framing, table in supp |
| C4 spatial localization on real crossings | yes | medium | unconditional hourly forecast eval |
| C5 physics ordering | yes | medium | none |
| C6 single model for both hemispheres | partial | medium | hemisphere-specific comparison |
| C7 SW signal beyond spatial prior | partial | medium | end-to-end shuffled-SW control |
| C8 unsupported MLT sectors | partial | medium | coverage figure + support mask |

## Paper narrative implications

- **Headline**: SW-conditioned probability map of DMSP-observed cusp crossing occurrence and spatial occupancy. NOT "P(cusp existence)".
- **Method central contribution**: explicit two-stage decomposition (occurrence x conditional spatial PMF) with cell-area-weighted renormalization, derived from a calibrated binary classifier per stage.
- **Empirical headline**: stage 2 temporal-stable AUC 0.92, combined logp +1.9 nats over uniform on real held-out crossings, 6/6 physics sanity.
- **Honest limitations**: opportunity-proxy negatives (not TLE), AE<100 filter inherited from paper 1, sun-synchronous DMSP MLT coverage, S hemisphere data-thin.
- **Novelty framing**: "SW reshapes and reweights a strong spatial prior" — do not claim SW dominates. Lean on AUC-PR improvements (+0.20 from shuffled-SW) more than AUC-ROC (+0.085).

## Pre-submission residual checklist (4 items, all small)

1. Window-sensitivity table to supplementary (already have data in `bundles/r017_window_sweep.json`)
2. Unconditional hourly forecast eval on positive + negative opportunity-restricted hours (~1 hour of new code, reuses R015 stage 1 and R016 inference)
3. Hemisphere-specific stage 2 comparison under temporal eval (~30 min: rerun R009 separately on N and S subsets)
4. Coverage / failure figure showing per-MLT and per-IMF-Bz training support (~1 hour: histogram from existing 48k table)

None block writing the paper outline. They are figures and tables to add as paper-writing proceeds.

## Ready to invoke

`/paper-plan` — input: this file + `refine-logs/results_final.md`. Output: paper 2 outline with abstract, sections, figure list, claim-to-section map.
