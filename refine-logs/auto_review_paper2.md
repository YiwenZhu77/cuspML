# Auto Review Loop — Cusp Probability Map MVP (Paper 2 follow-up)

**Started**: 2026-05-25
**Completed**: 2026-05-25 (same session, 3 rounds)
**Topic**: Two-stage P(cusp | MLAT, MLT, SW) — fix spatial calibration
**Reviewer**: GPT via Codex MCP (`model_reasoning_effort=xhigh`)
**Difficulty**: medium
**Thread**: 019e60ea-4809-7893-aee8-ad6907f88184
**Final score**: 9/10 Ready
**Final commit**: 32176da

(Note: this file is for paper 2 MVP review. Paper 1's prior auto-review log
lives at the gitignored `AUTO_REVIEW.md` in project root.)

## Round 1 — 14:50

### Assessment
- Score: 7/10
- Verdict: Almost
- Key criticisms:
  1. Stage 1 estimates observation-process proxy, not pure cusp occurrence
  2. No end-to-end / out-of-time validation yet
  3. Renormalization endpoint bias (MLT=24 duplicated, pole over-represented)

### Actions Taken (Round 1 → Round 2)
- R013 v2: cell-area weighting in renormalization, docstring reframed
- R009: stage 2 temporal split train<2008
- R014: end-to-end metric on 500 sampled real held-out crossings

### Results
- R013 v2: 6/6 sanity, max combined peak 0.00572
- R009: temporal AUC 0.9234 (drop 0.005 vs random) — PASS gate
- R014: median 3.32° peak distance, 59.2% top-1%, combined logp +1.63 nats

## Round 2 — 15:00

### Assessment
- Score: 8.5/10
- Verdict: Almost
- Single blocker: stage 1 needs opportunity-restricted negatives

### Actions Taken (Round 2 → Round 3)
- R015: opportunity-restricted stage 1 (+/-24h window around any crossing)
- R016: final combined product with R015 stage 1

### Results
- R015: AUC 0.7249 (vs R011 unrestricted 0.7285 — unchanged, signal not from data-gap), AUC-PR 0.32 (vs 0.25 improved), Brier 0.123
- R016: 6/6 sanity, combined logp +1.91 nats (up from +1.63)

## Round 3 — 15:10

### Assessment
- Score: 9/10
- Verdict: **READY** for paper writing

### Actions Taken (Closing)
- R017: window sensitivity sweep +/-{6, 12, 24, 48} h

### Results
| Window | n eligible | pos rate | AUC | AP | Brier | logp improvement |
|---|---|---|---|---|---|---|
| 6h | 107,662 | 22.22% | 0.6817 | 0.3599 | 0.1590 | +2.090 |
| 12h | 131,313 | 18.22% | 0.7086 | 0.3266 | 0.1358 | +1.993 |
| 24h | 147,521 | 16.21% | 0.7249 | 0.3157 | 0.1228 | +1.949 |
| 48h | 158,570 | 15.08% | 0.7373 | 0.3151 | 0.1150 | +1.956 |

Combined logp stable at +1.95 to +2.09 nats across all 4 windows. Proxy defensible.

### Status
- **STOPPED**: positive assessment threshold met
- Difficulty: medium

## Score progression

| Round | Score | Verdict | Key delta |
|---|---|---|---|
| 1 | 7 | Almost | initial PoC reviewed |
| 2 | 8.5 | Almost | renorm fixed, temporal + end-to-end added |
| 3 | 9 | **Ready** | opportunity-restricted stage 1 + window sweep |

## Method Description (for /paper-illustration handoff)

The cusp probability map MVP is a **two-stage XGBoost product** that predicts the calibrated joint probability that a DMSP cusp crossing observation will fall in a given (MLAT, MLT) cell on the polar dial under given solar wind state. The two stages decompose the problem along an explicit causal split: stage 1 estimates **whether** a cusp crossing will be observed in a given hour from the upstream SW state, and stage 2 estimates **where on the dial** it will be observed if one is.

Stage 1 is an XGBoost binary classifier trained on 147,521 OMNI2 hourly samples (1987-2014), restricted to "opportunity hours" within +/-24 h of any crossing in the existing 48k DMSP crossing table. Positives are hours that contain at least one crossing (16.2%); negatives are opportunity hours that do not. Features are 13 base + derived SW quantities (IMF Bx/By/Bz, V, n, Pdyn, B_T, clock angle, Newell coupling, Kan-Lee, vBs, day-of-year), with AE excluded to avoid leaking the Anderson 2024 AE<100 selection filter. The model is isotonic-calibrated and reaches test AUC 0.7249.

Stage 2 is the cuspML paper-1 binary classifier reused unchanged. It is trained on 2.18M rows expanded from the 39,668 quality-filtered crossings (5 positives per crossing inside the (eq_lat, pole_lat) x (eq_mlt, pole_mlt) window, 50 stratified-near + stratified-far synthetic negatives per crossing at the same SW state), with 74 SW features plus polar Cartesian spatial encoding `(x_polar, y_polar) = (90 - |MLAT|) * (cos, sin)(2 pi MLT / 24)`, paper-1 hyperparameters, scale_pos_weight = 10, and isotonic calibration. Random crossing-level holdout gives AUC 0.9283; temporal split (train<2008, test>=2008) gives AUC 0.9234 with drop 0.005.

At inference, given an SW state, the combined map is `combined(MLAT, MLT, SW) = stage1(SW) * area_weighted_PMF(stage2)`, where the stage 2 grid is renormalized over the polar dial with `cos(|MLAT|)` cell-area weights so the resulting map sums to `stage1(SW)` in solid-angle-weighted integral over the dial. The combined product is interpreted as "probability cusp footprint covers this 1deg x 0.5h cell during this hour given current SW".

End-to-end evaluation on 500 sampled real held-out crossings (same 20% test split as stage 2) yields median 2D peak-distance error of 3.32 deg, 59.2% of true cells inside the top-1% of map cells, and mean log-probability of the true cell of -5.65, a +1.91 nat improvement over uniform.

## Open items before submission (not blockers)

- per-MLT coverage figure showing nightside (MLT 0-4, 20-24) is data-unsupported and combined product returns ~0 there honestly
- (optional) TLE/SGP4-based orbit availability mask to replace the +/-24 h opportunity proxy
- paper framing: "SW-conditioned probability map of DMSP-observed cusp crossing occurrence and spatial occupancy" rather than "P(cusp existence)"
- novelty narrative: SW reshapes/reweights a strong spatial prior; do not claim SW dominates
