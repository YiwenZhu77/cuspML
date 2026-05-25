# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | sanity: pipeline + 1k overfit | spec default, n_crossings=1k | crossing-random 8:2 | AUC | MUST | TODO | confirms column names, expansion correctness, no NaN; expect AUC > 0.99 |
| R002 | M1 | B1 baseline | spec default | crossing-random 8:1:1 + 20% test | AUC, PR-AUC, Brier, reliability, per-MLT-bin AUC, hemi strat | MUST | TODO | first real run; passes gate if AUC >= 0.85 |
| R003 | M2 | B2 shuffled-SW control | SW columns row-shuffled, rest identical | same split as R002 | AUC vs R002 | MUST | TODO | gate: real_AUC - shuffled_AUC >= 0.10 |
| R004 | M3 | B3 K-sweep K=5 | K=5 neg/pos, polar enc | random | AUC, Brier | MUST | TODO | ablation |
| R005 | M3 | B3 K-sweep K=20 | K=20 neg/pos, polar enc | random | AUC, Brier | MUST | TODO | ablation; K=10 covered by R002 |
| R006 | M3 | B3 enc raw | raw (MLAT, MLT), K=10 | random | AUC, Brier | MUST | TODO | ablation |
| R007 | M3 | B3 enc cyclic | sin/cos MLT + MLAT, K=10 | random | AUC, Brier | MUST | TODO | ablation; polar covered by R002 |
| R008 | M3 | B3 case-study redo with winner | best K + encoding from R002, R004-R007 | random | 6 case study heatmaps + AUC | MUST | TODO | only if winner differs from default by > 0.02 AUC |
| R009 | M4 | B4 temporal split | spec default | temporal: train < 2008, test >= 2008 | AUC, Brier, reliability, 6 case studies | MUST | TODO | gate: AUC drop from R002 <= 0.05 |
| R010 | M5 | B5 coverage + failure plot | descriptive, reuses R002 model | N/A | sample density per (MLT, MLAT, Bz) bin + per-bin AUC | NICE | TODO | for paper, not MVP gate |
