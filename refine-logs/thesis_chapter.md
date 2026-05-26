# Thesis Chapter — 二维极隙概率图谱 (Cusp Probability Map)

**项目内部代号**: 二维极隙项目
**Date**: 2026-05-26
**状态**: 实验完成,prose draft 中

## 1. 章节定位与范围

本章在博士论文中接续 Paper 1 (Zhu et al. 2026 JGR:SP) 提交版的一维 cusp 边界回归。Paper 1 给出沿 DMSP 单 pass 的 equatorward MLAT 点估计 (MAE 1.11° on temporal holdout, 1.26° LOYO),误差跟 Newell 2006 线性 baseline 比降 40%。本章把同一份 DMSP 数据扩展到二维 **DMSP-可探测 cusp 概率图谱** `P(DMSP-detect cusp at (MLAT, MLT) in hour | SW)`,目标是给 operational space weather 一个全极区的实时 cusp 分布预报。注意 target 是 DMSP 观测意义上的 cusp,不是物理 cusp 本体 — 这点全章一贯。

本章不是论文,是博士论文里的一个研究 chapter。范围明确:
- 用同一份 48,056 DMSP 1987-2014 crossings
- 训二阶段 XGBoost(occurrence × spatial)
- 在数据支持区域给出量化误差,数据外区域明确为外推
- 物理解释 + 失效区域分析

不涵盖:
- 外部 instrument fusion (SuperDARN PCB / AMPERE / Polar UVI) — 留 future work
- 真负样本路径 (R020-R025 pilot 已证明在数据规模下不胜过合成 negatives)
- 实时部署 (需要 L1 上 1-min OMNI 流水线)

## 2. 方法

### 2.1 两阶段分解

单阶段二分类器把 (MLT, MLAT) 当 feature 给 stage 2 直接预测 P(DMSP-detect cusp at (MLAT, MLT) in hour | SW) 在 case study 出现 magnitude 反向:风暴时 peak P=0.02、quiet 时 peak P=0.56。诊断为训练集每 crossing 固定 1:10 正负比抹平了 P(any DMSP cusp obs | SW) 的全局信号。

两阶段拆分修复:

$$P(\text{DMSP-detect cusp at (MLAT, MLT) in hour} \mid \text{SW}) = \underbrace{P(\text{any DMSP cusp obs in hour} \mid \text{SW})}_{\text{stage 1}} \times \underbrace{P((\text{MLAT, MLT}) \mid \text{cusp obs, SW})}_{\text{stage 2}}$$

Stage 1 学时间维度上 SW 何时产生可探测 cusp;stage 2 学给定 SW 下 cusp 在极区表盘 (polar dial,简称表盘) 上的分布形状。两个分别在合适数据上训练,推理时相乘。

### 2.2 数据流

**Stage 2 训练数据 (空间分类)**
- 48,056 DMSP F06-F18 cusp crossings (1987-2014, Anderson 2024 criterion).  注意:本数据集**没有 AE<100 filter**(AE 实际分布 min 5 / median 187 / max 2421,70% > 100 nT)。Anderson 2024 原 paper 在他们的数据里加了 AE<100 filter,但我们继承的 cusp database 没加。后续 dropna 后 39,668 crossings 进训练
- 每 crossing 扩展为 5 真实 positive cells (在 `[eq_lat, pole_lat] × [eq_mlt, pole_mlt]` 内均匀采样) + 50 合成 negative cells (5 near-boundary + 5 dial-random per positive,同 SW state)
- 总扩展数据 ~2.18M rows
- 特征 76 维:74 SW (Paper 1 同套,含 15/30/60 分钟 mean/std/Δ history) + 2 维极坐标 `(x_polar, y_polar) = (90 - |MLAT|) · (cos, sin)(2π MLT / 24)`
- 极坐标编码让 XGBoost 在表盘上的几何切分对得上物理边界

**Stage 1 训练数据 (occurrence)**
- NASA OMNI2 hourly 1987-2014, 197,775 hours after dropna
- Opportunity 限制 (R015): 只保留 ±24 h 内有 DMSP crossing 的 hours (147,521 hours, 16.2% positive rate)
- 该限制排除了 DMSP 没在运行 / 在夜侧的小时,把 stage 1 negatives 限定为"DMSP 在场但 SW 状态没产生 cusp"
- 特征 13 维 SW (AE 排除。原 R015 设计 rationale 假设 48k 已被 AE<100 filter — 后来证实**没 filter**。当前选择保留 AE 排除以保持 stage 1 不依赖单一 geomagnetic activity proxy,而是从底层 SW 学 occurrence pattern。 注:AE-include 对照实验未跑,这是设计选择,不是 ablation 验证)

### 2.3 模型与推理

两阶段都是 XGBoost binary:logistic。Stage 2 复用 Paper 1 超参 (1000 trees, depth 8, lr 0.02, subsample 0.8, colsample 0.7, scale_pos_weight=10, early stopping on val logloss),isotonic 校准。Stage 1 用更轻 (600 trees, depth 6) + isotonic。

推理时:
1. 给定 SW state,stage 1 输出 scalar `s1`
2. Stage 2 在 1°×0.5h 表盘网格 (40×48=1920 cells) 上扫,每 cell 推 P
3. Stage 2 输出按 cell 面积 `cos(|MLAT|)` 加权后归一化为表盘上的 PMF
4. 合并: `combined(cell) = s1 × area_weighted_PMF_stage2(cell)`,total 表盘 = s1

合并图每 cell 的概率值意义:"在当前 SW 状态下,DMSP 风格 cusp footprint 覆盖这个 1°×0.5h cell 的概率"。不是物理 cusp 存在概率 — 是 DMSP-可探测 cusp 出现概率。这层 framing 在论文/本章里写清楚。

## 3. 验证

### 3.1 IID 验证 (Paper 1 同 split)

随机 8:1:1:0.2 split (按 crossing_id 分组,避免 within-crossing 扩展行泄漏):
- Stage 2 AUC-ROC: **0.9283**, Brier 0.0562
- Stage 1 AUC-ROC: **0.7249**, AP 0.32, Brier 0.123

### 3.2 时间泛化 (R009)

`train < 2008, test 2008-2014` (覆盖 solar minimum):
- Stage 2 AUC: **0.9234** (drop 0.005 vs random,远低于 0.05 gate)
- Brier 0.0579

### 3.3 5-block 时间 cross-validation (R030)

> **协议限制**(诚实声明):本节叫 5-block temporal CV,**不是严格 LOYO**:
> - 只 retrain stage 2,stage 1 (R015) 在所有 folds 间共享
> - 每 fold test 端到端 inference 仅采 500 真实 crossings(避免 ×27,000 推理时间)
> - 因此 fold-to-fold 方差被压缩:stage 1 不变 + 子采样统计 → 真实 LOYO 方差可能更大
> - 真严格 LOYO (23 折 + stage 1 也重训 + 全 test 推理) 约 100h wall,本章不做

5 折每折 train 所有年除该 5 年窗口外,test 在该窗口内 (n=500 sample per fold for end-to-end inference cost):

| Fold | 测试年 | n test crossings | median peak dist | p90 peak dist | mean true logp |
|---|---|---|---|---|---|
| 0 | 1990-1994 | 2689 | 4.80° | 10.28 | -6.53 |
| 1 | 1995-1999 | 2152 | 3.65° | 8.95 | -6.59 |
| 2 | 2000-2004 | 17404 | 3.81° | 8.57 | -6.03 |
| 3 | 2005-2009 | 9299 | 3.55° | 8.80 | -5.90 |
| 4 | 2010-2014 | 7969 | 3.28° | 7.67 | -5.78 |
| **mean ± std** | | | **3.82 ± 0.52°** | **8.86 ± 0.84** | **-6.17 ± 0.33** |

**观察**:
- Best fold = 2010-2014 (3.28°),worst = 1990-1994 (4.80°)。Train-on-recent-test-on-old 退化更多,原因 (a) 早期 DMSP F09-F12 era MLT 平面跟后期 F16-F18 不完全一致 (b) solar cycle 22 vs 23-24 SW 统计分布有差
- 跨 fold std 0.52° 是**协议压缩下的方差** (stage 1 不变 + 500 子采样),不能直接当 cross-solar-cycle robustness 证据。真严格 LOYO 方差大概率更大
- IID random-split (R028) 3.31°,5-block CV mean 3.82°,泛化代价 ~15%

这一节给的是"中等强度时间泛化证据",不是 thesis chapter 最强 validation。最强是 R009 temporal split (drop 0.005) + R028 全 7933 test。

### 3.4 端到端误差 (R028, 全 7933 holdout crossings)

每个 holdout crossing 经过完整推理 pipeline,与真实 (|MLAT|, MLT) 比较:
- Median peak distance: **3.31°**
- p90 peak distance: **8.04°**
- Mean true-cell log-probability: -5.68 (vs uniform -7.56,改善 +1.88 nats,约 6.5× random)
- 40.4% 真实 cell 在模型预测的 top-10 cells
- 59.2% 真实 cell 在 top 1% (前 19 cells of 1920)

### 3.5 分层验证

**Per-MLT bin** (median peak dist, n_test):
- MLT 8-12: **3.03°** (n=4661, 主覆盖区)
- MLT 12-16: **3.67°** (n=3272, 主覆盖区)
- MLT 0-4, 4-8, 16-20, 20-24: **0 crossings** (Anderson criterion 在这些 MLT 不触发 — DMSP 飞过但没 cusp 可识别)

**Per-|MLAT| bin** (median peak dist, n_test):
- 50-65°: 4.15° (n=7,几乎没数据)
- 65-75°: 4.34° (n=1817)
- 75-83°: **3.06°** (n=5985,best,主覆盖)
- 83-90°: 4.86° (n=124)

**Hemisphere**: N 3.35° (n=7149) vs S 3.19° (n=784)。S 实际略好,可能因 test 子样本偏。两半球都在同一 model 下 work,验证 hemi_code feature 足以处理对称性。

**IMF Bz sign**: N (3.36°) vs S (3.31°)。无显著差异 — 模型对南北 Bz 都能预测。

**AE level** (monotonic 退化):
- AE < 100: 2.90° (quiet)
- AE 100-300: 3.38°
- AE 300-500: 3.56°
- AE ≥ 500: **4.10°** (storm, +41% vs quiet)

**Storm flag** (`|Bz|≥10 OR V≥600 OR AE≥300`):
- 非 storm: 3.08° (n=4851)
- Storm: **3.76°** (n=3082)

风暴时误差升幅 ~20-40%,但 absolute level 仍远低于 Paper 1 linear baseline 的 1.83° (那是 1D MAE,投到 2D 大概 ~10°)。

### 3.6 覆盖密度 vs 误差相关性

逐 cell 算训练 crossing 密度 vs holdout 平均误差。Spearman ρ = **-0.578, p = 1.1×10⁻¹⁷, n=183 cells**。**强负相关,统计极显著**。模型在数据多的格点准确,数据少的格点退化。这是 thesis 失效分析的核心定量证据。

## 4. 物理解释

### 4.1 SHAP feature importance (R029)

Top 15 features by mean(|SHAP|) on stage 2 (5000 background samples):

| Rank | Feature | mean(|SHAP|) | Physics meaning |
|---|---|---|---|
| 1 | `x_polar` | 3.58 | 空间 — 模型知道 cusp 在表盘特定 (x, y) 范围 |
| 2 | `y_polar` | 1.17 | 空间 |
| 3 | **`newell_cf_mean60`** | 0.28 | **60 分钟 Newell 耦合函数 = 反映前 1 小时 reconnection rate 累积** |
| 4 | `imf_bz_mean15` | 0.16 | 近期 IMF Bz,触发 reconnection |
| 5 | **`dipole_tilt`** | 0.15 | **磁偶极倾角 — Newell 1989 季节性 control** |
| 6 | `imf_bz_mean30` | 0.15 | 中期 Bz |
| 7 | `newell_cf_int60` | 0.11 | 60 分钟 CF 积分 |
| 8 | `vBs_mean60` | 0.10 | 半波整流 vBs,同 CF 一类 |
| 9 | `vBs_int60` | 0.08 | vBs 积分 |
| 10 | `newell_cf` | 0.06 | 瞬时 CF |
| 11 | `imf_bz_mean60` | 0.06 | 60-min Bz |
| 12 | `doy` | 0.05 | 日序数 — 季节 |
| 13 | `hemi_code` | 0.05 | 半球 |
| 14 | `sw_pdyn` | 0.05 | 动压 |
| 15 | **`by_hemi`** | 0.04 | **IMF By × hemisphere sign — Cowley 1981 By 不对称** |

空间特征 (x, y) 占主导是正常的:给定 SW state,cusp 位置约束在表盘上一个相对窄的区域,model 必须先用空间找到大致位置。但 SW 特征里 60 分钟 Newell CF 第一、IMF Bz 15/30/60 分钟 mean 全在前 11,这跟 Paper 1 的发现一致:**磁层 reconfiguration 时间尺度 ~1 小时,过去 60 分钟的 reconnection 率比瞬时值预测力强**。

**注意 framing**:模型不是从原始 IMF/SW 自己"发现"了 Newell coupling function 的公式 — 我们已经把 `newell_cf`、`vBs`、`by_hemi` 等 engineered driver 当 features 喂进去 (Paper 1 同套 feature engineering)。SHAP 真正说明的是:**给定一组候选 features,模型优先 upweight 物理上预期的驱动量**。Dipole tilt 排第 5、by_hemi 排第 15,反映 Newell 1989 (tilt 控制 cusp equatorward shift) 和 Cowley 1981 (By 在两半球对 reconnection geometry 的对称破坏) — 模型 ranking 跟这些已知物理一致 (consistent with),不是从零 recovery。

### 4.2 60 分钟时间尺度的物理

为什么 60 分钟 Newell CF mean 主导而不是瞬时 Newell CF?Newell 2006 的 binned correlation 在 60-min averaging 时最优。物理解释:磁层 dayside reconnection 不是瞬时响应 — IMF 转向南后,需要约 30-60 分钟 reconfigure dayside magnetopause 拓扑、把新的 open field line 拖到极区。Cusp 位置反映的是已经累积的 open flux,不是当前瞬时输入。在我们给定的 15/30/60 分钟 history features 中,模型**优先选择** 60-min mean (rank 3, mean(|SHAP|) 0.28) 而非瞬时 Newell CF (rank 10, 0.06) 或更短窗口,这一选择跟 Cowley & Lockwood 1992 expansion-contraction 时间尺度一致。

### 4.3 Dipole tilt 和半球

`dipole_tilt` 第 5 重要。物理: 当磁偶极倾向太阳 (summer hemisphere),subsolar reconnection point 移向高纬,cusp 跟着 contract poleward;倾离太阳时 expand equatorward。Newell 1989 给斜率 ~-0.06°/°。

> **PDP 局限提醒**: R029 用的 "PDP" 是单 feature 在中位数 baseline 周围扫,不是 sklearn 标准 robust PDP (该法对所有 row 求 expectation 后再 vary 一个 feature)。单点 sweep 容易受 baseline 选择影响。所以"tilt PDP 斜率 ≈ -0.05°/°" 这种 quantitative match 当**定性 sanity check**,不当严格物理验证。要严格匹配 Newell 1989 数值得跑真 PDP 或 ICE plot,留 future work。

### 4.4 IMF By 的不对称

By > 0 在北半球把 dayside reconnection point 推向晨侧 (MLT < 12),南半球推向昏侧 (MLT > 12)。`by_hemi = imf_by × sgn(hemi)` 这个特征把对称破坏编进模型 — by_hemi > 0 总是把 cusp 推向晨侧,无论哪个半球。SHAP rank 15 看起来不算高,但它在 Bz 不强(主预测器之外)的情况下提供关键 MLT skew 信号。

## 5. 失效区域分析

### 5.1 数据支持区 vs 外推区

按 (MLT, |MLAT|) 1°×0.5h 网格分:
- **数据支持区**: MLT 5-19, |MLAT| 70-83° — 240/1920 cells (12.5%) 训练正样本 ≥ 5。这区内 holdout median peak dist 3.06° (per-MLAT 75-83 bin) 到 4.34° (per-MLAT 65-75 bin),AUC ~0.93。
- **极少支持区**: |MLAT| 65-70° 和 |MLAT| 83-86° — 训练样本零星,模型预测可信度低 (per-MLAT 83-90 bin 4.86° error)
- **零支持区**: MLT 0-4, MLT 20-24, |MLAT| 50-65°, |MLAT| > 86° — 训练数据完全为 0。模型在这些 cells 输出由两个先验决定: stage 2 训练时合成 negatives 暗示"非主表盘区都是 0",stage 1 SW signal 通过 occurrence rate 缩放整体幅度

### 5.2 为什么零支持区是 selection bias 不是 orbit bias

R026 + R027 分析回答了一个关键问题:DMSP 飞过这些 cells 没有?答:**飞过了,只是 Anderson cusp criterion 不触发**。R027 用 SGP4 模拟 DMSP F16/F17 + POES NOAA-15/18/19 + MetOp-A/B 一年轨道,在 |lat| 75-81 上 MLT 覆盖 5-19。所有 sun-synchronous 极轨卫星都到不了 MLT 0-4 / 20-24 的 cusp 纬度带,因为 cusp 纬度带本来就只在 dayside 存在 (cusp 物理上就是 dayside reconnection 产物)。

所以零支持区不是仪器问题,是 cusp 物理本身在那里不存在。模型外推为 0 是物理正确的方向,但 magnitude 没数据校准。

### 5.3 风暴时退化的原因

AE ≥ 500 时 median peak dist 4.10° (vs quiet 2.90°,+41%)。诊断:
1. 风暴时 cusp 向低纬推 (Bz 南向强,reconnection 强,oval expand)。但训练数据里 |MLAT| 65-75° 只有 1817 个 crossings (vs 75-83° 的 5985 个),所以低纬 cusp 训练支持稀疏
2. 风暴时 SW 1 小时内可能从 Bz=-5 跳到 Bz=+10 (R018/R019 在 2011-08-05 风暴 06:00 UT 那个 Bz +19.7 尖峰说明)。stage 2 的 60-min history features 假设过去 60 分钟连续平稳,在 IMF 快速变化时这个假设破
3. 风暴时 dayside reconnection 强度可能超出训练数据 SW 分布的 quasi-stationary 假设;stage 2 history features (60 分钟 mean) 在 IMF 快速变化时 mean 平滑掉关键瞬时,模型 effective input 偏 quiet-state

可改进路径(留 future work):
- 加 instantaneous IMF derivative features 让模型识别 SW 突变
- 单独训 "storm-time cusp" 模型用 AE > 300 的 storm-time 子集 (现有数据已足,见 section 3.5 AE level n=2660 for AE>=300)
- 在 stage 2 inference 时对 storm-time test sample 用 attention-style reweighting 强调最近 15 分钟 SW 而不是 60 分钟 mean

### 5.4 半球不对称

S hemisphere training 数据只有 N 的 10% (DMSP descending 半 pass 在 S 半球时数据采样不如 N 半球 dense)。但 S holdout median peak dist 3.19° 反而比 N 的 3.35° 略好 — 不是 model 在 S 上更准,是 S holdout 样本量小 (n=784) 的统计偶然 + S 半球 cusp 物理上对称 N。Hemisphere stratified SHAP (thesis_shap_hemi.png) 显示 top features ranking 在两半球一致,差异主要在 abs magnitude (S 上信号弱因数据少)。

## 6. 局限和 future work

1. **零支持区无 ground truth 校准**: MLT 0-4, MLT 20-24, |MLAT| < 65°, |MLAT| > 86° 这些 cells 的预测 P 是模型先验外推,不应作业务决策依据。Coverage mask figure (thesis_coverage_vs_error.png) 显式标记
2. **Stage 1 target 是观测 proxy 不是物理存在**: stage 1 训"DMSP-detectable cusp crossing 在 hour 内发生" 不是"物理 cusp 存在"。Future: 加专门的 storm-time augmentation + 与 SuperDARN PCB / Polar UVI cusp imaging 做 cross-validation
3. **Opportunity proxy 是 ±24h heuristic 不是 TLE 真 orbit availability**: R017 window sweep 在 ±6/12/24/48h 显示 combined logp 稳定 (+1.95 to +2.09 nats 范围 0.14),proxy 在敏感性上是 defensible。Future: SGP4-based 真 orbit availability mask
4. **没用 multi-instrument**: SuperDARN PCB、AMPERE FAC、Polar UVI cusp imaging 可以在零支持区给约束 (它们覆盖全极区)。需要 Globus / JHU/APL 账号 + 几周整合工作。本章不做,留 future paper / 后续 chapter
5. **真负样本路径已 explored 但 not adopted**: R020-R025 pilot 在 F10 1993-94 测试用 real 1Hz spectra negatives 替代合成 negatives。结论:real-only 严重退化 (median 38° vs 合成 2.4°),hybrid (real-near + synth-far) 接近但没赢 (5.6°)。原因:DMSP 实际覆盖只占表盘 ~12%,real negatives 学不到 dial 远端先验。合成 dial-random negatives 在当前数据量级仍是最优设计

## 7. 章节总结

二维 DMSP-可探测 cusp 概率图谱用两阶段 XGBoost (occurrence × spatial) 把 Paper 1 的一维边界点估计扩展到全极区表盘概率场。在 DMSP 数据支持区域 (MLT 5-19, |MLAT| 70-83) median 2D 误差 3.31°、true cell top 1% 命中 59.2%。SHAP 显示模型 ranking 跟已知物理一致 (consistent with),60 分钟 Newell CF + dipole tilt + by_hemi 都在 top 15,但不能宣称从原始 SW 自己 "discovered" 这些 — 这些是 engineered features 喂进去的。覆盖密度和误差强负相关 (Spearman ρ=-0.58, p=1.1e-17),失效集中在 storm-time 低纬和数据稀疏的高纬。零支持区 (MLT 0-4, 20-24 / |MLAT|<65, >86) 的概率值是模型先验外推,thesis 显式 mask。
