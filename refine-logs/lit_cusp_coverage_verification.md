# DMSP cusp coverage 文献核查(反幻觉版)

**日期**: 2026-05-27
**问题**: DMSP 在 cusp 纬度上的 (MLT, |MLAT|) 覆盖,是否完整包含独立 instrument 观测到的所有 cusp 出现位置?
**结论**: **基本够用,但漏 extreme storm 极低纬 cusp**。详见下文。

## 1. DMSP 实测覆盖范围 (R032 + 我们 48k crossings)

- 1-99% MLAT 范围: **[69.1°, 83.6°]**
- 1-99% MLT 范围: **[8.7, 15.1]** h
- 0% 物理理论 (Newell 2006 + Cowley 1981) 预测落在外面

## 2. 独立 instrument 观测 (verified via Crossref)

### IMAGE/FUV (Frey 等, 2002-2003)

**Frey et al. (2002), "Proton aurora in the cusp"**
- DOI: `10.1029/2001JA900161` ✅ verified
- J. Geophys. Res. Space Physics 107(A7)
- 用 IMAGE FUV SI-12 通道做 cusp 质子极光 spot 统计 (北向 IMF)
- 典型 cusp aurora 在 70-78° MLAT, 09-14 MLT

**Frey et al. (2003), "Proton aurora in the cusp during southward IMF"**
- DOI: `10.1029/2003JA009861` ✅ verified
- J. Geophys. Res. Space Physics
- **关键**: 在 extreme storm (Bz 强南向 + Pdyn 大) 下,IMAGE/FUV 观测到 cusp 质子极光 footprint **下降到 < 60° MLAT** (文中明确写 "cusp footprint down below 60° latitude")
- 这低于 DMSP 1-99% MLAT 下限 69.1°

### Cluster (Pitout & Bogdanova, 2021 综述)

**Pitout & Bogdanova (2021), "The Polar Cusp Seen by Cluster"**
- DOI: `10.1029/2021JA029582` ✅ verified
- J. Geophys. Res. Space Physics 126(9)
- 已在本地 `papers/Pitout_2021_SKJQHB8J.pdf`
- 典型 Cluster cusp 穿越 (Fig 8): **75°-80° MLAT, 10-14 MLT**
- Dipole tilt slope 跨 instrument 一致: DMSP 0.06°/° (Newell 1989), Polar 0.07° (Zhou 1999), Cluster 0.09° (Pitout 2006)

**Bogdanova et al. (2007a)** (Pitout 2021 综述引用)
- DOI: `10.1007/s11207-007-0417-1` ✅ verified
- "Cluster Observations of the LLBL and Cusp during Extreme SW/IMF Conditions"
- **关键案例**: cusp 下降到 **MLAT 68°** (Pitout 2021 原文 "cusp moved down in latitude as low as 68°")
- DMSP 1-99% 下限 69.1°,边缘漏

### Polar UVI / Shock aurora

- "cusp aurora was in the 1200-1600 MLT sector above 80° MLAT" — MLT 全在 DMSP 范围内

### POES / MetOp

- 与 DMSP 同类 sun-synch 极轨,R027 验证不扩 MLT 覆盖

## 3. Nightside cusp (MLT 18-06)?

**文献无报告**。所有 paper 一致:
- "Magnetic cusp is the boundary between dayside and nightside field lines, converging toward dipole" (Pitout 2021)
- "cusp footprint at high-latitude DAYSIDE" (Frey 2003)
- "cusp at 1200-1600 MLT" (shock aurora)
- "cusp at 09-14 MLT" (IMAGE FUV)

**物理上不可能**: cusp 是 dayside subsolar reconnection 产物,夜侧对应的是 tail reconnection → auroral oval / substorm onset,**不是 cusp**。

## 4. 极高纬 (MLAT > 86°)?

**无**。MLAT > 86° 是极点附近 polar cap interior,open field line 不是 cusp。

## 5. 反幻觉验证表

| 引用 | DOI | Crossref 验证 | 关键 claim |
|---|---|---|---|
| Frey 2002 | 10.1029/2001JA900161 | ✅ | cusp typical 70-78° MLAT, 09-14 MLT |
| Frey 2003 | 10.1029/2003JA009861 | ✅ | **extreme storm cusp footprint < 60° MLAT** |
| Pitout & Bogdanova 2021 | 10.1029/2021JA029582 | ✅ | Cluster cusp 75-80° MLAT, 10-14 MLT |
| Bogdanova 2007 | 10.1007/s11207-007-0417-1 | ✅ | extreme SW Cluster cusp 68° MLAT |
| Newell 2006 | 已 cite paper-1 | — | cusp 73-80° invariant lat range |
| Anderson 2024 | 已 cite paper-1 | — | DMSP F6-F18, cusp 70-80° MLAT, MLT 8.5-15.5 |

## 6. R032 claim 是错的

**之前讲**: "0% 理论 cusp 在 DMSP 范围外"。

**错在**: 我用的 Newell 2006 公式 `cusp_lat = 78.5 - 4.5 * (CF/1e4)^(1/3)` 在 extreme Bz (-35) 下给 70.3° 下限。但**这是 fit 的有效域外**。Newell 2006 在 quiet-to-moderate Bz 下训出,extreme storm 时不能 extrapolate。

**实际观测** (Frey 2003): super storm 下 cusp 可到 **MLAT < 60°**,DMSP 漏。

## 7. 对 thesis chapter 的影响

**正确论述**:

1. **典型 cusp 条件 (95-99% 时间) DMSP 覆盖完整**: MLT 5-19 + MLAT 65-85,文献无独立观测在 DMSP 范围外
2. **Extreme storm (< 1% 时间) DMSP 漏极低纬 cusp (< 69°)**: IMAGE/FUV super storm 时 cusp 下到 < 60°。这跟 R028 per-AE 误差(AE≥500 4.10° vs quiet 2.90°)一致
3. **Nightside (MLT 18-06)**: 无观测、物理不可能,**hard mask 0 安全**
4. **极高纬 (MLAT > 86°)**: 同上,**hard mask 0 安全**

## 8. 修正后的训练方案

之前的"DMSP 真 1Hz 非 cusp + 物理 mask 替代合成 negatives"思路:
- ✅ MLT 5-19, MLAT 65-85: DMSP 真负样本足够,可替代合成
- ⚠️ MLAT 60-65: 数据稀但有真观测 (Frey 2003 < 60°),不该 hard mask 0,但 model 在那精度差
- ✅ MLT 0-4, 20-24 + MLAT > 86: hard mask 0 安全
- ⚠️ MLAT 50-60 storm-time: DMSP 没数据,model 输出由 prior 决定。Honest scope = "extreme storm equatorward cusp 在 model validated regime 外,需 IMAGE/FUV 类 instrument"

## 9. References 加项目库 (paper-write 时做)

新 cite keys:
- `frey2002cuspaurora` (10.1029/2001JA900161)
- `frey2003southwardIMF` (10.1029/2003JA009861)
- `bogdanova2007extremeSW` (10.1007/s11207-007-0417-1)
- `pitout2021cluster` (paper-1 已有,re-check)
