# Abstract Draft — cuspML paper

## 核心 story

1. **第一个 ML cusp 预测模型**：过去 20 年都是 1-2 参数线性回归，我们第一次用 74 特征 + ML
2. **物理发现**：模型自动发现 60 分钟时间尺度的主导地位（与磁层对流周期理论一致）
3. **性能归因**：baseline ladder 量化了「更多特征」vs「非线性建模」各自贡献多少
4. **可操作性**：所有输入来自上游 L1 太阳风监测，可实时化

---

## New Abstract

The ionospheric cusp is where solar wind plasma directly enters the magnetosphere, and its latitude serves as a real-time proxy for the state of solar wind–magnetosphere coupling. For two decades, cusp latitude prediction has relied on linear regressions against one or two solar wind coupling functions, capturing only about half the crossing-to-crossing variability. Here we show that a gradient-boosted tree model (XGBoost), trained on ~40,000 DMSP cusp crossings spanning 27 years and 74 solar wind/IMF features, reduces prediction error by nearly half compared to the best existing approach—and that this improvement persists under strict temporal holdout, ruling out temporal leakage. The model independently recovers a key physical result: cusp latitude is governed not by the instantaneous solar wind, but by the reconnection rate integrated over the preceding ~60 minutes, consistent with the magnetospheric convection timescale. A structured baseline ladder (linear → Ridge → gradient boosting → tuned XGBoost) reveals that roughly equal contributions come from richer feature engineering and from nonlinear modeling, quantifying for the first time how much predictive information the solar wind carries about cusp position beyond what a single coupling function can extract. Because all inputs are available from upstream L1 monitors, these results establish a pathway toward real-time, data-driven cusp boundary prediction.

---

## 对比旧摘要的改进

- 去掉了 MAE=1.11°, r=0.886, 0.95°, 0.77hr, 33% 等细节数字堆砌
- 只保留一个核心数字："reduces prediction error by nearly half"
- 突出物理发现（60 分钟时间尺度）而不是罗列性能
- 加入 baseline ladder 作为方法论贡献
- 结尾指向 operational potential 而非简单总结
