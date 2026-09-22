# cuspML

Machine learning prediction of ionospheric cusp precipitation boundaries from solar-wind measurements.

Code for Zhu, Y., Michael, A. T., & Toffoletto, F. R., *Predicting Ionospheric Cusp Location from Solar Wind: An XGBoost Model Trained on 27 Years of DMSP Data*, Journal of Geophysical Research: Space Physics, accepted. DOI: [10.1029/2026JA035567](https://doi.org/10.1029/2026JA035567).

## Reproduce the revised manuscript

The current entry point uses the fixed temporal split: 29,935 crossings before 2008 for training and 9,733 crossings during 2008 through 2014 for testing. The main target is absolute equatorward boundary magnetic latitude. Three other targets are reported in the Supporting Information.

The **current processed inputs, primary temporal model, reference predictions and ordered feature manifest are included in this repository**, under [`reproduce/current/data`](reproduce/current/data). No external data download is required for this reproduction.

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r reproduce/current/requirements.txt
# Verify checksums, predictions, SHAP and PDP, then render all 10 figures:
python reproduce/reproduce.py --output-dir results/check
# Independently train all models and controls, verify, then render:
python reproduce/reproduce.py --train --jobs 6 --output-dir results/retrained
```

Use a new output directory for each run. See the [complete reproduction instructions](reproduce/README.md) for the figure map, baseline calibration, history experiments, data definitions and verification scope.

The primary temporal MAE is **1.109533 degrees**. The Newell (2006) coupling-function baseline gives **1.748628 degrees** on the same test crossings, corresponding to **36.5484% lower MAE**. The baseline is a local training-period calibration of the published function, not a comparison with a performance number reported by Newell et al.

## Historical archive

The [Zenodo data archive](https://doi.org/10.5281/zenodo.19340792) contains the earlier crossing catalog and random-split model package. Those files are **not interchangeable with the current temporal snapshot**. Historical code remains available in the repository history, including commit `827c2c0`. The current branch contains only the processed-data reproduction workflow.

The current reproduction starts from the released processed crossing table. It does not rerun raw satellite label identification or claim that the historical upstream pipeline has been independently revalidated. Earlier upstream and exploratory scripts are available in the repository history.

## License

Code: [MIT](LICENSE). Released processed data and model weights: CC BY 4.0, with attribution to this study and the upstream DMSP/OMNI data providers.
