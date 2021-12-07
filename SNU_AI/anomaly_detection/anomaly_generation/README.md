# Anomaly generation
This is a resource for generating outliers from original data. 
## Usage
---
Generate anomalies for the test data. `sample.csv` and `sample_figure.png` are created.

```bash
python univariate.py
```


Generate anomalies for custom data (Incomplete).
```bash
python run_custom.py
```

## TODO:
- tweak collective_seasonal_outliers for custom data (freq is not defined)

## Acknowledgement
---
The codes are based on the following paper:   
https://arxiv.org/abs/2009.09822