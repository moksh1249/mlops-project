"""MP112 - EV charging session energy (Bhopal Green Charge Network).

Modules
-------
config       read params.yaml, paths, data hash
data_checks  detect the seven kinds of data problem in train.csv
cleaning     turn raw train.csv into a clean training table
features     feature engineering + the scikit-learn pipeline
train        train v1 candidates with MLflow, register the best as champion
monitoring   PSI drift, per-group MAE, retrain v2, promotion rule
predict      predict the holdout file with the @champion model
findings     write reports/findings.yaml
"""

__version__ = "1.0.0"
