# Model_Development/ml_src/utils/mlflow_models.py
"""One way to log scikit-learn-style models to MLflow.

MLflow 3 saves sklearn models with skops, which refuses classes it doesn't know.
LightGBM's classifier isn't on its list, so model_train failed when logging it.
Keep skops (it doesn't execute arbitrary code on load, unlike pickle) and trust
only the LightGBM classes this project trains."""

import mlflow.sklearn

TRUSTED_TYPES = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
]


def log_sklearn_model(model, name, **kwargs):
    return mlflow.sklearn.log_model(model, name=name, serialization_format="skops",
                                    skops_trusted_types=TRUSTED_TYPES, **kwargs)
