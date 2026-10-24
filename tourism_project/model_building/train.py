"""Train + tune an XGBoost model, track everything in MLflow, save the best model for deployment."""
import os
import joblib
import mlflow
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

MODEL_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(MODEL_DIR, "best_tourism_model_v1.joblib")
os.makedirs(MODEL_DIR, exist_ok=True)

# The workflow starts an MLflow server on port 5000 before this script runs
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("tourism-wellness-package")

# --- Load the splits produced by the data-prep job ---
Xtrain, Xtest = pd.read_csv("Xtrain.csv"), pd.read_csv("Xtest.csv")
ytrain, ytest = pd.read_csv("ytrain.csv").squeeze(), pd.read_csv("ytest.csv").squeeze()

num_cols = Xtrain.select_dtypes("number").columns.tolist()
cat_cols = Xtrain.select_dtypes(exclude="number").columns.tolist()

preprocessor = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
    (make_pipeline(SimpleImputer(strategy="most_frequent"),
                   OneHotEncoder(handle_unknown="ignore")), cat_cols),
)

# --- Model + hyperparameter grid ---
class_ratio = (ytrain == 0).sum() / (ytrain == 1).sum()   # handles the class imbalance
xgb = XGBClassifier(scale_pos_weight=class_ratio, eval_metric="logloss", random_state=42)
pipeline = make_pipeline(preprocessor, xgb)

param_grid = {
    "xgbclassifier__n_estimators": [100, 300],
    "xgbclassifier__max_depth": [3, 6],
    "xgbclassifier__learning_rate": [0.1, 0.3],
}

with mlflow.start_run(run_name="xgboost_grid_search"):
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1", n_jobs=-1)
    grid.fit(Xtrain, ytrain)

    # Log every parameter combination as a nested run
    results = grid.cv_results_
    for i, params in enumerate(results["params"]):
        with mlflow.start_run(nested=True, run_name=f"combo_{i}"):
            mlflow.log_params({k.split("__")[-1]: v for k, v in params.items()})
            mlflow.log_metric("mean_cv_f1", results["mean_test_score"][i])
            mlflow.log_metric("std_cv_f1", results["std_test_score"][i])

    best_model = grid.best_estimator_
    mlflow.log_params({f"best_{k.split('__')[-1]}": v for k, v in grid.best_params_.items()})
    mlflow.log_metric("best_cv_f1", grid.best_score_)

    # --- Evaluate the best model ---
    for split, X_, y_ in [("train", Xtrain, ytrain), ("test", Xtest, ytest)]:
        pred = best_model.predict(X_)
        proba = best_model.predict_proba(X_)[:, 1]
        mlflow.log_metrics({
            f"{split}_accuracy": accuracy_score(y_, pred),
            f"{split}_precision": precision_score(y_, pred),
            f"{split}_recall": recall_score(y_, pred),
            f"{split}_f1": f1_score(y_, pred),
            f"{split}_roc_auc": roc_auc_score(y_, proba),
        })

    print("Best parameters:", grid.best_params_)
    print("\nTest set classification report:")
    print(classification_report(ytest, best_model.predict(Xtest), target_names=["No purchase", "Purchase"]))

# --- Save the model into the deployment folder; the workflow commits it to the repo ---
joblib.dump(best_model, MODEL_PATH)
print("Model saved to", MODEL_PATH)
