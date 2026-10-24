# Visit with Us – Wellness Tourism Package Prediction (MLOps)

Predicts whether a customer will purchase the Wellness Tourism Package *before* they are contacted, so the sales team can prioritise the right leads.

## Project structure
```
.github/workflows/pipeline.yml        CI/CD pipeline (GitHub Actions)
tourism_project/
├── data/tourism.csv                  raw dataset
├── model_building/
│   ├── data_register.py              validates the dataset (expected columns, summary)
│   ├── prep.py                       cleaning + stratified train/test split
│   └── train.py                      XGBoost tuning, MLflow tracking, saves the best model
├── deployment/
│   ├── app.py                        Streamlit app
│   ├── requirements.txt              app dependencies
│   └── best_tourism_model_v1.joblib  trained model (committed by the pipeline)
└── requirements.txt                  pipeline dependencies
```

## Pipeline (runs on every push to `main`)
1. **Register Dataset** – validate columns and print a summary.
2. **Data Preparation** – clean the data, split into train/test, pass the splits on as an artifact.
3. **Model Training** – tune XGBoost with every trial logged to MLflow, evaluate, commit the best model back to the repo.

## App
Deployed with Streamlit Community Cloud from `tourism_project/deployment/app.py` (Python 3.11).
