import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from will_you_buy_an_ev.config.paths import RAW_DATA_PATH
from will_you_buy_an_ev.data.feature_engineer import create_features
from will_you_buy_an_ev.data.preprocess import get_preprocessor

# Load data
train = pd.read_csv(RAW_DATA_PATH / "train.csv")
test = pd.read_csv(RAW_DATA_PATH / "test.csv")

# Drop unused 'id' column
train = train.drop(columns=["id"])

# Create X/y split
X = train.drop(columns=["Will_Buy_EV"])
y = train["Will_Buy_EV"].map({"No": 0, "Yes": 1})

# Feature engineering
X = create_features(X)

# Get categorical and numeric columns
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns

# Create preprocessor
preprocessor = get_preprocessor(cat_cols, num_cols)

# Models
xgbc = XGBClassifier()
lr = LogisticRegression()
lgbc = LGBMClassifier(verbose=-1)
catboost = CatBoostClassifier(allow_writing_files=False, verbose=False)

# Model list
models = {
    "XGBClassifier": xgbc,
    "Logistic Regression": lr,
    "LightGBM": lgbc,
    "CatBoost": catboost,
}

model_scores = {}


# Cross validation
for model_name, model in models.items():
    # Pipeline
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", model)])

    # CV with scoring as ROC AUC
    print(f"Starting {model_name} CV...")
    scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")

    model_scores.update({model_name: np.mean(scores)})


# Sort and print model scores
model_scores = {
    k: v
    for k, v in sorted(model_scores.items(), key=lambda item: item[1], reverse=True)
}

for model, score in model_scores.items():
    print(f"{model}: {score}\n")
