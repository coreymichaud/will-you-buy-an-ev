import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from will_you_buy_an_ev.config.paths import RAW_DATA_PATH
from will_you_buy_an_ev.data.feature_engineer import create_features
from will_you_buy_an_ev.data.preprocess import get_preprocessor
from will_you_buy_an_ev.hyperparameters.best_parameters import get_best_params

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

# Best parameters from hyperparameter tuning
best_params = get_best_params()

# Tuned model
lgbmc_tuned = LGBMClassifier(**best_params, verbose=-1, random_state=42)

# Pipeline
pipe = Pipeline([("preprocessor", preprocessor), ("classifier", lgbmc_tuned)])

# 5-fold cross validation
scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
print(np.mean(scores))
