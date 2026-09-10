import wandb
from lightgbm import LGBMClassifier
from dotenv import load_dotenv
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from will_you_buy_an_ev.config.sweep import sweep_config
from will_you_buy_an_ev.data.preprocess import get_preprocessor
from will_you_buy_an_ev.config.paths import RAW_DATA_PATH
from will_you_buy_an_ev.data.feature_engineer import create_features
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

# Log into wandb
load_dotenv()
wandb.login()

WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")

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


def sweeps():
    with wandb.init() as run:
        config = wandb.config

        # Parameters from
        params = {
            "n_estimators": config.n_estimators,
            "learning_rate": config.learning_rate,
            "num_leaves": config.num_leaves,
            "max_depth": config.max_depth,
            "min_child_samples": config.min_child_samples,
            "subsample": config.subsample,
            "colsample_bytree": config.colsample_bytree,
            "reg_alpha": config.reg_alpha,
            "reg_lambda": config.reg_lambda,
            "verbose": -1,
            "random_state": 42,
        }

        lgbmc = LGBMClassifier(**params)

        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", lgbmc)])

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        pipe.fit(X_train, y_train)

        val_probs = pipe.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_probs)
        loss = log_loss(y_val, val_probs)

        wandb.log({"roc_auc": auc, "loss": loss})

        model_path = f"model_{run.id}.txt"
        lgbmc.booster_.save_model(model_path)
        artifact = wandb.Artifact(f"lgbm-model-{run.id}", type="model")
        artifact.add_file(model_path)
        run.log_artifact(artifact)


sweep_id = wandb.sweep(sweep=sweep_config, entity=WANDB_ENTITY, project=WANDB_PROJECT)

wandb.agent(sweep_id, function=sweeps, count=10)
