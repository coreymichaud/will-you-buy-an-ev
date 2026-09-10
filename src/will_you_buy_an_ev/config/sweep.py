# Sweep configuration for Weight & Biases' hyperparameter tuning
sweep_config = {
    "method": "bayes",
    "name": "bayes-optimization-lgbmclassifier",
    "metric": {"name": "roc_auc", "goal": "maximize"},
    "early_terminate": {"type": "hyperband", "min_iter": 10, "eta": 2},
    "parameters": {
        "n_estimators": {"values": [100, 200, 500, 1000]},
        "learning_rate": {"min": 0.01, "max": 0.3},
        "num_leaves": {"values": [15, 31, 63, 127]},
        "max_depth": {"values": [-1, 5, 10, 15]},
        "min_child_samples": {"values": [5, 10, 20, 50]},
        "subsample": {"min": 0.5, "max": 1.0},
        "colsample_bytree": {"min": 0.5, "max": 1.0},
        "reg_alpha": {"min": 0.0, "max": 5.0},
        "reg_lambda": {"min": 0.0, "max": 5.0},
    },
}
