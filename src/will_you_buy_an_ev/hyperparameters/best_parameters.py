import wandb
from dotenv import load_dotenv

# Log into wandb
load_dotenv()
wandb.login()

sweep_id = "otysdcfs"  # best sweep


def get_best_params():
    api = wandb.Api()
    sweep = api.sweep(
        f"coreymichaud-projects/Predicting Electric Vehicle Purchases/{sweep_id}"
    )
    best_run = sweep.best_run()

    best_params = dict(best_run.config)
    print(f"Best ROC AUC: {best_run.summary.get('roc_auc'):.5f}")
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    return best_params
