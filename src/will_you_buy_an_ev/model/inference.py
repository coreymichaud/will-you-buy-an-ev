import joblib
from will_you_buy_an_ev.config.paths import RAW_DATA_PATH, MODELS_PATH
import pandas as pd

# Load test
test = pd.read_csv(RAW_DATA_PATH / "test.csv")

# Load model
pipe = joblib.load(MODELS_PATH / "lgbmc.joblib")

# The Mode Collapse Spike
test['is_30k_spike'] = (test['Annual_Income_USD'] == 30000.0).astype('int8')

# The Millionaire Cliff (100% buy rate region)
test['is_millionaire_cliff'] = (test['Annual_Income_USD'] >= 170537.0).astype('int8')

# The Dead Zone (0% buy rate region)
test['is_dead_zone'] = ((test['Annual_Income_USD'] >= 38000.0) & (test['Annual_Income_USD'] <= 42000.0)).astype('int8')

probs = pipe.predict_proba(test.drop(columns = ["id"]))
preds = probs[:,1]
print(preds)
