import kagglehub
from dotenv import load_dotenv
from will_you_buy_an_ev.config.paths import RAW_DATA_PATH

load_dotenv()

path = kagglehub.competition_download(
    "playground-series-s6e9", output_dir=RAW_DATA_PATH, force_download=True
)

print("Downloaded to:", path)
