import os
from dotenv import load_dotenv

load_dotenv()

# Core paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA = os.getenv("RAW_DATA", os.path.join(PROJECT_ROOT, "reviews_master.csv"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
MODELS_DIR = os.path.join(DATA_DIR, "models")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Alerting
NEGATIVE_SPIKE_THRESHOLD = float(os.getenv("NEGATIVE_SPIKE_THRESHOLD", 0.2))  # 20%
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "")
ALERT_EMAIL_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD", "")

# Modeling
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
MAX_ROWS = int(os.getenv("MAX_ROWS", 50000))

# Dashboard
APP_TITLE = os.getenv("APP_TITLE", "NLP-Powered Sentiment & Review Analytics")
COLOR_PALETTE = {
    # Selected professional palette (Option 2): golds + charcoal + light gray
    "indigo": "#AD974F",   # primary gold
    "emerald": "#8E793E",  # secondary gold (used as accent)
    "amber": "#EAEAEA",    # light neutral accent
    "light": "#EAEAEA",    # background light
    "dark": "#231F20",     # charcoal text
}
