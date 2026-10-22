from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

RAW_FILE = DATA_DIR / "raw_reviews.csv"
CLEAN_FILE = DATA_DIR / "clean_reviews.csv"
ASPECT_FILE = DATA_DIR / "aspect_mentions.csv"

REVIEWS_PER_APP = 2000
BATCH_SIZE = 200
LANGUAGE = "en"
COUNTRY = "in"
REQUEST_PAUSE_SECONDS = 1
MAX_ERRORS = 5

ALPHA = 0.05
LOW_RATING_MAX = 2
RANDOM_STATE = 42

# These are Google Play app package IDs.
APPS = {
    "Zepto": "com.zeptoconsumerapp",
    "Blinkit": "com.grofers.customerapp",
    "Swiggy Instamart": "in.swiggy.android.instamart",
}

SWIGGY_KEYWORD_FILTER = True

SWIGGY_KEYWORDS = [
    "instamart", "grocery", "groceries", "vegetable", "vegetables",
    "fruit", "fruits", "milk", "bread", "eggs", "rice", "atta",
    "grocery order", "dark store", "quick commerce", "quick delivery",
    "zepto", "blinkit"
]

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)