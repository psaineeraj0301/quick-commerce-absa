
import re
import pandas as pd
import emoji

from config import (
    RAW_FILE,
    CLEAN_FILE,
    OUTPUT_DIR,
    LOW_RATING_MAX,
)
    
# CONFIGURATION

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)



# CONTRACTION EXPANSION


CONTRACTIONS = {
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "can't": "cannot",
    "couldn't": "could not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "won't": "will not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "mustn't": "must not",
    "needn't": "need not",
    "mightn't": "might not",
    "shan't": "shall not",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "here's": "here is",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "we'd": "we would",
    "they'd": "they would",
    "what's": "what is",
    "who's": "who is",
    "where's": "where is",
    "could've": "could have",
    "would've": "would have",
    "should've": "should have",
}


def expand_contractions(text):
    """Expand common contractions while preserving negation."""

    for contraction, replacement in CONTRACTIONS.items():
        text = re.sub(
            r"\b" + re.escape(contraction) + r"\b",
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    return text



# SELECTIVE ABBREVIATION EXPANSION


ABBREVIATIONS = {
    "pls": "please",
    "plz": "please",
    "asap": "as soon as possible",
    "btw": "by the way",
    "idk": "i do not know",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "tbh": "to be honest",
    "rn": "right now",
    "coz": "because",
    "bcoz": "because",
    "thx": "thanks",
    "ty": "thank you",
    "tysm": "thank you so much",
    "u": "you",
    "ur": "your",
}


def expand_abbreviations(text):
    """Expand selected common review abbreviations."""

    for abbreviation, replacement in ABBREVIATIONS.items():
        text = re.sub(
            r"\b" + re.escape(abbreviation) + r"\b",
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    return text


# HTML/XML TAG AND URL REMOVAL

def remove_html_xml_and_urls(text):
    """Remove URLs, email addresses and HTML/XML markup."""

    # Remove web links.
    text = re.sub(
        r"(?:https?://|www\.)\S+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove email addresses.
    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b",
        " ",
        text,
    )

    # Remove HTML/XML comments.
    text = re.sub(
        r"<!--.*?-->",
        " ",
        text,
        flags=re.DOTALL,
    )

    # Remove HTML/XML tags.
    text = re.sub(r"<[^>]*>", " ", text)

    return text

# TEXT CLEANING AND NORMALIZATION

def prepare_base_text(text):
    """
    Common normalization:
    - Convert text to lowercase.
    - Remove URLs and HTML/XML markup.
    - Expand contractions and selected abbreviations.
    - Normalize whitespace.
    - Preserve emojis, punctuation and numbers.
    """

    text = str(text)
    text = remove_html_xml_and_urls(text)
    text = text.lower()
    text = expand_contractions(text)
    text = expand_abbreviations(text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_text(text):
    """
    Text for aspect detection and VADER sentiment analysis.

    Emojis, punctuation and numbers are preserved.
    """

    return prepare_base_text(text)


def create_topic_text(text):
    """
    Text for LDA topic modelling.

    Convert emojis into text descriptions and remove
    punctuation and special characters.
    """

    text = prepare_base_text(text)

    # Convert emoji symbols to descriptive words.
    text = emoji.demojize(
        text,
        delimiters=(" ", " "),
    )

    # Keep English letters, numbers and whitespace.
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text

# DEVANAGARI-SCRIPT FILTER

DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def devanagari_ratio(text):
    """
    Calculate the proportion of alphabetic characters
    that belong to the Devanagari Unicode block.
    """

    letters = [
        char for char in str(text)
        if char.isalpha()
    ]

    if not letters:
        return 0.0

    devanagari_letters = sum(
        bool(DEVANAGARI.match(char))
        for char in letters
    )

    return devanagari_letters / len(letters)


# PREPROCESS RAW DATA

def preprocess_data():

    # LOAD RAW DATA


    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw file not found: {RAW_FILE}\n"
            "Run collect_reviews.py first."
        )

    df = pd.read_csv(RAW_FILE)

    required_columns = [
        "app",
        "review_text",
        "rating",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Original dataset shape:", df.shape)

    # EMOVE MISSING AND EMPTY REVIEWS

    df = df.dropna(
        subset=["app", "review_text", "rating"]
    ).copy()

    df["app"] = df["app"].astype(str).str.strip()

    df["review_text"] = (
        df["review_text"].astype(str).str.strip()
    )

    df = df[
        df["app"].ne("")
        & df["review_text"].ne("")
    ].copy()


    # VALIDATE RATINGS


    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce",
    )

    df = df[
        df["rating"].notna()
        & df["rating"].between(1, 5)
    ].copy()

    df["rating"] = df["rating"].astype(int)


    # REMOVE DUPLICATE REVIEWS


    normalized_text = (
        df["review_text"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    if "review_id_original" in df.columns:

        has_id = (
            df["review_id_original"].notna()
            & df["review_id_original"]
            .astype(str)
            .str.strip()
            .ne("")
        )

        df["_duplicate_key"] = normalized_text

        df.loc[has_id, "_duplicate_key"] = (
            "id:"
            + df.loc[
                has_id, "review_id_original"
            ].astype(str)
        )

    else:
        df["_duplicate_key"] = normalized_text

    df = df.drop_duplicates(
        subset=["app", "_duplicate_key"],
        keep="first",
    ).copy()

    df = df.drop(columns=["_duplicate_key"])

    print("Shape after duplicate removal:", df.shape)

    
    # FILTER PREDOMINANTLY DEVANAGARI REVIEWS


    before_language_filter = len(df)

    df = df[
        df["review_text"].apply(devanagari_ratio) < 0.30
    ].copy()

    dropped_count = before_language_filter - len(df)

    print(
        f"Dropped {dropped_count} predominantly "
        "Devanagari-script reviews"
    )

    print(
        "Shape after script filtering:",
        df.shape,
    )

   
    # CREATE SEPARATE TEXT COLUMNS
   
    # For aspect detection and VADER.
    df["text_clean"] = (
        df["review_text"].apply(clean_text)
    )

    # For LDA topic modelling.
    df["text_topic"] = (
        df["review_text"].apply(create_topic_text)
    )

    # Remove reviews with no usable English letters or numbers.
    df = df[
        df["text_clean"].str.contains(
            r"[a-z0-9]",
            regex=True,
            na=False,
        )
        & df["text_topic"].str.contains(
            r"[a-z0-9]",
            regex=True,
            na=False,
        )
    ].copy()

   
    # STANDARDIZE DATES
 

    if "review_date" in df.columns:

        df["review_date"] = pd.to_datetime(
            df["review_date"],
            errors="coerce",
            utc=True,
        )

        df["month"] = (
            df["review_date"].dt.strftime("%Y-%m")
        )

    else:
        df["review_date"] = pd.NaT
        df["month"] = pd.NA


    # HANDLE OPTIONAL METADATA
  

    if "app_version" in df.columns:
        df["app_version"] = (
            df["app_version"].fillna("Unknown")
        )
    else:
        df["app_version"] = "Unknown"

    if "thumbs_up" in df.columns:
        df["thumbs_up"] = pd.to_numeric(
            df["thumbs_up"],
            errors="coerce",
        )
    else:
        df["thumbs_up"] = 0

        
    # CREATE ANALYSIS COLUMNS


    df["n_words"] = (
        df["text_clean"].str.split().str.len()
    )

    df["low_rating"] = (
        df["rating"] <= LOW_RATING_MAX
    ).astype(int)

    df = df.reset_index(drop=True)

    # Unique row ID for downstream analysis.
    df["review_id"] = range(len(df))

    
    # DATA QUALITY REPORTS
   

    missing_report = pd.DataFrame({
        "column": df.columns,
        "missing_count": [
            df[column].isna().sum()
            for column in df.columns
        ],
    })

    missing_report["missing_percentage"] = (
        missing_report["missing_count"]
        / max(len(df), 1)
        * 100
    ).round(2)

    missing_report.to_csv(
        OUTPUT_DIR / "missing_values_report.csv",
        index=False,
    )

    # Rating distribution by app.
    rating_distribution = pd.crosstab(
        df["app"],
        df["rating"],
    )

    rating_distribution.to_csv(
        OUTPUT_DIR / "rating_distribution.csv",
    )

    # Review length statistics.
    review_length_summary = (
        df.groupby("app")["n_words"]
        .describe()
        .round(2)
    )

    review_length_summary.to_csv(
        OUTPUT_DIR / "review_length_summary.csv",
    )

    # Count reviews containing non-ASCII characters.
    # This includes emojis and non-English scripts.
    non_ascii_count = df["review_text"].apply(
        lambda text: bool(
            re.search(r"[^\x00-\x7F]", str(text))
        )
    ).sum()

    # Count reviews with emojis in the original text.
    emoji_count = df["review_text"].apply(
        lambda text: emoji.emoji_count(str(text)) > 0
    ).sum()

   
    # PRINT REPORTS
   

    print("\nFinal dataset shape:", df.shape)

    print("\nReview count by app:")
    print(df.groupby("app").size())

    print("\nRating distribution:")
    print(rating_distribution)

    print("\nReview length summary:")
    print(review_length_summary)

    print(
        "\nReviews containing non-ASCII characters:",
        int(non_ascii_count),
    )

    print(
        "Reviews containing emojis:",
        int(emoji_count),
    )

    print("\nMissing values by column:")
    print(df.isna().sum())

    # Show an example of the transformations.
    if not df.empty:
        example = df.iloc[0]

        print("\nTEXT PREPROCESSING EXAMPLE")
        print("Original review:", example["review_text"])
        print("Clean text:", example["text_clean"])
        print("Topic text:", example["text_topic"])

   
    # SAVE CLEANED DATA
    

    df.to_csv(
        CLEAN_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nCleaned dataset saved to: {CLEAN_FILE}")

    return df

# RUN SCRIPT

if __name__ == "__main__":
    preprocess_data()
