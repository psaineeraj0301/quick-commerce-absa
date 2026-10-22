import contextlib
import io
import re
import warnings

import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords, wordnet
from nltk import pos_tag, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from scipy.stats import kruskal, chi2_contingency
from statsmodels.stats.multitest import multipletests

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from config import (
    APPS,
    CLEAN_FILE,
    ASPECT_FILE,
    OUTPUT_DIR,
    ALPHA,
    LOW_RATING_MAX,
    RANDOM_STATE,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
ASPECT_FILE.parent.mkdir(parents=True, exist_ok=True)

N_TOPICS = 5
TOP_WORDS_PER_TOPIC = 10
MIN_REVIEWS_PER_TOPIC_MODEL = 5
MAX_FEATURES = 5000
USE_SENTENCE_TOKENIZATION = True

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 2. NLTK RESOURCES
# ============================================================

def prepare_nltk_resources():
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        (
            "taggers/averaged_perceptron_tagger_eng",
            "averaged_perceptron_tagger_eng",
        ),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("sentiment/vader_lexicon.zip", "vader_lexicon"),
    ]

    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=True)


prepare_nltk_resources()

sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()


# ============================================================
# 3. ASPECT DEFINITIONS
# ============================================================

ASPECT_PATTERNS = {
    "delivery_speed": [
        r"\bdeliv\w*\b", r"\blate\b", r"\bearly\b",
        r"\bdelay\w*\b", r"\bfast\b", r"\bquick\w*\b",
        r"\bslow\w*\b", r"\bon time\b", r"\bminutes?\b",
        r"\bhours?\b", r"\bwaiting\b", r"\bwaited\b",
        r"\beta\b", r"\btime\b",
    ],
    "product_freshness_quality": [
        r"\bfresh\w*\b", r"\bstale\b", r"\brotten\b",
        r"\bspoiled\b", r"\bquality\b", r"\bexpiry\b",
        r"\bexpired\b", r"\bdamaged\b", r"\bbroken\b",
        r"\bcrushed\b", r"\btaste\b", r"\btast\w*\b",
        r"\bpackaging quality\b",
    ],
    "product_availability": [
        r"\bavailab\w*\b", r"\bout of stock\b",
        r"\bstock\b", r"\bunavailable\b",
        r"\bmissing item\b", r"\bnot available\b",
        r"\bproduct\b", r"\bitem\b",
        r"\bsubstitut\w*\b", r"\binventory\b",
    ],
    "pricing_fees": [
        r"\bprice\b", r"\bpricing\b", r"\bcost\b",
        r"\bexpensive\b", r"\bcheap\b", r"\bfee\b",
        r"\bfees\b", r"\bcharge\w*\b",
        r"\bdelivery charge\b", r"\bextra charge\b",
        r"\bconvenience fee\b", r"\bminimum order\b",
        r"\bworth\b", r"\bvalue for money\b",
        r"\brs\.?\b", r"\brupees?\b", r"\binr\b",
    ],
    "offers_discounts": [
        r"\boffer\w*\b", r"\bdiscount\w*\b",
        r"\bcoupon\w*\b", r"\bpromo\w*\b",
        r"\bcashback\b", r"\bsale\b", r"\bdeal\w*\b",
        r"\breward\w*\b", r"\bvoucher\w*\b",
        r"\bfree delivery\b",
    ],
    "order_accuracy_packaging": [
        r"\border\w*\b", r"\bwrong item\b",
        r"\bincorrect item\b", r"\bmissing item\b",
        r"\bmissing product\b", r"\bquantity\b",
        r"\bpackag\w*\b", r"\bsealed\b",
        r"\bunsealed\b", r"\bspill\w*\b",
        r"\bleak\w*\b", r"\bincorrect order\b",
        r"\bcomplete order\b",
    ],
    "delivery_partner": [
        r"\bdelivery partner\b", r"\bdelivery boy\b",
        r"\bdelivery person\b", r"\brider\b",
        r"\bdriver\b", r"\bbehavio\w*\b",
        r"\brude\b", r"\bpolite\b", r"\bcourteous\b",
        r"\bprofessional\b", r"\battitude\b",
    ],
    "customer_support": [
        r"\bcustomer support\b", r"\bcustomer service\b",
        r"\bcustomer care\b", r"\bsupport team\b",
        r"\bhelpline\b", r"\bcontact support\b",
        r"\bresponse\b", r"\brespond\w*\b",
        r"\bcomplaint\w*\b", r"\bagent\b",
        r"\bchat support\b",
    ],
    "refund_payment": [
        r"\brefund\w*\b", r"\breimburse\w*\b",
        r"\bpayment\w*\b", r"\bpaid\b", r"\bpaying\b",
        r"\btransaction\b", r"\bwallet\b", r"\bupi\b",
        r"\bcash\b", r"\bcredit\b", r"\bdebit\b",
        r"\bfailed payment\b", r"\bmoney\b",
        r"\bcharged\b",
    ],
    "app_experience": [
        r"\bapp\b", r"\bapplication\b", r"\binterface\b",
        r"\bui\b", r"\bux\b", r"\blogin\b",
        r"\blogging in\b", r"\bcrash\w*\b",
        r"\bbug\w*\b", r"\berror\w*\b",
        r"\bglitch\w*\b", r"\bsearch\b",
        r"\bnavigat\w*\b", r"\bloading\b",
        r"\bupdate\b", r"\bnotification\w*\b",
    ],
}

COMPILED_ASPECT_PATTERNS = {
    aspect: [
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in patterns
    ]
    for aspect, patterns in ASPECT_PATTERNS.items()
}


# ============================================================
# 4. LOAD CLEANED DATA
# ============================================================

def load_clean_data():
    if not CLEAN_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned file not found: {CLEAN_FILE}\n"
            "Run preprocess.py before running pipeline.py."
        )

    df = pd.read_csv(CLEAN_FILE)

    required_columns = [
        "app", "review_text", "text_clean",
        "text_topic", "rating", "review_id",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in cleaned data: {missing}. "
            "Run the latest preprocess.py."
        )

    df["rating"] = pd.to_numeric(
        df["rating"], errors="coerce"
    )

    df = df.dropna(
        subset=["app", "text_clean", "text_topic", "rating"]
    ).copy()

    df = df[df["rating"].between(1, 5)].copy()
    df["rating"] = df["rating"].astype(int)
    df["app"] = df["app"].astype(str).str.strip()

    df = df[df["app"].isin(list(APPS.keys()))].copy()

    df["text_clean"] = df["text_clean"].fillna("").astype(str)
    df["text_topic"] = df["text_topic"].fillna("").astype(str)
    df["review_id"] = df["review_id"].astype(str)

    df = df[df["text_clean"].str.strip().ne("")].copy()

    if "month" not in df.columns:
        df["month"] = pd.NA

    if "low_rating" not in df.columns:
        df["low_rating"] = (
            df["rating"] <= LOW_RATING_MAX
        ).astype(int)

    if df.empty:
        raise ValueError(
            "No usable reviews were found for the configured apps."
        )

    return df.reset_index(drop=True)


# ============================================================
# 5. SENTENCE AND CLAUSE PROCESSING
# ============================================================

def split_sentences(text):
    text = str(text).strip()

    if not text:
        return []

    if not USE_SENTENCE_TOKENIZATION:
        return [text]

    try:
        return [
            sentence.strip()
            for sentence in sent_tokenize(text)
            if sentence.strip()
        ]
    except LookupError:
        return [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", text)
            if part.strip()
        ]


def split_clauses(text):
    clauses = []

    for sentence in split_sentences(text):
        parts = re.split(
            r"\s+\b(?:but|however|although|though|whereas)\b\s+",
            sentence,
            flags=re.IGNORECASE,
        )

        clauses.extend(
            part.strip() for part in parts if part.strip()
        )

    return clauses


# ============================================================
# 6. ASPECT DETECTION AND SENTIMENT
# ============================================================

def detect_clause_aspects(clause):
    return [
        aspect
        for aspect, patterns in COMPILED_ASPECT_PATTERNS.items()
        if any(pattern.search(clause) for pattern in patterns)
    ]


def classify_sentiment(compound):
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def detect_aspects_and_sentiment(df):
    records = []

    for row in df.itertuples(index=False):
        for clause_index, clause in enumerate(
            split_clauses(row.text_clean)
        ):
            aspects = detect_clause_aspects(clause)

            if not aspects:
                continue

            scores = sia.polarity_scores(clause)
            compound = scores["compound"]
            sentiment = classify_sentiment(compound)

            for aspect in aspects:
                records.append({
                    "review_id": row.review_id,
                    "app": row.app,
                    "rating": row.rating,
                    "month": row.month,
                    "clause_index": clause_index,
                    "aspect": aspect,
                    "clause": clause,
                    "compound": compound,
                    "sentiment": sentiment,
                    "positive_score": scores["pos"],
                    "negative_score": scores["neg"],
                    "neutral_score": scores["neu"],
                })

    columns = [
        "review_id", "app", "rating", "month",
        "clause_index", "aspect", "clause", "compound",
        "sentiment", "positive_score", "negative_score",
        "neutral_score",
    ]

    absa = pd.DataFrame(records, columns=columns)

    absa.to_csv(
        ASPECT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    return absa


# ============================================================
# 7. ASPECT SUMMARY
# ============================================================

def wilson_interval(successes, total, z=1.96):
    if total <= 0:
        return np.nan, np.nan

    proportion = successes / total
    denominator = 1 + z**2 / total

    centre = (
        proportion + z**2 / (2 * total)
    ) / denominator

    margin = (
        z * np.sqrt(
            proportion * (1 - proportion) / total
            + z**2 / (4 * total**2)
        ) / denominator
    )

    return max(0, centre - margin), min(1, centre + margin)


def create_summary(absa):
    output_file = OUTPUT_DIR / "aspect_summary.csv"

    columns = [
        "app", "aspect", "mentions", "positive_pct",
        "negative_pct", "neutral_pct", "net_sentiment",
        "negative_ci_lower", "negative_ci_upper",
        "mean_compound",
    ]

    records = []

    for (app, aspect), group in absa.groupby(
        ["app", "aspect"], observed=True
    ):
        total = len(group)

        positive_count = (group["sentiment"] == "positive").sum()
        negative_count = (group["sentiment"] == "negative").sum()
        neutral_count = (group["sentiment"] == "neutral").sum()

        positive_pct = 100 * positive_count / total
        negative_pct = 100 * negative_count / total
        neutral_pct = 100 * neutral_count / total

        ci_lower, ci_upper = wilson_interval(
            negative_count, total
        )

        records.append({
            "app": app,
            "aspect": aspect,
            "mentions": total,
            "positive_pct": round(positive_pct, 2),
            "negative_pct": round(negative_pct, 2),
            "neutral_pct": round(neutral_pct, 2),
            "net_sentiment": round(
                positive_pct - negative_pct, 2
            ),
            "negative_ci_lower": round(ci_lower * 100, 2),
            "negative_ci_upper": round(ci_upper * 100, 2),
            "mean_compound": round(
                group["compound"].mean(), 4
            ),
        })

    summary = pd.DataFrame(records, columns=columns)
    summary.to_csv(output_file, index=False)

    return summary


# ============================================================
# 8. KRUSKAL-WALLIS TEST
# ============================================================

def run_kruskal_wallis(df):
    output_file = OUTPUT_DIR / "kruskal_wallis.csv"

    groups = [
        group["rating"].dropna().to_numpy()
        for _, group in df.groupby("app")
    ]

    apps_with_data = [
        app for app, group in df.groupby("app")
        if not group["rating"].dropna().empty
    ]

    if len(groups) < 2 or any(len(group) == 0 for group in groups):
        result = pd.DataFrame([{
            "test": "Kruskal-Wallis",
            "statistic": np.nan,
            "p_value": np.nan,
            "alpha": ALPHA,
            "significant": False,
            "apps_compared": ", ".join(apps_with_data),
            "interpretation": "Insufficient data for comparison",
        }])
    else:
        statistic, p_value = kruskal(*groups)

        # Epsilon-squared effect size.
        total_n = sum(len(group) for group in groups)
        number_groups = len(groups)

        epsilon_squared = (
            (statistic - number_groups + 1)
            / (total_n - number_groups)
            if total_n > number_groups
            else np.nan
        )

        result = pd.DataFrame([{
            "test": "Kruskal-Wallis",
            "statistic": statistic,
            "p_value": p_value,
            "alpha": ALPHA,
            "significant": p_value < ALPHA,
            "effect_size_epsilon_squared": max(
                0, epsilon_squared
            ) if pd.notna(epsilon_squared) else np.nan,
            "apps_compared": ", ".join(apps_with_data),
            "interpretation": (
                "At least one app's rating distribution differs."
                if p_value < ALPHA
                else "No significant difference detected."
            ),
        }])

    result.to_csv(output_file, index=False)
    return result


# ============================================================
# 9. CHI-SQUARE TESTS AND BONFERRONI CORRECTION
# ============================================================

def cramers_v(contingency_table):
    n = contingency_table.to_numpy().sum()
    rows, cols = contingency_table.shape

    if n == 0 or min(rows - 1, cols - 1) <= 0:
        return np.nan

    chi2, _, _, _ = chi2_contingency(
        contingency_table,
        correction=False,
    )

    return np.sqrt(
        chi2 / (n * min(rows - 1, cols - 1))
    )


def run_chi_square(absa):
    output_file = OUTPUT_DIR / "chi_square_results.csv"
    records = []

    for aspect, group in absa.groupby("aspect"):
        table = pd.crosstab(
            group["app"], group["sentiment"]
        )

        if table.shape[0] < 2 or table.shape[1] < 2:
            continue

        chi2, p_value, dof, expected = chi2_contingency(table)

        records.append({
            "aspect": aspect,
            "chi_square": chi2,
            "degrees_of_freedom": dof,
            "p_value": p_value,
            "cramers_v": cramers_v(table),
            "min_expected_count": expected.min(),
            "expected_count_warning": bool(
                (expected < 5).any()
            ),
        })

    results = pd.DataFrame(records)

    if not results.empty:
        reject, adjusted_p, _, _ = multipletests(
            results["p_value"].to_numpy(),
            alpha=ALPHA,
            method="bonferroni",
        )

        results["adjusted_p_value"] = adjusted_p
        results["significant"] = reject
        results["interpretation"] = results[
            "significant"
        ].map({
            True: "Sentiment distribution differs by app.",
            False: "No significant association detected.",
        })

        results = results.sort_values("adjusted_p_value")
    else:
        results = pd.DataFrame(columns=[
            "aspect", "chi_square", "degrees_of_freedom",
            "p_value", "cramers_v", "min_expected_count",
            "expected_count_warning", "adjusted_p_value",
            "significant", "interpretation",
        ])

    results.to_csv(output_file, index=False)
    return results


# ============================================================
# 10. LEMMATIZATION AND TOPIC MODELLING
# ============================================================

def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


STOP_WORDS = set(stopwords.words("english"))

# Preserve negation words.
STOP_WORDS.difference_update({
    "not", "no", "nor", "never", "neither", "without",
})

STOP_WORDS.update({
    "app", "apps", "please", "pls", "also",
    "get", "got", "one", "would", "could",
    "really", "just", "zepto", "blinkit",
    "swiggy", "instamart",
})


def normalize_for_lda(text):
    words = re.findall(r"[a-z]+", str(text).lower())

    words = [
        word for word in words
        if word not in STOP_WORDS and len(word) > 2
    ]

    if not words:
        return ""

    tagged_words = pos_tag(words)

    lemmas = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged_words
    ]

    return " ".join(lemmas)


def run_topic_modelling(df):
    output_file = OUTPUT_DIR / "negative_review_topics.csv"

    low_df = df[
        df["rating"] <= LOW_RATING_MAX
    ].copy()

    topic_records = []

    for app in list(APPS.keys()):
        app_df = low_df[low_df["app"] == app].copy()

        if len(app_df) < MIN_REVIEWS_PER_TOPIC_MODEL:
            continue

        app_df["lda_text"] = app_df["text_topic"].apply(
            normalize_for_lda
        )

        app_df = app_df[
            app_df["lda_text"].str.strip().ne("")
        ].copy()

        if len(app_df) < MIN_REVIEWS_PER_TOPIC_MODEL:
            continue

        vectorizer = CountVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b[a-z][a-z]+\b",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=MAX_FEATURES,
        )

        try:
            document_term_matrix = vectorizer.fit_transform(
                app_df["lda_text"]
            )
        except ValueError:
            continue

        if document_term_matrix.shape[1] < 2:
            continue

        n_topics = min(
            N_TOPICS,
            max(1, document_term_matrix.shape[0] // 2),
            document_term_matrix.shape[1],
        )

        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=RANDOM_STATE,
            learning_method="batch",
            max_iter=10,
        )

        lda.fit(document_term_matrix)
        feature_names = vectorizer.get_feature_names_out()

        for topic_index, topic_weights in enumerate(lda.components_):
            top_indices = topic_weights.argsort()[
                -TOP_WORDS_PER_TOPIC:
            ][::-1]

            top_terms = [
                feature_names[index] for index in top_indices
            ]

            topic_records.append({
                "app": app,
                "topic_number": topic_index + 1,
                "review_count": len(app_df),
                "top_terms": ", ".join(top_terms),
            })

    topics = pd.DataFrame(
        topic_records,
        columns=[
            "app", "topic_number", "review_count", "top_terms"
        ],
    )

    topics.to_csv(output_file, index=False)
    return topics


# ============================================================
# 11. MANUAL VALIDATION SAMPLE
# ============================================================

def create_validation_sample(absa):
    output_file = OUTPUT_DIR / "manual_validation_sample.csv"

    if absa.empty:
        pd.DataFrame().to_csv(output_file, index=False)
        return pd.DataFrame()

    sample = absa.sample(
        n=min(100, len(absa)),
        random_state=RANDOM_STATE,
    ).copy()

    sample["human_aspect"] = ""
    sample["human_sentiment"] = ""
    sample["notes"] = ""

    sample.to_csv(output_file, index=False)
    return sample


# ============================================================
# 12. VISUALIZATIONS
# ============================================================

def create_visualizations(df, absa, summary):
    # Rating distribution.
    plt.figure(figsize=(9, 5))

    sns.countplot(
        data=df,
        x="rating",
        hue="app",
    )

    plt.title("Rating Distribution by App")
    plt.xlabel("Star Rating")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "rating_distribution.png",
        dpi=200,
    )
    plt.close()

    if absa.empty or summary.empty:
        return

    # Aspect mention frequency.
    aspect_counts = (
        absa.groupby(["app", "aspect"])
        .size()
        .reset_index(name="mentions")
    )

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=aspect_counts,
        x="aspect",
        y="mentions",
        hue="app",
    )

    plt.title("Aspect Mention Frequency by App")
    plt.xlabel("Aspect")
    plt.ylabel("Mentions")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "aspect_frequency.png",
        dpi=200,
    )
    plt.close()

    # Negative sentiment percentages.
    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=summary,
        x="aspect",
        y="negative_pct",
        hue="app",
    )

    plt.title("Negative Sentiment by Aspect and App")
    plt.xlabel("Aspect")
    plt.ylabel("Negative Sentiment (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "negative_sentiment_by_aspect.png",
        dpi=200,
    )
    plt.close()

    # Net sentiment heatmap.
    heatmap_data = summary.pivot(
        index="aspect",
        columns="app",
        values="net_sentiment",
    )

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        center=0,
        cmap="RdYlGn",
    )

    plt.title("Net Sentiment Heatmap")
    plt.xlabel("App")
    plt.ylabel("Aspect")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "net_sentiment_heatmap.png",
        dpi=200,
    )
    plt.close()

    # Monthly review trend.
    if "month" in df.columns:
        monthly = (
            df.dropna(subset=["month"])
            .groupby(["month", "app"])
            .size()
            .reset_index(name="reviews")
        )

        if not monthly.empty:
            plt.figure(figsize=(12, 5))

            sns.lineplot(
                data=monthly,
                x="month",
                y="reviews",
                hue="app",
                marker="o",
            )

            plt.title("Monthly Review Volume")
            plt.xlabel("Month")
            plt.ylabel("Number of Reviews")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            plt.savefig(
                OUTPUT_DIR / "monthly_review_trend.png",
                dpi=200,
            )
            plt.close()


# ============================================================
# 13. MAIN PIPELINE
# ============================================================

def main():
    # Suppress normal output while the pipeline runs.
    # Exceptions are not swallowed; real errors remain visible.
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        df = load_clean_data()

        absa = detect_aspects_and_sentiment(df)

        summary = create_summary(absa)

        run_kruskal_wallis(df)
        run_chi_square(absa)

        run_topic_modelling(df)

        create_validation_sample(absa)

        create_visualizations(df, absa, summary)

    print("Pipeline completed.", flush=True)


if __name__ == "__main__":
    main()
