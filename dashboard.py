
import pandas as pd
import streamlit as st
import plotly.express as px

from config import OUTPUT_DIR, CLEAN_FILE, ASPECT_FILE

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Quick-Commerce ABSA",
    page_icon="🛒",
    layout="wide",
)

APP_COLORS = {
    "Blinkit": "#F8CB46",
    "Zepto": "#950EDB",
    "Swiggy Instamart": "#0050FF",
}

CHART_TEMPLATE = "plotly_white"

APP_ORDER = [
    "Blinkit",
    "Zepto",
    "Swiggy Instamart",
]

st.title("Quick-Commerce App Review Analysis")
st.caption(
    "Aspect-Based Sentiment Analysis | "
    "Zepto vs Blinkit vs Swiggy Instamart"
)

# HELPER FUNCTIONS

def normalize_app_name(name):
    """Standardize app names for filtering and chart colours."""

    if pd.isna(name):
        return name

    name_lower = str(name).strip().lower()

    if "blinkit" in name_lower or "grofers" in name_lower:
        return "Blinkit"

    if "zepto" in name_lower:
        return "Zepto"

    if "swiggy" in name_lower or "instamart" in name_lower:
        return "Swiggy Instamart"

    return str(name).strip()


def style_chart(fig):
    """Apply consistent chart styling."""

    fig.update_layout(
        template=CHART_TEMPLATE,
        legend_title_text="App",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def load_csv(path):
    """Read a CSV file safely."""

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)

    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as error:
        st.warning(f"Could not read {path.name}: {error}")
        return None

    except OSError as error:
        st.warning(f"Could not open {path.name}: {error}")
        return None


def show_result_table(filename, description=None):
    """Display a pipeline result file when available."""

    path = OUTPUT_DIR / filename

    if description:
        st.caption(description)

    result = load_csv(path)

    if result is None:
        st.info(
            f"{filename} is not available. "
            "Run pipeline.py first."
        )
        return

    if result.empty:
        st.info(f"{filename} exists but contains no results.")
        return

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True,
    )


def empty_message(message):
    """Display a consistent empty-data message."""
    st.info(message)


def build_aspect_summary(absa):
    """
    Build an aspect summary directly from aspect-level records.

    Required source columns:
    app, aspect, sentiment
    """

    required = {"app", "aspect", "sentiment"}

    if not required.issubset(absa.columns) or absa.empty:
        return pd.DataFrame(
            columns=[
                "app",
                "aspect",
                "mentions",
                "positive_pct",
                "negative_pct",
                "neutral_pct",
                "net_sentiment",
            ]
        )

    data = absa.copy()

    data["sentiment"] = (
        data["sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    data = data[
        data["sentiment"].isin(
            ["positive", "neutral", "negative"]
        )
    ].copy()

    if data.empty:
        return pd.DataFrame(
            columns=[
                "app",
                "aspect",
                "mentions",
                "positive_pct",
                "negative_pct",
                "neutral_pct",
                "net_sentiment",
            ]
        )

    counts = (
        data.groupby(["app", "aspect", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(
            columns=["positive", "neutral", "negative"],
            fill_value=0,
        )
        .reset_index()
    )

    counts["mentions"] = (
        counts["positive"]
        + counts["neutral"]
        + counts["negative"]
    )

    denominator = counts["mentions"].replace(0, float("nan"))

    counts["positive_pct"] = (
        counts["positive"] / denominator * 100
    ).fillna(0)

    counts["negative_pct"] = (
        counts["negative"] / denominator * 100
    ).fillna(0)

    counts["neutral_pct"] = (
        counts["neutral"] / denominator * 100
    ).fillna(0)

    counts["net_sentiment"] = (
        counts["positive_pct"] - counts["negative_pct"]
    )

    return counts[
        [
            "app",
            "aspect",
            "mentions",
            "positive_pct",
            "negative_pct",
            "neutral_pct",
            "net_sentiment",
        ]
    ]

# LOAD REQUIRED DATA

required_files = [CLEAN_FILE, ASPECT_FILE]

missing_files = [
    path.name
    for path in required_files
    if not path.exists()
]

if missing_files:
    st.warning(
        "Required analysis files are missing: "
        + ", ".join(missing_files)
    )

    st.info(
        "Run the scripts in this order: "
        "collect_reviews.py → preprocess.py → pipeline.py"
    )

    st.stop()


df = load_csv(CLEAN_FILE)
absa = load_csv(ASPECT_FILE)

if df is None or absa is None:
    st.error("The review data could not be loaded.")
    st.stop()


# Load saved summary if available.
summary = load_csv(OUTPUT_DIR / "aspect_summary.csv")

if summary is None:
    summary = pd.DataFrame()

# NORMALIZE AND VALIDATE DATA

for data in (df, absa, summary):
    if "app" in data.columns:
        data["app"] = data["app"].apply(normalize_app_name)


# Validate cleaned review dataset.
required_review_columns = {"app", "rating"}

if not required_review_columns.issubset(df.columns):
    st.error(
        "The cleaned review dataset must contain "
        "'app' and 'rating' columns."
    )
    st.stop()


# Validate aspect-level dataset.
required_aspect_columns = {
    "app",
    "aspect",
    "sentiment",
    "compound",
    "clause",
}

missing_aspect_columns = (
    required_aspect_columns - set(absa.columns)
)

if missing_aspect_columns:
    st.error(
        "The aspect dataset is missing these columns: "
        + ", ".join(sorted(missing_aspect_columns))
        + ". Check pipeline.py and aspect_mentions.csv."
    )
    st.stop()


# Standardize saved summary column names.
if not summary.empty:
    summary.columns = (
        summary.columns.astype(str)
        .str.strip()
        .str.lower()
    )

    summary = summary.rename(
        columns={
            "aspect_name": "aspect",
            "aspect_category": "aspect",
            "positive_percentage": "positive_pct",
            "negative_percentage": "negative_pct",
            "neutral_percentage": "neutral_pct",
            "positive_percent": "positive_pct",
            "negative_percent": "negative_pct",
            "neutral_percent": "neutral_pct",
            "net_sentiment_score": "net_sentiment",
            "mention_count": "mentions",
            "total_mentions": "mentions",
        }
    )


required_summary_columns = {
    "app",
    "aspect",
    "mentions",
    "positive_pct",
    "negative_pct",
    "neutral_pct",
    "net_sentiment",
}

missing_summary_columns = (
    required_summary_columns - set(summary.columns)
)

# FIX: rebuild summary if missing, empty, or incompatible.
if summary.empty or missing_summary_columns:
    summary = build_aspect_summary(absa)

    if not summary.empty:
        st.info(
            "The saved aspect summary was missing or had an "
            "unexpected format. It has been rebuilt from "
            "aspect_mentions.csv for this dashboard session."
        )

        # Save the corrected summary for future dashboard runs.
        try:
            summary.to_csv(
                OUTPUT_DIR / "aspect_summary.csv",
                index=False,
            )
        except OSError as error:
            st.warning(
                f"Could not save the rebuilt summary: {error}"
            )

    else:
        st.warning(
            "No valid aspect-level sentiment records were found. "
            "Check aspect_mentions.csv and run pipeline.py again."
        )
        summary = pd.DataFrame(
            columns=sorted(required_summary_columns)
        )


# Convert ratings and summary measures to numeric values.
df["rating"] = pd.to_numeric(
    df["rating"],
    errors="coerce",
)

df = df.dropna(subset=["app", "rating"]).copy()
df = df[df["rating"].between(1, 5)].copy()
df["rating"] = df["rating"].astype(int)

for column in [
    "mentions",
    "positive_pct",
    "negative_pct",
    "neutral_pct",
    "net_sentiment",
]:
    if column in summary.columns:
        summary[column] = pd.to_numeric(
            summary[column],
            errors="coerce",
        )

absa["sentiment"] = (
    absa["sentiment"]
    .astype(str)
    .str.strip()
    .str.lower()
)

absa["compound"] = pd.to_numeric(
    absa["compound"],
    errors="coerce",
)

if "rating" in absa.columns:
    absa["rating"] = pd.to_numeric(
        absa["rating"],
        errors="coerce",
    )


# Filter to the three configured apps.
df = df[df["app"].isin(APP_ORDER)].copy()
absa = absa[absa["app"].isin(APP_ORDER)].copy()
summary = summary[summary["app"].isin(APP_ORDER)].copy()

if df.empty:
    st.error(
        "No reviews were found for the three configured apps. "
        "Check the app names in config.py and clean_reviews.csv."
    )
    st.stop()

# SIDEBAR FILTERS

st.sidebar.header("Dashboard Filters")

available_apps = [
    app for app in APP_ORDER
    if app in df["app"].dropna().unique()
]

selected_app = st.sidebar.selectbox(
    "Select app",
    ["All apps"] + available_apps,
)

min_rating = st.sidebar.selectbox(
    "Minimum star rating",
    [1, 2, 3, 4, 5],
    index=0,
)

max_rating = st.sidebar.selectbox(
    "Maximum star rating",
    [1, 2, 3, 4, 5],
    index=4,
)

if min_rating > max_rating:
    st.sidebar.warning(
        "Minimum rating exceeds maximum rating. "
        "The rating filter will not be applied."
    )
    min_rating, max_rating = 1, 5


# Apply filters to review-level data.
filtered_df = df.copy()

if selected_app != "All apps":
    filtered_df = filtered_df[
        filtered_df["app"] == selected_app
    ].copy()

filtered_df = filtered_df[
    filtered_df["rating"].between(min_rating, max_rating)
].copy()


# Apply app filter to aspect-level data.
if selected_app != "All apps":
    filtered_absa = absa[
        absa["app"] == selected_app
    ].copy()

    filtered_summary = summary[
        summary["app"] == selected_app
    ].copy()

else:
    filtered_absa = absa.copy()
    filtered_summary = summary.copy()


# Restrict aspect records to reviews that pass the filters.
if (
    "review_id" in filtered_df.columns
    and "review_id" in filtered_absa.columns
):
    selected_review_ids = set(
        filtered_df["review_id"].astype(str)
    )

    filtered_absa = filtered_absa[
        filtered_absa["review_id"].astype(str).isin(
            selected_review_ids
        )
    ].copy()


# Recalculate sentiment summary when rating filters are applied.
if min_rating != 1 or max_rating != 5:
    filtered_summary = build_aspect_summary(filtered_absa)

# OVERVIEW AND KEY PERFORMANCE INDICATORS

st.subheader("Overview")

col1, col2, col3, col4 = st.columns(4)

review_count = len(filtered_df)

average_rating = (
    filtered_df["rating"].mean()
    if review_count
    else 0
)

low_rating_pct = (
    (filtered_df["rating"] <= 2).mean() * 100
    if review_count
    else 0
)

aspect_mentions_count = len(filtered_absa)

col1.metric(
    "Reviews analysed",
    f"{review_count:,}",
)

col2.metric(
    "Average star rating",
    f"{average_rating:.2f} / 5",
)

col3.metric(
    "1–2 star reviews",
    f"{low_rating_pct:.1f}%",
)

col4.metric(
    "Aspect mentions",
    f"{aspect_mentions_count:,}",
)

st.divider()

# STAR-RATING DISTRIBUTION

st.subheader("1. Star-rating distribution")

if not filtered_df.empty:
    rating_counts = (
        filtered_df.groupby(["rating", "app"])
        .size()
        .reset_index(name="reviews")
    )

    fig_rating = px.bar(
        rating_counts,
        x="rating",
        y="reviews",
        color="app",
        barmode="group",
        category_orders={
            "app": APP_ORDER,
            "rating": [1, 2, 3, 4, 5],
        },
        color_discrete_map=APP_COLORS,
        title="Number of reviews by star rating",
        labels={
            "rating": "Star rating",
            "reviews": "Number of reviews",
            "app": "App",
        },
    )

    fig_rating.update_layout(xaxis=dict(dtick=1))

    st.plotly_chart(
        style_chart(fig_rating),
        use_container_width=True,
    )

else:
    empty_message("No rating data is available for this selection.")

# CUSTOMER SENTIMENT BY ASPECT - FIXED

st.subheader("2. Customer sentiment by aspect")

if not filtered_summary.empty:
    # Convert summary to long format for plotting all sentiments.
    sentiment_chart_data = filtered_summary.melt(
        id_vars=["app", "aspect"],
        value_vars=[
            "positive_pct",
            "neutral_pct",
            "negative_pct",
        ],
        var_name="sentiment",
        value_name="percentage",
    )

    sentiment_chart_data["sentiment"] = (
        sentiment_chart_data["sentiment"]
        .str.replace("_pct", "", regex=False)
        .str.capitalize()
    )

    sentiment_chart_data["percentage"] = pd.to_numeric(
        sentiment_chart_data["percentage"],
        errors="coerce",
    )

    sentiment_chart_data = sentiment_chart_data.dropna(
        subset=["percentage", "aspect", "app"]
    )

    if not sentiment_chart_data.empty:
        fig_aspect = px.bar(
            sentiment_chart_data,
            x="aspect",
            y="percentage",
            color="sentiment",
            barmode="group",
            facet_col="app",
            facet_col_wrap=1,
            category_orders={
                "app": APP_ORDER,
                "sentiment": [
                    "Positive",
                    "Neutral",
                    "Negative",
                ],
            },
            color_discrete_map={
                "Positive": "#2E8B57",
                "Neutral": "#A0A0A0",
                "Negative": "#D9534F",
            },
            hover_data={"percentage": ":.1f"},
            title="Positive, neutral and negative sentiment by aspect",
            labels={
                "aspect": "Customer aspect",
                "percentage": "Aspect mentions (%)",
                "sentiment": "Sentiment",
                "app": "App",
            },
        )

        fig_aspect.update_layout(
            xaxis_tickangle=-35,
            yaxis_range=[0, 100],
            legend_title_text="Sentiment",
        )

        fig_aspect.for_each_annotation(
            lambda annotation: annotation.update(
                text=annotation.text.split("=")[-1]
            )
        )

        st.plotly_chart(
            style_chart(fig_aspect),
            use_container_width=True,
        )

    else:
        empty_message(
            "The summary exists, but it has no valid sentiment percentages."
        )

else:
    empty_message(
        "No aspect sentiment summary is available. "
        "Check aspect_mentions.csv for app, aspect and sentiment columns, "
        "then run pipeline.py."
    )

# NET SENTIMENT HEATMAP

st.subheader("3. Net sentiment heatmap")

if not filtered_summary.empty:
    pivot = filtered_summary.pivot_table(
        index="aspect",
        columns="app",
        values="net_sentiment",
        aggfunc="mean",
    )

    heatmap_columns = [
        app for app in APP_ORDER
        if app in pivot.columns
    ]

    pivot = pivot.reindex(columns=heatmap_columns)

    if not pivot.empty:
        fig_heatmap = px.imshow(
            pivot,
            text_auto=".1f",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            zmin=-100,
            zmax=100,
            aspect="auto",
            title=(
                "Net sentiment: positive percentage "
                "minus negative percentage"
            ),
            labels={
                "x": "App",
                "y": "Customer aspect",
                "color": "Net sentiment",
            },
        )

        st.plotly_chart(
            style_chart(fig_heatmap),
            use_container_width=True,
        )

    else:
        empty_message("There is not enough data to display the heatmap.")

else:
    empty_message("No sentiment data is available for the heatmap.")

# ASPECT SUMMARY TABLE

st.subheader("4. Aspect summary table")

if not filtered_summary.empty:
    sort_column = (
        "negative_pct"
        if "negative_pct" in filtered_summary.columns
        else "aspect"
    )

    st.dataframe(
        filtered_summary.sort_values(
            sort_column,
            ascending=(sort_column == "aspect"),
        ),
        use_container_width=True,
        hide_index=True,
    )

else:
    empty_message("No aspect summary rows are available.")


# MOST DISCUSSED ASPECTS

st.subheader("5. What do customers talk about most?")

if not filtered_absa.empty:
    frequency = (
        filtered_absa.groupby(["aspect", "app"])
        .size()
        .reset_index(name="mentions")
    )

    fig_frequency = px.bar(
        frequency,
        x="aspect",
        y="mentions",
        color="app",
        barmode="group",
        category_orders={"app": APP_ORDER},
        color_discrete_map=APP_COLORS,
        title="Number of mentions by aspect",
        labels={
            "aspect": "Customer aspect",
            "mentions": "Number of mentions",
            "app": "App",
        },
    )

    fig_frequency.update_layout(xaxis_tickangle=-35)

    st.plotly_chart(
        style_chart(fig_frequency),
        use_container_width=True,
    )

else:
    empty_message("No aspect mentions are available for this selection.")


# SENTIMENT DISTRIBUTION BY ASPECT

st.subheader("6. Sentiment distribution by aspect")

if not filtered_absa.empty:
    sentiment_counts = (
        filtered_absa.groupby(
            ["app", "aspect", "sentiment"]
        )
        .size()
        .reset_index(name="mentions")
    )

    fig_sentiment = px.bar(
        sentiment_counts,
        x="aspect",
        y="mentions",
        color="sentiment",
        facet_row="app",
        barmode="stack",
        category_orders={
            "app": APP_ORDER,
            "sentiment": [
                "positive",
                "neutral",
                "negative",
            ],
        },
        color_discrete_map={
            "positive": "#2E8B57",
            "neutral": "#A0A0A0",
            "negative": "#D9534F",
        },
        title="Positive, neutral and negative aspect mentions",
        labels={
            "aspect": "Customer aspect",
            "mentions": "Number of mentions",
            "sentiment": "Sentiment",
            "app": "App",
        },
    )

    fig_sentiment.update_layout(xaxis_tickangle=-35)

    st.plotly_chart(
        style_chart(fig_sentiment),
        use_container_width=True,
    )

else:
    empty_message("No sentiment records are available.")

# STATISTICAL TEST RESULTS

st.subheader("7. Statistical test results")

st.caption(
    "Results are generated by pipeline.py. "
    "Interpret p-values alongside sample sizes, assumptions, "
    "effect sizes, and any expected-count warnings."
)

with st.expander(
    "Kruskal–Wallis rating comparison",
    expanded=False,
):
    st.write(
        "Compares rating distributions across the apps. "
        "A p-value below the selected significance level indicates "
        "evidence that at least one app's rating distribution differs."
    )

    show_result_table("kruskal_wallis.csv")


with st.expander(
    "Chi-square test of independence",
    expanded=False,
):
    st.write(
        "Tests whether sentiment categories for an aspect are "
        "associated with the app. Check adjusted p-values, Cramér's V, "
        "and expected cell counts when interpreting results."
    )

    show_result_table("chi_square_results.csv")


with st.expander(
    "Topics in low-rated reviews (LDA)",
    expanded=False,
):
    st.write(
        "Shows recurring word patterns in low-rated reviews. "
        "Interpret the top terms together to identify possible "
        "customer pain points; the topics are not automatically "
        "verified or labelled business themes."
    )

    show_result_table("negative_review_topics.csv")

# EXAMPLE CUSTOMER REVIEWS

st.subheader("8. Example customer reviews")

if not filtered_absa.empty:
    available_aspects = sorted(
        filtered_absa["aspect"].dropna().unique().tolist()
    )

    selected_aspect = st.selectbox(
        "Choose an aspect",
        ["All aspects"] + available_aspects,
    )

    selected_sentiment = st.selectbox(
        "Choose a sentiment",
        [
            "All sentiments",
            "positive",
            "neutral",
            "negative",
        ],
    )

    examples = filtered_absa.copy()

    if selected_aspect != "All aspects":
        examples = examples[
            examples["aspect"] == selected_aspect
        ].copy()

    if selected_sentiment != "All sentiments":
        examples = examples[
            examples["sentiment"] == selected_sentiment
        ].copy()

    # Attach original review text when IDs are available.
    if (
        "review_id" in examples.columns
        and "review_id" in filtered_df.columns
        and "review_text" in filtered_df.columns
    ):
        original_reviews = filtered_df[
            ["review_id", "review_text"]
        ].copy()

        original_reviews["review_id"] = (
            original_reviews["review_id"].astype(str)
        )

        examples["review_id"] = (
            examples["review_id"].astype(str)
        )

        examples = examples.merge(
            original_reviews.drop_duplicates("review_id"),
            on="review_id",
            how="left",
        )

    display_columns = [
        column
        for column in [
            "app",
            "rating",
            "aspect",
            "sentiment",
            "compound",
            "clause",
            "review_text",
        ]
        if column in examples.columns
    ]

    if not examples.empty and display_columns:
        st.caption(
            "The clause is the text used for aspect-level sentiment "
            "scoring. The original review is shown when available."
        )

        st.dataframe(
            examples[display_columns].head(100),
            use_container_width=True,
            hide_index=True,
        )

    else:
        empty_message(
            "No example reviews match the selected aspect and sentiment."
        )

else:
    empty_message("No example reviews are available for this selection.")

# METHODOLOGY AND LIMITATIONS

with st.expander("Methodology and limitations"):
    st.markdown(
        """
        - **Data source:** Google Play Store reviews collected by the project.
        - **Preprocessing:** Lowercasing, contraction and selected
          abbreviation expansion, URL and HTML/XML removal, whitespace
          normalization, and separate text representations for analysis.
        - **Aspect identification:** Dictionary-based aspect detection
          at clause level.
        - **Sentiment analysis:** VADER compound sentiment scores,
          classified as positive, neutral, or negative.
        - **Stopword removal and lemmatization:** Applied for LDA topic
          modelling rather than to the main sentiment text.
        - **Kruskal–Wallis:** Compares rating distributions across apps.
        - **Chi-square:** Tests association between app and aspect-level
          sentiment categories, with adjusted p-values reported.
        - **LDA topic modelling:** Explores recurring terms in low-rated
          reviews separately for each app.
        - **Limitations:** Keyword matching may miss relevant mentions
          or assign an incorrect aspect. VADER can misinterpret sarcasm,
          context, and mixed-language reviews. LDA topics require manual
          interpretation. Statistical results depend on sample sizes
          and test assumptions.
        """
    )

st.divider()

st.caption(
    "Quick-Commerce ABSA Dashboard | Zepto • Blinkit • Swiggy Instamart"
)
