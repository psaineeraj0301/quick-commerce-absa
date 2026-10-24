# Quick-Commerce App Review Analysis
### Aspect-Based Sentiment Analysis: Blinkit vs Zepto vs Swiggy Instamart

## 1. Project Overview

This project analyses customer reviews of three quick-commerce applications — **Blinkit, Zepto, and Swiggy Instamart** — using Natural Language Processing (NLP).

The project implements an Aspect-Based Sentiment Analysis (ABSA) pipeline to identify customer opinions about specific service areas, compare star-rating distributions, examine sentiment differences between applications, and explore recurring themes in low-rated reviews.

Customer reviews are collected from the Google Play Store using Python. The collected data is processed and analysed, and the results are presented through an interactive Streamlit dashboard.

The project aims to understand:
- What customers appreciate about each application.
- Which service aspects customers discuss most frequently.
- Which aspects receive more positive or negative feedback.
- Whether rating and aspect-level sentiment distributions differ between applications.
- What recurring word patterns appear in low-rated reviews.

The findings can help businesses identify areas that may require further investigation and improvement.

## 2. Objectives

- Collect customer reviews and star ratings for Blinkit, Zepto, and Swiggy Instamart.
- Clean and prepare the collected review data for analysis.
- Identify positive, neutral, and negative sentiment for different service aspects.
- Determine which customer-experience aspects are mentioned most frequently.
- Compare star-rating distributions across the three applications.
- Examine associations between applications and aspect-level sentiment categories.
- Explore recurring themes in low-rated reviews using topic modelling.
- Visualise the results through an interactive Streamlit dashboard.
- Provide evidence-based business insights and recommendations.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data cleaning, manipulation, and analysis |
| NumPy | Numerical operations |
| Google Play Scraper | Collection of Google Play Store reviews |
| Regular Expressions (`re`) | Text cleaning, pattern matching, and aspect detection |
| Emoji | Conversion of emojis into descriptive text during topic preparation |
| NLTK | Sentence tokenisation, stop-word handling, lemmatisation, and sentiment analysis |
| VADER | Rule-based sentiment scoring |
| SciPy | Kruskal–Wallis and chi-square statistical tests |
| Statsmodels | Multiple-testing correction using the Bonferroni method |
| Scikit-learn | Count vectorisation and Latent Dirichlet Allocation |
| Matplotlib | Statistical and analytical visualisations |
| Seaborn | Heatmaps and statistical visualisations |
| Plotly | Interactive dashboard charts |
| Streamlit | Interactive web dashboard |

## 4. Customer Aspects Analysed

The project analyses ten aspects of the quick-commerce customer experience.

1. **App Experience** — application usability, navigation, loading, errors, and other technical issues.
2. **Customer Support** — customer service, complaint handling, and support responses.
3. **Delivery Partner** — customer experiences with delivery personnel and their behaviour.
4. **Delivery Speed** — delivery timing, delays, waiting time, and delivery speed.
5. **Offers and Discounts** — promotional offers, coupons, cashback, and discounts.
6. **Order Accuracy and Packaging** — incorrect or missing items, order completeness, and packaging-related issues.
7. **Pricing and Fees** — product prices, additional charges, delivery fees, and perceived value.
8. **Product Availability** — product availability, stock-related issues, and substitutions.
9. **Product Freshness and Quality** — freshness, damaged products, expiry-related complaints, and product quality.
10. **Refunds and Payments** — payment failures, transactions, refunds, and payment-related complaints.

Aspect identification uses predefined keyword and regular-expression patterns. Therefore, the results depend on the coverage and accuracy of these patterns.

## 5. Project Workflow

The project follows an end-to-end review-analysis workflow.

1. **Data Collection:** Collect reviews and star ratings from Google Play Store listings.
2. **Data Preprocessing:** Handle missing values, remove duplicates, validate ratings, normalise text, and prepare separate text representations.
3. **Sentence and Clause Processing:** Split reviews into sentences and clauses to analyse specific portions of customer feedback.
4. **Aspect Identification:** Detect service aspects using predefined keyword patterns.
5. **Sentiment Analysis:** Apply VADER to calculate sentiment scores and classify aspect-related clauses as positive, neutral, or negative.
6. **Exploratory Analysis:** Examine rating distributions, aspect mention frequencies, and sentiment percentages.
7. **Statistical Testing:** Use the Kruskal–Wallis test to compare ratings and the chi-square test of independence to examine associations between applications and sentiment categories.
8. **Topic Modelling:** Use Latent Dirichlet Allocation (LDA) to explore recurring word patterns in low-rated reviews.
9. **Visualisation:** Generate analytical charts and summary outputs.
10. **Dashboard Development:** Present the results through an interactive Streamlit dashboard.

## 6. Data Collection and Preprocessing

### 6.1 Data Collection

The project collects reviews from the following Google Play Store applications:

| Application | Google Play Package ID |
|---|---|
| Blinkit | `com.grofers.customerapp` |
| Zepto | `com.zeptoconsumerapp` |
| Swiggy Instamart | `in.swiggy.android.instamart` |

The current configuration sets a target of **up to 10,000 reviews per application**, with a batch size of 500 reviews.

The collection script:
- Retrieves reviews using the Google Play Scraper library.
- Uses the English-language setting and India as the country.
- Sorts reviews by newest first.
- Checks for duplicate review identifiers during collection.
- Handles repeated collection errors and pauses between requests.
- Saves the collected data to a CSV file.

The target is not a guarantee that exactly 10,000 reviews will be collected for each application. Actual counts depend on review availability, pagination, and collection errors.

### 6.2 Data Preprocessing

The preprocessing script prepares the collected reviews for analysis.

The main operations include:

- Removing reviews with missing or empty text.
- Converting ratings to numeric values and retaining valid ratings from 1 to 5.
- Removing duplicate reviews.
- Filtering reviews based on the proportion of Devanagari characters.
- Converting text to lowercase.
- Removing URLs, email addresses, and HTML/XML markup.
- Expanding selected contractions and abbreviations.
- Normalising whitespace.
- Creating separate text columns for sentiment analysis and topic modelling.
- Standardising review dates where available.
- Creating variables such as review length and low-rating status.
- Producing data-quality summaries.

Two text representations are prepared:

| Column | Purpose |
|---|---|
| `text_clean` | Used for aspect detection and VADER sentiment analysis. It preserves punctuation, numbers, and emojis. |
| `text_topic` | Used for topic modelling. Emojis are converted into descriptive text, and punctuation and special characters are removed. |

The topic-modelling pipeline also applies its own text preparation, including stop-word handling and lemmatisation.

The preprocessing process does not use a separate general-purpose word-tokenisation pipeline for all sentiment text. Sentence tokenisation is performed later during aspect-level analysis.

## 7. Interactive Dashboard

The Streamlit dashboard presents the results of the review analysis through charts, summary metrics, tables, and statistical outputs.

### Dashboard Components

- **Overview Metrics:** Summarises the number of reviews, average rating, low-rated reviews, and aspect mentions.
- **Star-Rating Distribution:** Compares the number of one-star to five-star reviews across the applications.
- **Customer Sentiment by Aspect:** Displays positive, neutral, and negative sentiment percentages for different service aspects.
- **Net Sentiment Heatmap:** Compares net sentiment values across applications and aspects.
- **Aspect Mention Frequency:** Identifies the service areas discussed most frequently.
- **Sentiment Distribution by Aspect:** Shows the distribution of sentiment categories for the identified aspects.
- **Monthly Review Volume:** Displays review counts grouped by month when usable review-date information is available.
- **Statistical Test Results:** Presents the rating comparison and aspect-level chi-square analysis.
- **Topic Modelling Results:** Displays topic-modelling outputs for low-rated reviews.
- **Methodology and Limitations:** Provides information about the analytical approach and important limitations.

The dashboard includes filters for application and minimum and maximum star ratings.

Dashboard metrics can change when filters are applied or when the underlying dataset is updated.

## 8. Statistical Analysis and Net Sentiment

### 8.1 Kruskal–Wallis Test

The Kruskal–Wallis test is used to compare star-rating distributions across Blinkit, Zepto, and Swiggy Instamart.

The test evaluates whether the rating distributions differ significantly across the three independent application groups.

The analysis uses:
- The H statistic.
- The p-value.
- A significance level of 0.05.
- An epsilon-squared effect-size measure.

**Interpretation:** If the p-value is below 0.05, the test provides evidence of a statistically significant difference between at least two application groups. It does not, by itself, identify which specific applications differ.

### 8.2 Chi-Square Test of Independence

The chi-square test examines whether the distribution of positive, neutral, and negative sentiment categories is associated with the application for a particular aspect.

The analysis includes:
- Chi-square statistics.
- p-values.
- Bonferroni-adjusted p-values for multiple comparisons.
- Cramér's V as an effect-size measure.
- Expected cell counts and related warnings.

A p-value below the selected significance threshold indicates evidence of an association between the application and the sentiment-category distribution for the aspect being tested.

The results should be interpreted alongside effect sizes and expected cell counts. Where expected counts are too small, the chi-square approximation may be unreliable.

### 8.3 Net Sentiment

Net sentiment summarises the difference between positive and negative aspect-level sentiment percentages.

**Formula:**

`Net Sentiment = Positive Sentiment (%) − Negative Sentiment (%)`

For example, if 60% of aspect mentions are positive and 25% are negative:

`Net Sentiment = 60 − 25 = +35 percentage points`

Interpretation:
- **Positive value:** Positive mentions exceed negative mentions.
- **Negative value:** Negative mentions exceed positive mentions.
- **Zero:** Positive and negative mentions are equal.

Net sentiment is expressed in percentage points. It is not the same as average star rating and does not directly measure overall customer satisfaction.

### 8.4 Topic Modelling of Low-Rated Reviews

Latent Dirichlet Allocation (LDA) is used to explore recurring word patterns in low-rated reviews.

The current pipeline configuration specifies five topics and up to ten displayed words per topic.

Topic modelling involves:
- Selecting low-rated reviews using the configured rating threshold.
- Preparing text for topic modelling.
- Converting the text into numerical word-count features.
- Applying LDA to identify groups of words that occur together.
- Presenting topic terms for interpretation.

The resulting word groups may help identify recurring complaint patterns. However, LDA topics are not automatically verified business categories and require manual interpretation.

## 9. Key Business Insights

The analysis can help businesses investigate several areas of customer experience.

- **Delivery Experience:** Review the frequency and sentiment of comments about delivery speed and delays.
- **Customer Support:** Examine recurring complaints about response times, complaint handling, and resolution.
- **Refunds and Payments:** Investigate negative feedback about refunds, payment failures, and transaction issues.
- **Order Accuracy and Packaging:** Examine comments about incorrect orders, missing items, and packaging quality.
- **Product Availability:** Investigate complaints about unavailable products and substitutions.
- **Product Freshness and Quality:** Review feedback about damaged, expired, or poor-quality products.
- **Offers and Discounts:** Examine customer responses to promotional offers and discount-related experiences.
- **Application Experience:** Identify recurring usability, navigation, loading, and technical complaints.

These are areas for investigation rather than proof of operational failure. The priority of each issue should be determined from the latest analysis outputs, mention counts, sentiment percentages, statistical results, and relevant business evidence.

## 10. Business Recommendations

The following recommendations describe potential actions informed by the types of results produced by the project. Their priority should be based on the actual output generated from the current dataset.

### 10.1 Improve Delivery Reliability

Examine recurring complaints about delivery speed, delays, waiting time, and communication about late orders.

If supported by the evidence, investigate delivery processes and customer updates about delays.

### 10.2 Strengthen Customer Support

Review feedback about support responses, complaint handling, and issue resolution.

Where recurring problems are identified, consider improving escalation processes and communicating resolution progress more clearly.

### 10.3 Improve Refund and Payment Transparency

Investigate complaints about unsuccessful payments, refunds, and transaction status.

Potential actions include clearer refund-status information, understandable timelines, and improved communication about payment failures.

### 10.4 Reduce Order Errors

Review comments about incorrect or missing items and packaging problems.

If these complaints recur, investigate item-picking accuracy, order verification, and packaging procedures.

### 10.5 Monitor Product Availability and Quality

Examine feedback about unavailable items, substitutions, freshness, and product condition.

Use these findings to identify areas requiring further review of stock accuracy and product-handling processes.

### 10.6 Review Pricing and Promotions

Investigate customer comments about additional fees, pricing clarity, discounts, and promotional conditions.

Evaluate promotions alongside profitability and other business measures rather than relying only on positive sentiment.

### 10.7 Monitor Customer Experience Over Time

Run the analysis regularly and track aspect mention counts, negative sentiment percentages, and net sentiment.

Changes in these measures can help identify issues requiring further investigation. Comparisons over time are most useful when the collection periods and sampling approach are documented consistently.

### 10.8 Prioritise Actions Using Multiple Measures

When deciding which issues to address first, consider:
- Frequency of mentions.
- Percentage of negative sentiment.
- Severity and recurrence of complaints.
- Statistical evidence and effect sizes.
- Feasibility and cost of improvement.
- Whether the pattern persists in subsequent data.

Customer reviews can identify potential problems, but operational data is needed to verify their causes.

## 11. Methodology and Limitations

### Methodology

- **Data Source:** Google Play Store reviews collected using the Google Play Scraper library.
- **Collection Target:** Up to 10,000 reviews per application, as configured in `config.py`.
- **Preprocessing:** Missing-value handling, duplicate removal, rating validation, text normalisation, contraction and abbreviation expansion, and separate text representations.
- **Aspect Identification:** Keyword- and regular-expression-based aspect detection at clause level.
- **Sentiment Analysis:** VADER compound scores classify aspect-related clauses as positive, neutral, or negative.
- **Statistical Analysis:** Kruskal–Wallis rating comparison and chi-square tests of independence.
- **Multiple-Testing Correction:** Bonferroni adjustment for the relevant chi-square comparisons.
- **Topic Modelling:** LDA explores recurring word patterns in low-rated reviews.
- **Visualisation:** Matplotlib, Seaborn, Plotly, and Streamlit support analytical charts and interactive exploration.

### Limitations

- **Review Coverage:** Google Play reviews may not represent all customers or users of the applications.
- **Collection Constraints:** The number of reviews collected may be lower than the configured target.
- **Aspect Detection:** Keyword-based matching can miss relevant feedback or assign clauses to an incorrect aspect.
- **Broad Keywords:** Generic terms may match reviews that do not meaningfully discuss the intended aspect.
- **Sentiment Classification:** VADER may misinterpret sarcasm, context, mixed sentiment, and multilingual expressions.
- **Language Filtering:** Filtering by script does not guarantee that all remaining reviews are written entirely in English.
- **Topic Interpretation:** LDA identifies word patterns that require manual interpretation and may not correspond to clear business themes.
- **Statistical Assumptions:** Statistical conclusions depend on the data, sample sizes, and test assumptions.
- **Changing Data:** Review samples and dashboard values may change when the collection and analysis pipeline is rerun.
- **No Causal Conclusions:** Associations in review data do not establish that a particular service feature caused a rating or sentiment outcome.
- **Limited Validation:** Dictionary-based aspect identification and rule-based sentiment analysis should be evaluated against manually labelled reviews before their accuracy is assumed.

## 12. Future Scope

Possible extensions include:

- Use transformer-based models for aspect extraction and sentiment classification.
- Improve support for multilingual and Romanized Hindi reviews.
- Create a manually labelled validation dataset.
- Evaluate aspect detection using precision, recall, and F1-score.
- Evaluate sentiment predictions against human-labelled results.
- Compare rule-based and machine-learning approaches.
- Improve keyword dictionaries to reduce incorrect aspect matches.
- Analyse sentiment changes across consistently defined time periods.
- Investigate additional quick-commerce applications.
- Connect customer-review findings with operational data, where available.
- Improve topic interpretability and validate discovered themes with manual review.

## 13. Repository Structure

```text
quick-commerce-absa/
│
├── config.py
├── collect_reviews.py
├── preprocess.py
├── pipeline.py
├── dashboard.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw_reviews.csv
│   └── clean_reviews.csv
│
└── outputs/
    ├── aspect_mentions.csv
    └── generated analysis outputs
```

## 14. Installation and Usage

### 14.1 Create a virtual environment

```bash
python -m venv .venv
```

### 14.2 Install dependencies

```bash
python -m pip install -r requirements.txt
```
### 14.3 Run the pipeline

```bash
python pipeline.py
```

### 14.4 Launch the dashboard

```bash
python -m streamlit run dashboard.py
```
