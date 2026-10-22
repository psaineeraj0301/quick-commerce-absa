# Quick-Commerce App Review Analysis
### Aspect-Based Sentiment Analysis: Blinkit vs Zepto vs Swiggy Instamart

## 1. Project Overview

This project analyses customer reviews of three quick-commerce applications — **Blinkit, Zepto, and Swiggy Instamart** — using Natural Language Processing (NLP).

The project applies aspect-based sentiment analysis to identify customer opinions about specific service areas, compare rating distributions, discover recurring complaints in low-rated reviews, and present the results through an interactive Streamlit dashboard.

The goal is to understand what customers appreciate, what frustrates them, and which aspects of the customer experience may require improvement.

## 2. Objectives

- Compare customer ratings across Blinkit, Zepto, and Swiggy Instamart.
- Identify positive, neutral, and negative sentiment for different service aspects.
- Understand the aspects customers mention most frequently.
- Compare sentiment distributions across applications.
- Test whether rating distributions differ significantly between applications.
- Identify recurring themes in low-rated reviews using topic modelling.
- Present findings through an interactive dashboard to support business decisions.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas and NumPy | Data cleaning and analysis |
| NLTK and VADER | Rule-based sentiment analysis |
| Emoji | Emoji conversion during preprocessing |
| SciPy | Statistical testing |
| Scikit-learn | Text vectorisation and topic modelling utilities, where used |
| Google Play Scraper | Review collection |
| Plotly | Interactive charts |
| Matplotlib and Seaborn | Visualisation, where used |
| Streamlit | Interactive dashboard |

## 4. Customer Aspects Analysed

The project analyses ten aspects of the quick-commerce customer experience:

1. **App experience** — usability and application-related issues.
2. **Customer support** — help and complaint resolution.
3. **Delivery partner** — customer experiences with delivery personnel.
4. **Delivery speed** — timeliness of deliveries.
5. **Offers and discounts** — promotions and discount-related experiences.
6. **Order accuracy and packaging** — correct items and packaging quality.
7. **Pricing and fees** — prices and additional charges.
8. **Product availability** — availability of requested products.
9. **Product freshness and quality** — condition and quality of delivered products.
10. **Refunds and payments** — payment issues and refund experiences.

## 5. Project Workflow

1. **Data collection:** Collect customer reviews and star ratings from Google Play.
2. **Preprocessing:** Clean text, remove duplicates, handle missing values, normalise text, and prepare separate text representations for analysis.
3. **Aspect identification:** Use a keyword dictionary to detect aspect-related clauses.
4. **Sentiment analysis:** Apply VADER compound sentiment scores to classify aspect mentions as positive, neutral, or negative.
5. **Exploratory analysis:** Examine star ratings, aspect frequencies, and sentiment distributions.
6. **Statistical testing:** Compare app rating distributions and test associations between apps and aspect-level sentiment.
7. **Topic modelling:** Use Latent Dirichlet Allocation (LDA) to explore recurring word patterns in low-rated reviews.
8. **Dashboard development:** Display the analysis through interactive charts, tables, and filters.

## 6. Interactive Dashboard

The Streamlit dashboard provides an overview of customer feedback and allows users to filter results by application and minimum and maximum star ratings.

### Dashboard components

- **Overview metrics:** Reviews analysed, average star rating, percentage of 1–2-star reviews, and aspect mentions.
- **Star-rating distribution:** Compares the number of reviews at each rating from one to five stars.
- **Customer sentiment by aspect:** Compares positive, neutral, and negative sentiment percentages for each app.
- **Net sentiment heatmap:** Displays positive sentiment percentage minus negative sentiment percentage for each app and aspect.
- **Aspect mention frequency:** Shows which customer-experience aspects are discussed most frequently.
- **Sentiment distribution by aspect:** Displays counts of positive, neutral, and negative aspect mentions.
- **Statistical test results:** Presents the Kruskal–Wallis test, chi-square test of independence, and LDA topic modelling results.
- **Methodology and limitations:** Explains the analytical approach and important caveats.

## 7. Results Shown in the Dashboard

The following figures are from the dashboard screenshots provided for this project. They represent the displayed analysis and may change when filters or underlying data change.

### 7.1 Dataset overview

| Metric | Dashboard value |
|---|---:|
| Reviews analysed | 5,827 |
| Average star rating across the filtered dataset | 3.90 / 5 |
| 1–2-star reviews | 24.0% |
| Aspect mentions | 5,683 |

The dashboard shows that the analysis covers thousands of reviews across the three applications. The average rating and low-rating percentage summarise the selected dataset under the active filters.

### 7.2 Star-rating distribution

The rating distribution chart compares one-star through five-star reviews for Blinkit, Zepto, and Swiggy Instamart.

The displayed chart shows substantial differences in the number of low-rated reviews across applications. Zepto has a visibly larger number of one-star reviews, while five-star reviews form the largest rating group for all three applications.

These patterns are descriptive. Statistical testing is required to assess whether rating distributions differ beyond what might be expected by chance.

### 7.3 Customer sentiment by aspect

The dashboard compares positive, neutral, and negative sentiment percentages across the ten aspects for each application.

The chart helps identify aspects that may be associated with customer satisfaction or dissatisfaction. However, sentiment percentages should be interpreted alongside the number of mentions for each aspect, because some aspects may have fewer observations than others.

### 7.4 Net sentiment heatmap

Net sentiment is calculated as:

**Net sentiment = Positive sentiment percentage − Negative sentiment percentage**

A positive value means positive mentions exceed negative mentions. A negative value means negative mentions exceed positive mentions.

The heatmap displays the following values:

| Customer aspect | Blinkit | Zepto | Swiggy Instamart |
|---|---:|---:|---:|
| App experience | 53.7 | 1.6 | 39.1 |
| Customer support | -10.1 | -32.3 | -30.1 |
| Delivery partner | 2.8 | -29.6 | -21.2 |
| Delivery speed | 24.6 | -3.0 | 17.2 |
| Offers and discounts | 16.7 | 24.7 | 44.7 |
| Order accuracy and packaging | 0.0 | -34.5 | -27.5 |
| Pricing and fees | -11.6 | 11.0 | 19.1 |
| Product availability | -8.8 | -32.7 | 6.5 |
| Product freshness and quality | -11.6 | -26.0 | 3.3 |
| Refunds and payments | -27.7 | -37.0 | -27.2 |

*Values are percentage-point differences, as displayed in the dashboard.*

The displayed results suggest that customer support and refunds/payments are potential pain points across multiple apps. Offers and discounts have positive net sentiment for all three applications. These findings reflect detected review mentions, not a direct measurement of overall service quality.

### 7.5 Most-discussed customer aspects

The aspect-frequency chart shows how often customers discuss each service area.

Delivery speed has the highest number of mentions in the displayed chart, particularly for Zepto. App experience and order accuracy/packaging are also frequently discussed.

This indicates which topics receive the most attention in the analysed reviews. High mention frequency alone does not mean that an aspect has positive or negative sentiment.

## 8. Statistical Analysis

### 8.1 Kruskal–Wallis rating comparison

The Kruskal–Wallis test compares rating distributions across the three applications.

The dashboard displays:

| Statistic | Value |
|---|---:|
| H statistic | 145.6967 |
| p-value | Approximately 2 × 10⁻³² |
| Significance level (α) | 0.05 |
| Epsilon-squared effect size | 0.0247 |

The displayed p-value is below 0.05, indicating a statistically significant difference in rating distributions across the applications.

The epsilon-squared value indicates a relatively small effect size. Statistical significance therefore does not imply that the practical difference is large.

### 8.2 Chi-square test of independence

The chi-square test examines whether the distribution of positive, neutral, and negative sentiment for an aspect is associated with the application.

The dashboard reports adjusted p-values using the Bonferroni correction and includes Cramér's V, minimum expected cell count, expected-count warnings, and interpretations.

Based on the displayed adjusted results, the following aspects are statistically significant at the 0.05 level:

| Aspect | Chi-square statistic | Adjusted p-value | Cramér's V |
|---|---:|---:|---:|
| App experience | 74.0794 | Approximately 3 × 10⁻¹⁵ | 0.1877 |
| Delivery speed | 38.5156 | 0.0000008 | 0.1166 |
| Order accuracy and packaging | 23.4816 | 0.0010 | 0.1272 |
| Product availability | 20.8723 | 0.0034 | 0.1419 |

These results indicate that the detected sentiment distributions for these aspects differ by application in the analysed dataset.

Cramér's V provides information about association strength. Statistical significance should be interpreted together with this effect-size measure and the expected cell counts.

### 8.3 Topic modelling of low-rated reviews

The dashboard uses Latent Dirichlet Allocation (LDA) to explore recurring word patterns in low-rated reviews, separately by application.

The displayed topic terms include words associated with customer service, orders, delivery, refunds, products, and app experience.

LDA identifies patterns of words rather than verified business categories. The topics must be interpreted manually, and repetitive or mixed-language terms may reduce their clarity.

## 9. Key Business Insights

Based on the displayed dashboard results:

- **Delivery speed deserves attention:** It is the most frequently mentioned aspect in the displayed aspect-frequency chart.
- **Customer support may need improvement:** Net sentiment is negative for all three applications.
- **Refunds and payments are a shared concern:** The displayed net sentiment values are negative across all applications.
- **App experience is a differentiating aspect:** The chi-square test indicates significant differences in its sentiment distribution by application.
- **Offers and discounts receive favourable sentiment:** Net sentiment is positive for all three applications.
- **Order accuracy and packaging merit investigation:** Zepto and Swiggy Instamart show negative net sentiment values for this aspect.
- **Ratings differ statistically:** The Kruskal–Wallis test indicates a significant difference in rating distributions, with a relatively small effect size.

These insights are based on the available review sample and the project's dictionary-based aspect detection and VADER sentiment classification. They should not be treated as proof of causal relationships.

## 10. Business Recommendations

1. **Improve delivery reliability:** Investigate late deliveries and recurring complaints about delivery timing.
2. **Strengthen customer support:** Review complaint resolution time, escalation processes, and customer communication.
3. **Improve refund transparency:** Make refund status, timelines, and payment-failure resolution easier to understand.
4. **Reduce order errors:** Improve picking accuracy, item verification, and packaging checks.
5. **Monitor app experience:** Investigate usability issues and recurring technical complaints.
6. **Maintain attractive promotions:** Evaluate offers based on customer response and business profitability.
7. **Track aspect-level metrics:** Monitor sentiment and mention volume over time rather than relying only on overall star ratings.

## 11. Methodology and Limitations

- **Data source:** Google Play Store reviews collected by the project.
- **Preprocessing:** Lowercasing, contraction and selected abbreviation expansion, URL and HTML/XML removal, whitespace normalisation, and separate text representations for analysis.
- **Aspect identification:** Dictionary-based aspect detection at clause level.
- **Sentiment analysis:** VADER compound sentiment scores classified as positive, neutral, or negative.
- **Stopword removal and lemmatisation:** Applied for LDA topic modelling rather than the main sentiment text.
- **Kruskal–Wallis:** Compares rating distributions across applications.
- **Chi-square:** Tests associations between application and aspect-level sentiment categories, with adjusted p-values reported.
- **LDA topic modelling:** Explores recurring terms in low-rated reviews separately for each application.

### Important limitations

- Keyword matching may miss relevant mentions or assign a clause to an incorrect aspect.
- VADER can misinterpret sarcasm, context, and mixed-language reviews.
- LDA topics require manual interpretation and are not automatically validated business themes.
- Statistical results depend on sample sizes and test assumptions.
- Google Play reviews may not represent every customer or every user of the applications.
- Results are descriptive and associational; they do not establish that an app feature caused a rating or sentiment outcome.

## 12. Future Scope

- Use transformer-based models for aspect extraction and sentiment classification.
- Improve support for multilingual and Romanized Hindi reviews.
- Conduct manual annotation and evaluate precision, recall, and F1-score.
- Compare model predictions with human-labelled results.
- Analyse changes in sentiment over time.
- Expand the dataset and compare additional quick-commerce services.

## 13. Repository Structure

```text
quick-commerce-absa/
├── config.py
├── collect_reviews.py
├── preprocess.py
├── pipeline.py
├── dashboard.py
├── requirements.txt
├── README.md
├── data/
└── outputs/
```

The exact files and generated outputs included in the repository may vary. Large raw datasets and private information should not be committed.

## 14. Installation and Usage

### Clone the repository

```bash
git clone https://github.com/psaineeraj0301/quick-commerce-absa.git
cd quick-commerce-absa
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Run the pipeline

```bash
python pipeline.py
```

Run the pipeline before starting the dashboard if the dashboard requires generated analysis files.

### Launch the dashboard

```bash
python -m streamlit run dashboard.py
```

Open the local URL displayed in the terminal, typically `http://localhost:8501`.

**Note:** The commands assume the required source files and dependencies are included and that input datasets are available. If datasets or outputs are excluded from GitHub, regenerate them locally or provide suitable setup instructions.



