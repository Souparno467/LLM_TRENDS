# LLM Usage Trends — Exploratory Data Analysis (EDA)

**Dataset:** - https://www.kaggle.com/datasets/souparnogoswami44/llm-trends
**Rows × Columns:** 600 × 30
**Type:** Synthetic / simulated dataset for EDA practice

---

## 1. Why This Topic?

Large Language Models (LLMs) — ChatGPT, Gemini, Claude, Perplexity, Copilot, DeepSeek, Grok, Meta AI, Mistral, Qwen, Cohere — have moved from novelty tools to everyday work software in almost every job function. But adoption isn't uniform: a software engineer, a content writer, and an HR manager don't use the same tool, at the same frequency, for the same reason, or with the same result.

This project exists to practice a **complete, structured EDA workflow** on a realistic, multi-type dataset (categorical, numerical, ordinal, and date fields) that mirrors a genuine business question:

> *"How does LLM adoption, tool choice, usage intensity, and satisfaction vary across job roles, industries, and subscription tiers — and what patterns/outliers should a data team flag before building any predictive or recommendation model on top of this data?"*

It is deliberately built to require every classic EDA technique — missing-value checks, distribution analysis, correlation analysis, group comparisons, and outlier detection — so it functions as a template you can reapply to any similar survey-style dataset.

---

## 2. Target Audience

| Audience | Why this project is useful to them |
|---|---|
| **Data Analytics / Data Science learners** | A guided, section-by-section EDA template (data quality → univariate → bivariate → multivariate → outliers) they can reuse on other datasets. |
| **Students practicing pandas / seaborn / scipy** | Every snippet is small, single-purpose, and runnable independently — good for building muscle memory. |
| **Product / Growth analysts at AI companies** | Illustrates how to slice adoption data by role, tool, and tier — the same lens used for real product analytics (e.g., "which persona underuses our Pro tier?"). |
| **HR / L&D teams** | Demonstrates how to measure AI-tool adoption and training gaps across a workforce. |
| **Anyone preparing this data for machine learning** | The EDA findings (skew, outliers, correlations, encoding needs) directly inform the preprocessing steps required before modeling (e.g., predicting `Would_Recommend` or `Overall_Satisfaction_Score_1to10`). |

---

## 3. Dataset at a Glance

- **Unit of observation:** one row = one professional's LLM usage profile
- **Sheets:**
  - `LLM_Usage_Data` — the 600×30 dataset
  - `Data_Dictionary` — full column definitions and allowed values
  - `Notes` — dataset purpose, quality notes, suggested EDA angles

### Column groups

| Group | Columns |
|---|---|
| **Identity / Demographics** | `User_ID`, `Job_Role`, `Industry`, `Experience_Level`, `Years_of_Experience`, `Age_Group`, `Country`, `Company_Size` |
| **Tool & Subscription** | `Primary_LLM_Tool`, `Primary_Model_Used`, `Secondary_LLM_Tool`, `Secondary_Model_Used`, `Subscription_Type`, `Monthly_Cost_USD` |
| **Usage Behavior** | `Usage_Frequency`, `Hours_Per_Week`, `Adoption_Date`, `Months_Since_Adoption`, `Primary_Use_Case`, `Secondary_Use_Case` |
| **Outcomes** | `Tasks_Automated_Per_Week`, `Productivity_Gain_Percent`, `Time_Saved_Hours_Per_Week`, `Accuracy_Satisfaction_Score_1to10`, `Overall_Satisfaction_Score_1to10`, `Trust_Level_1to10`, `Would_Recommend` |
| **Sentiment / Org Context** | `Main_Concern`, `Team_Also_Uses_LLM`, `Company_Provided_Training` |

**Feature types:**
- **Numerical (continuous/discrete):** `Years_of_Experience`, `Monthly_Cost_USD`, `Hours_Per_Week`, `Months_Since_Adoption`, `Tasks_Automated_Per_Week`, `Productivity_Gain_Percent`, `Time_Saved_Hours_Per_Week`
- **Numerical (ordinal, 1–10 scale):** `Accuracy_Satisfaction_Score_1to10`, `Overall_Satisfaction_Score_1to10`, `Trust_Level_1to10`
- **Categorical (nominal):** `Job_Role`, `Industry`, `Country`, `Primary_LLM_Tool`, `Primary_Model_Used`, `Primary_Use_Case`, `Main_Concern`
- **Categorical (ordinal):** `Experience_Level`, `Age_Group`, `Company_Size`, `Subscription_Type`, `Usage_Frequency`
- **Binary:** `Would_Recommend`, `Team_Also_Uses_LLM`, `Company_Provided_Training`
- **Date:** `Adoption_Date`

---

## 4. Key Statistical Terms Used in This EDA

A quick reference for every technique/metric referenced in the notebook, in plain language.

### 4.1 Data Quality Terms

| Term | Meaning |
|---|---|
| **Missing value** | A cell with no recorded data (`NaN`/`null`). Distorts averages and breaks most ML algorithms if left unhandled. |
| **Encoded missing value** | A "fake" placeholder standing in for a missing value (e.g., `-999`, `"unknown"`, `"N/A"` typed as text) that `isnull()` won't catch — must be checked manually via `value_counts()`. |
| **Duplicate row** | Two or more rows that are byte-for-byte identical across all columns — usually a data-entry or merge error. |
| **Constant column** | A column where every row has the exact same value. Carries zero information for analysis or modeling and can be dropped. |
| **Quasi-constant column** | A column where one value dominates (commonly ≥95% of rows). Rarely useful and often a class-imbalance risk if used as a feature. |
| **ID-like column** | A column where every value is unique (e.g., `User_ID`). Useful for joins/tracking but must be excluded from statistical/model features — it has no predictive signal, only cardinality. |

### 4.2 Descriptive Statistics Terms

| Term | Meaning |
|---|---|
| **Mean** | The arithmetic average. Sensitive to extreme values (outliers pull it up/down). |
| **Median** | The middle value when data is sorted. Robust to outliers — a better "typical value" for skewed data. |
| **Mode** | The most frequently occurring value. Most useful for categorical or discrete numeric data. |
| **Variance** | The average squared distance of each point from the mean — a measure of spread, in *squared* units. |
| **Standard Deviation (Std)** | The square root of variance — spread expressed in the *same unit* as the original data, so it's more interpretable than variance. |
| **Skewness** | Measures asymmetry of a distribution. `0` = symmetric, `>0` = right-tailed (long tail toward high values), `<0` = left-tailed. Guides whether a log/Box-Cox transform is needed before modeling. |
| **Kurtosis** | Measures "tailedness" — how much of the data sits in extreme tails vs. the center, compared to a normal distribution. High kurtosis = more outlier-prone. |
| **Quartiles (Q1, Q2, Q3)** | The values that split sorted data into four equal parts. Q1 = 25th percentile, Q2 = median (50th), Q3 = 75th percentile. |

### 4.3 Correlation Terms

| Term | Meaning |
|---|---|
| **Pearson correlation (r)** | Measures the strength and direction of a *linear* relationship between two numeric variables. Ranges from **−1** (perfect negative) to **+1** (perfect positive); **0** means no linear relationship. |
| **p-value** | The probability of seeing a correlation this strong (or stronger) purely by chance, if no real relationship existed. **p < 0.05** is the conventional threshold for calling a correlation "statistically significant" (i.e., unlikely to be a fluke). |
| **Multicollinearity** | When two or more numeric features are highly correlated with each other (not just the target). Problematic for linear models — it makes coefficients unstable and hard to interpret. |
| **Correlation heatmap** | A color-coded grid of pairwise correlation coefficients across all numeric columns — used to spot strong relationships and multicollinearity at a glance. |
| **Cluster map** | A correlation heatmap where rows/columns are additionally reordered by *hierarchical clustering*, grouping similar variables together — makes hidden variable clusters easier to see. |

### 4.4 Outlier Detection Terms

| Term | Meaning |
|---|---|
| **Outlier** | A data point that lies unusually far from the rest of the distribution — could be a genuine extreme case or a data-entry error. |
| **IQR (Interquartile Range) Method** | `IQR = Q3 − Q1` (the spread of the middle 50% of the data). A point is flagged as an outlier if it falls **below `Q1 − 1.5×IQR`** or **above `Q3 + 1.5×IQR`**. Robust to skewed data because it's based on quartiles, not the mean. This is the same rule that draws the "whiskers" on a boxplot. |
| **Z-score Method** | `Z = (value − mean) / std`. Measures how many standard deviations a point is from the mean. A common rule: **`\|Z\| > 3`** = outlier. **Assumes a roughly normal (bell-curve) distribution** — unreliable on heavily skewed columns, which is why IQR is usually the safer default for real-world business data. |
| **Boxplot** | A visual 5-number summary (min-whisker, Q1, median, Q3, max-whisker) with outlier points plotted individually beyond the whiskers. The fastest way to *see* outliers and spread at once. |

### 4.5 Relationship / Plot Terms

| Term | Meaning |
|---|---|
| **Univariate analysis** | Studying *one variable at a time* — its distribution, center, spread, and shape. |
| **Bivariate analysis** | Studying the *relationship between two variables* — e.g., does `Hours_Per_Week` relate to `Productivity_Gain_Percent`? |
| **Multivariate analysis** | Studying *three or more variables together* — e.g., how `Hours_Per_Week` and `Productivity_Gain_Percent` relate, *split by* `Subscription_Type`. |
| **Histogram** | Shows the distribution/shape of one numeric variable by binning values and counting frequency per bin. |
| **Countplot** | The categorical equivalent of a histogram — bar chart of frequency per category. |
| **Scatter plot** | Plots two numeric variables against each other as points — reveals linear/non-linear relationships and clusters. |
| **Crosstab** | A frequency table of two categorical variables — shows how often each combination occurs. |
| **Grouped statistics (`groupby`)** | Aggregating a numeric column (mean, median, etc.) *within* each category of a categorical column — e.g., average `Productivity_Gain_Percent` per `Primary_LLM_Tool`. |
| **Pair plot** | A grid of scatter plots for every pair of numeric variables, with histograms on the diagonal — the fastest way to eyeball all pairwise relationships at once, optionally colored by a category. |
| **Pivot table heatmap** | A 2D aggregation (e.g., role × tool → average productivity gain) rendered as a heatmap — good for spotting which *combinations* of two categorical variables drive an outcome. |

---

## 5. Methodology / EDA Workflow

The analysis follows a fixed 7-stage pipeline (matches the section order in the companion notebook):

```
1. Setup            → import libraries, set display/plot defaults
2. Load Dataset      → read the Excel sheet into a DataFrame
3. Data Quality       → shape, dtypes, missing values, duplicates, constant/ID-like columns
4. Univariate         → one variable at a time: distribution, center, spread, shape
5. Bivariate          → two variables at a time: numeric-numeric, numeric-categorical, categorical-categorical
6. Multivariate       → 3+ variables together: pairplot, heatmap, clustermap, interaction plots
7. Outlier Detection  → boxplot, IQR method, Z-score method, before/after comparison
```

Each stage ends with a short **Summary** cell — this mirrors how EDA is done in practice: analyze, then immediately write down what it means before moving to the next stage, so insights aren't lost.

---

## 6. Section-by-Section: What to Look For & How to Conclude

> Since the underlying data is **synthetic** (see §8), the exact numbers will vary slightly each time it's regenerated. The guidance below tells you *how to read* each output and what a good conclusion sentence looks like — fill in the actual numbers from your run.

### 6.1 Data Quality Checks
**Look for:** any `missing_percent > 0`, `duplicate rows > 0`, constant/quasi-constant columns, mismatched dtypes.
**Good conclusion looks like:** *"No missing values or duplicate `User_ID`s were found; all columns matched expected dtypes after parsing `Adoption_Date`. The dataset is clean and analysis-ready without imputation."*

### 6.2 Univariate Analysis
**Look for:**
- Which numeric columns are **skewed** (e.g., `Monthly_Cost_USD` — many zeros for free-tier users pull it right-skewed).
- Which categorical columns are **imbalanced** (e.g., is one `Primary_LLM_Tool` dominant?).
- Histogram shapes: normal-ish vs. right-skewed vs. bimodal.

**Good conclusion looks like:** *"`Monthly_Cost_USD` is right-skewed due to the large share of Free-tier users; `Productivity_Gain_Percent` is roughly bell-shaped; `Primary_LLM_Tool` is imbalanced, with ChatGPT as the dominant tool across most roles."*

### 6.3 Bivariate Analysis
**Look for:**
- Pearson `r` and `p` between `Hours_Per_Week` and `Productivity_Gain_Percent` (expect a **positive, significant** relationship — usage intensity was built to drive productivity gain in this synthetic dataset).
- Whether `Overall_Satisfaction_Score_1to10` differs meaningfully by `Subscription_Type` in the boxplots.
- Whether `Primary_LLM_Tool` choice differs by `Job_Role` in the crosstab (e.g., engineers vs. writers).

**Good conclusion looks like:** *"`Hours_Per_Week` shows a moderate-to-strong positive correlation with `Productivity_Gain_Percent` (r≈0.8, p<0.001), confirming usage intensity is the primary driver of self-reported productivity gains. Paid and Enterprise subscribers report higher median satisfaction than Free users."*

### 6.4 Multivariate Analysis
**Look for:**
- In the pairplot, do the point clusters separate cleanly by `Subscription_Type`?
- In the correlation heatmap, are any two *predictor* columns (not just predictor-vs-outcome) strongly correlated — a multicollinearity flag?
- In the role × tool pivot heatmap, which specific combinations show the highest average productivity gain?

**Good conclusion looks like:** *"No strong multicollinearity was found among the numeric predictors. The role × tool heatmap shows Software Engineers using Claude or Copilot report the highest average productivity gains, while Content Writers using ChatGPT cluster in the mid-range."*

### 6.5 Outlier Detection
**Look for:**
- Which columns the **IQR method** flags most (usually `Hours_Per_Week`, `Monthly_Cost_USD`, `Productivity_Gain_Percent` — the widest-spread continuous fields).
- Whether the **Z-score method** agrees with IQR, or disagrees because a column is skewed (Z-score assumes normality, so it under- or over-flags on skewed columns like `Monthly_Cost_USD`).
- How much the mean shifts once outliers are removed from `Productivity_Gain_Percent` — a big shift means outliers matter a lot for that metric.

**Good conclusion looks like:** *"IQR flags a small number of high-end outliers in `Hours_Per_Week` and `Productivity_Gain_Percent` — power users reporting unusually large weekly usage/gains. The Z-score method under-flags `Monthly_Cost_USD` because that column is right-skewed rather than normal, confirming IQR is the more reliable method here."*

---

## 7. Tools & Libraries

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, aggregation (`groupby`, `crosstab`, `pivot_table`) |
| `numpy` | Numeric operations, Z-score calculation |
| `matplotlib` | Base plotting engine |
| `seaborn` | Statistical plots (histplot, boxplot, heatmap, pairplot, clustermap) |
| `scipy.stats` | `pearsonr` for correlation + significance testing |
| `openpyxl` | Reading the `.xlsx` file's multiple sheets |

---

## 8. Limitations

- **Synthetic data:** Rows are generated with role-based weighted randomness, not collected from real survey respondents. Patterns (e.g., engineers preferring Claude/Copilot) are modeled to be *plausible*, not empirically verified.
- **No causal claims:** Correlations found here (e.g., hours used vs. productivity gain) describe the dataset as generated — they should not be read as real-world causal evidence about LLM effectiveness.
- **Point-in-time model/tier mapping:** `Primary_Model_Used` reflects tool-and-tier naming as of mid-2026 and will go stale as vendors ship new model versions.

---

## 9. Suggested Next Steps (Post-EDA)

1. **Encode categoricals** (`Job_Role`, `Primary_LLM_Tool`, etc.) via one-hot or target encoding if building a model.
2. **Address skew** in `Monthly_Cost_USD` / `Productivity_Gain_Percent` with a log or Box-Cox transform if used as model inputs.
3. **Decide on outlier treatment** — cap, transform, or keep, depending on whether the downstream model is sensitive to extreme values (e.g., linear regression vs. tree-based models).
4. **Feature engineering** — e.g., `Hours_Per_Week × Months_Since_Adoption` as a cumulative-usage proxy.
5. **Modeling target ideas** — predict `Would_Recommend` (classification) or `Overall_Satisfaction_Score_1to10` (regression/ordinal).

---

