# BCG X Customer Churn Prediction

Machine learning and generative AI solution for predicting customer churn in telecommunications. Developed for BCG X competition at University of Bologna.

## Overview

This project addresses customer churn for ABC TelCo by combining traditional ML with generative AI. The solution analyzes 7,043 customer records and 1,605 complaint texts to predict churn and provide actionable retention strategies.

**Best Model:** Lasso Regression (18.46% misclassification error)

## Challenge

**Data:**
- Structured: customer demographics, services, billing, contract info
- Unstructured: 1,605 customer complaints (text)

**Objective:** Predict customers at risk of churning and extract insights from complaints.

## Solution Approach

### Machine Learning
- Logistic Regression, Lasso, Gradient Boosting, Random Forest
- Feature engineering: phonelines, internet, streaming variables
- 5-fold cross-validation

### Generative AI
- Sentiment analysis using Bing lexicon
- Topic modeling with Latent Semantic Analysis (54 topics explain 70% variance)
- Customer-level sentiment aggregation

### Business Intelligence
- K-Means customer segmentation with PCA
- Prioritization model combining potential and timing

## Key Results

| Model | Error Rate |
|-------|-----------|
| **Lasso** | **18.46%** |
| Logistic | 18.58% |
| GBM | 18.70% |
| Random Forest | 18.93% |

**Churn Drivers:**
- Contract type (longer = lower churn)
- Tenure (established customers stay)
- Internet type (fiber optic increases churn)
- Payment method (electronic check increases churn)
- Negative sentiment and complaint count

## Project Structure

```
bcg-churn-prediction/
├── README.md
├── code/
│   ├── functions.R              # Reusable functions
│   ├── bcg_churn_analysis.R     # Complete analysis
│   └── final_report.Rmd         # RMarkdown report
├── data/
│   ├── ABCTelco_clients.csv
│   └── ABCTelco_complaints.csv
├── report/
│   └── final-report.pdf
└── .gitignore
```

## Setup

### Prerequisites
- R 4.0+
- RStudio (recommended)

### Install Packages

```r
install.packages(c(
  "tidyverse", "caret", "glmnet", "randomForest", "gbm",
  "tidytext", "tm", "lsa", "DescTools", "e1071"
))
```

## Usage

### Option 1: Run Complete Analysis

```r
# Set working directory
setwd("/path/to/bcg-churn-prediction")

# Run full analysis
source("code/bcg_churn_analysis.R")
```

This will:
1. Load and preprocess both datasets
2. Perform correlation and feature engineering
3. Execute sentiment analysis on complaints
4. Train 4 ML models with 5-fold CV
5. Print performance comparison table

### Option 2: Use Individual Functions

```r
# Load custom functions
source("code/functions.R")

# Convert variables
data <- convert_chr_to_factor(data)

# Calculate associations
assoc_matrix <- VCramers(data)

# Create features
data <- create_phonelines(data)
data <- create_internet(data)
data <- create_streaming(data)
```

### Option 3: Render Report

```r
rmarkdown::render("code/final_report.Rmd")
```

## Methodology

**Data Preprocessing:**
- Format conversion, NA removal (11 observations)
- Correlation analysis (Pearson, Cramér's V)
- Feature engineering

**Sentiment Analysis Pipeline:**
1. Tokenization and stopword removal
2. Bing lexicon sentiment scoring
3. Aggregation by customerID (score, count, weighted score)

**Model Training:**
- 5-fold cross-validation
- Hyperparameter tuning for GBM
- Evaluation metric: misclassification error

**Segmentation:**
- K-Means clustering with business steering
- PCA dimensionality reduction
- Two-step aggregation (granular → macro segments)

## Key Functions

From `functions.R`:

```r
# Data preprocessing
convert_chr_to_factor(data)
VCramers(data)  # Cramér's V association matrix

# Feature engineering
create_phonelines(data)
create_internet(data)
create_streaming(data)

# Sentiment analysis
tokenize_text(data)
remove_stopwords(data)
add_sentiment(data)
calculate_sentiment_score(data)
```

## Technologies

**R Packages:** tidyverse, caret, glmnet, randomForest, gbm, tidytext, tm, lsa, cluster, e1071

**Methods:** Classification, clustering, NLP, cross-validation, hyperparameter tuning

## Authors

**Group 9 - Six Sigma:**  
Egle Caudullo, Andrea Bianco, Federico Baio, Alessio Pistone, Alice Bianchi, Festus Udealor

**Competition:** BCG X @ University of Bologna, April 2024

## References

1. Boston Consulting Group (2024). BCG X Presentation - Customer Segmentation & Prioritization.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning.
3. Silge, J., & Robinson, D. (2017). Text Mining with R: A Tidy Approach.

## License

MIT License

## Contact

**Egle Caudullo**  
Email: egle.caudullo.22@gmail.com  
GitHub: [@Egle22](https://github.com/Egle22)  
LinkedIn: [Egle Caudullo](https://www.linkedin.com/in/egle-caudullo-177993266)
