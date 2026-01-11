
# BCG X Customer Churn Prediction - ABC TelCo

Machine learning and generative AI solution for predicting and preventing customer churn in telecommunications, combining structured customer data analysis with unstructured complaint text mining.

## Table of Contents

- [Overview](#overview)
- [Challenge](#challenge)
- [Solution Approach](#solution-approach)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Authors](#authors)

## Overview

This project was developed for the BCG X competition at University of Bologna, addressing customer churn for ABC TelCo, a fictional telecommunications company. The solution combines traditional machine learning with generative AI to predict churn and provide actionable business insights.

Customer churn is a critical concern that impacts revenue and increases acquisition costs. This project demonstrates how data-driven approaches can identify at-risk customers and suggest targeted retention strategies.

## Challenge

ABC TelCo provided two types of data:

**Structured Data:**
- Customer demographics (7,043 customers)
- Subscription plans and services
- Usage patterns and activity metrics
- Contract information and billing data

**Unstructured Data:**
- Customer complaints (1,605 complaints)
- Text-based feedback from multiple channels

**Objective:** Build a predictive system that identifies customers at risk of churning and extracts insights from complaints to inform retention strategies.

## Solution Approach

### 1. Machine Learning Component

- Predictive modeling using structured customer data
- Feature engineering (phonelines, internet, streaming variables)
- Multiple classification models tested:
  - Logistic Regression
  - Lasso Regression (best performer: 18.46% error)
  - Gradient Boosting (with hyperparameter tuning)
  - Random Forest
- 5-fold cross-validation for robust evaluation

### 2. Generative AI Component

- Sentiment analysis on customer complaints using Bing lexicon
- Text preprocessing: tokenization, stopword removal, stemming
- Topic modeling using Latent Semantic Analysis (LSA)
- Sentiment scoring: positive vs negative word counts
- Aggregated customer-level metrics: score, complaint count, weighted score

### 3. Business Intelligence

- Customer segmentation using K-Means clustering
- Bottom-up segmentation with feature scaling and PCA
- Prioritization model combining potential-based and timing-based approaches
- Actionable insights for retention campaigns

## Methodology

### Data Preprocessing

**Clients Dataset:**
- Converted character variables to factors
- Removed 11 observations with missing TotalCharges
- Analyzed correlations (Pearson for numeric, Cramér's V for categorical)
- Created engineered features

**Complaints Dataset:**
- Tokenized text into individual words
- Removed stopwords and applied stemming
- Joined with Bing sentiment lexicon
- Calculated sentiment scores per complaint
- Aggregated by customerID

### Feature Engineering

Created new variables:
- `phonelines`: 0 (no phone), 1 (single line), 2+ (multiple lines)
- `internet`: Yes/No (simplified from InternetService)
- `streaming`: 0, 1, or 2 services
- `score`: sum of sentiment scores
- `n_complaints`: complaint count per customer
- `weighted_score`: weighted sentiment metric

### Model Training

All models evaluated using 5-fold cross-validation:
- Training/validation split: 80/20 per fold
- Metric: misclassification error rate
- Hyperparameter tuning for GBM (trees, depth, shrinkage)

### Customer Segmentation

Two-step segmentation approach:
1. Granular clustering using K-Means on scaled features
2. PCA for dimensionality reduction
3. Coarse aggregation into business-relevant macro-segments
4. Business steering through dimension weighting

## Results

### Model Performance

| Model | Misclassification Error |
|-------|------------------------|
| Logistic Regression | 18.58% |
| **Lasso Regression** | **18.46%** |
| Gradient Boosting | 18.70% |
| Random Forest | 18.93% |

### Key Churn Drivers

**Significant Predictors:**
- Contract type (longer contracts reduce churn)
- Tenure (established customers less likely to churn)
- Internet service type (fiber optic increases churn)
- Payment method (electronic check increases churn)
- Sentiment scores (negative sentiment correlates with churn)
- Number of complaints (more complaints = higher churn risk)

### Sentiment Analysis Insights

- 54 topics explain 70% of complaint variability
- Strong correlation between negative sentiment and churn
- Main complaint themes: service quality, billing issues, technical problems

### Business Impact

The solution enables:
- Proactive identification of at-risk customers
- Targeted retention campaigns based on segmentation
- Root cause analysis through complaint text mining
- Prioritization of customer contacts based on conversion probability

## Project Structure

```
bcg-churn-prediction/
├── README.md
├── code/
│   ├── functions.R              # Custom R functions
│   └── final_report.Rmd         # Complete analysis
├── data/
│   ├── ABCTelco_clients.csv
│   ├── ABCTelco_complaints.csv
│   └── clientcomplain.csv       # Merged dataset
├── report/
│   └── final-report.pdf         # Final presentation
└── .gitignore
```

## Technologies Used

**R Packages:**
- Data manipulation: tidyverse, dplyr, tidyr
- Machine learning: caret, glmnet, randomForest, gbm, xgboost
- Text mining: tidytext, tm, lsa
- Clustering: cluster, e1071
- Visualization: ggplot2, RColorBrewer, wordcloud
- Statistical analysis: rstatix, DescTools

**Techniques:**
- Supervised learning (classification)
- Unsupervised learning (clustering, PCA)
- Natural Language Processing (sentiment analysis, topic modeling)
- Cross-validation and hyperparameter tuning

## Setup Instructions

### Prerequisites

R version 4.0 or higher

### Installation

Clone the repository:
```bash
git clone https://github.com/Egle22/bcg-churn-prediction.git
cd bcg-churn-prediction
```

Install required packages:
```r
install.packages(c(
  "tidyverse", "rstatix", "cluster", "gridExtra",
  "caret", "DescTools", "gbm", "glmnet", "randomForest",
  "tidytext", "tm", "RColorBrewer", "wordcloud",
  "xgboost", "e1071", "kableExtra", "lsa"
))
```

## Usage

### Running the Analysis

1. Open RStudio and set working directory:
```r
setwd("/path/to/bcg-churn-prediction")
```

2. Load custom functions:
```r
source("code/functions.R")
```

3. Run the complete analysis:
```r
rmarkdown::render("code/final_report.Rmd")
```

### Key Functions

The `functions.R` file contains reusable functions for:
- Data preprocessing and type conversion
- Association analysis (Cramér's V)
- Feature engineering
- Sentiment analysis pipeline
- Model evaluation utilities
- Topic modeling helpers

## Authors

**Group 9 - Six Sigma:**
- Egle Caudullo - [GitHub](https://github.com/Egle22) | [LinkedIn](https://www.linkedin.com/in/egle-caudullo-177993266)
- Andrea Bianco
- Federico Baio
- Alessio Pistone
- Alice Bianchi
- Festus Udealor

**Competition:** BCG X @ University of Bologna  
**Date:** April 2024

## Acknowledgments

- BCG X for organizing the competition and providing the challenge
- University of Bologna for hosting the event
- BCG X team: Giovanni Fasano, Matteo Zucca, Gabriele Corni, Giovanni Scabbia, Andrea Scarpelli

## References

1. Boston Consulting Group (2024). BCG X Presentation - Customer Segmentation & Prioritization for Personalization.

2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.

3. Silge, J., & Robinson, D. (2017). Text Mining with R: A Tidy Approach. O'Reilly Media.

## License

This project is licensed under the MIT License.

## Contact

For questions or collaborations:
- Email: egle.caudullo.22@gmail.com
- GitHub: @Egle22
- LinkedIn: Egle Caudullo
