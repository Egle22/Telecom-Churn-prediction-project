# Custom Functions for Telecom Churn Prediction Analysis
# Authors: Group 9 - Six Sigma
# Date: 2025

# 1. DATA PREPROCESSING FUNCTIONS

# Convert character variables to factors
# Input: dataframe with character variables
# Output: dataframe with character variables converted to factors
convert_chr_to_factor <- function(data) {
  data %>%
    mutate(across(where(is.character), as.factor))
}

# 2. ASSOCIATION ANALYSIS FUNCTIONS

# Compute Cramér's V association matrix for categorical variables
# Input: dataframe containing factor variables
# Output: matrix of Cramér's V values (0-1) measuring association strength
# Note: 0 = no association, 1 = perfect association
VCramers <- function(data) {
  # Extract only categorical variables
  cat_data <- data[, sapply(data, function(x) is.factor(x))]
  n <- ncol(cat_data)
  
  # Initialize empty matrix
  mat <- matrix(NA, nrow = n, ncol = n)
  colnames(mat) <- rownames(mat) <- colnames(cat_data)
  
  # Compute Cramér's V for each pair
  for (i in 1:n) {
    for (j in 1:n) {
      if (i == j) {
        mat[i, j] <- 1  # Perfect association with itself
      } else {
        table <- table(cat_data[[i]], cat_data[[j]])
        mat[i, j] <- CramerV(table)
      }
    }
  }
  
  return(as.data.frame(mat))
}

# Filter high association pairs from Cramér's V matrix
# Input: Cramér's V matrix, threshold value (default 0.7)
# Output: dataframe of variable pairs with association > threshold
filter_high_associations <- function(cramer_matrix, threshold = 0.7) {
  # Set lower triangle and diagonal to NA
  cramer_matrix[lower.tri(cramer_matrix, diag = TRUE)] <- NA
  
  # Find pairs above threshold
  corr_var <- which(abs(cramer_matrix) > threshold, arr.ind = TRUE)
  
  # Create dataframe of results
  corr_pairs <- data.frame(
    var1 = rownames(cramer_matrix)[corr_var[, 1]],
    var2 = colnames(cramer_matrix)[corr_var[, 2]],
    association = cramer_matrix[corr_var]
  )
  
  # Sort by association strength
  corr_pairs <- corr_pairs[order(-corr_pairs$association), ]
  
  return(corr_pairs)
}

# 3. FEATURE ENGINEERING FUNCTIONS

# Create phonelines variable from MultipleLines
# Input: dataframe with MultipleLines variable
# Output: dataframe with new phonelines factor variable (0, 1, 2)
create_phonelines <- function(data) {
  data$phonelines <- factor(
    ifelse(data$MultipleLines == "No phone service", 0,
           ifelse(data$MultipleLines == "No", 1, 2))
  )
  return(data)
}

# Create internet binary variable from InternetService
# Input: dataframe with InternetService variable
# Output: dataframe with new internet factor variable (Yes/No)
create_internet <- function(data) {
  data$internet <- factor(
    ifelse(data$InternetService == "No", "No", "Yes")
  )
  return(data)
}

# Create streaming variable from StreamingTV and StreamingMovies
# Input: dataframe with StreamingTV and StreamingMovies variables
# Output: dataframe with new streaming variable (0, 1, 2)
create_streaming <- function(data) {
  data$streaming <- ifelse(
    data$StreamingTV == "No" & data$StreamingMovies == "No", 0,
    ifelse(data$StreamingTV == "Yes" & data$StreamingMovies == "Yes", 2,
           ifelse(data$StreamingTV == "No" & data$StreamingMovies == "Yes", 1,
                  ifelse(data$StreamingTV == "Yes" & data$StreamingMovies == "No", 1, 0)))
  )
  return(data)
}

# 4. SENTIMENT ANALYSIS FUNCTIONS

# Prepare text corpus for sentiment analysis
# Input: vector of text documents
# Output: cleaned corpus (lowercase, no punctuation, no stopwords, stemmed)
prepare_corpus <- function(texts) {
  corpus <- Corpus(VectorSource(texts))
  corpus <- tm_map(corpus, tolower)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, function(x) removeWords(x, stopwords("english")))
  corpus <- tm_map(corpus, stemDocument, language = "english")
  return(corpus)
}

# Tokenize text into words
# Input: dataframe with text column
# Output: dataframe with one word per row
tokenize_text <- function(data, text_col = "text") {
  data %>%
    unnest_tokens(word, !!sym(text_col))
}

# Remove stopwords from tokenized text
# Input: dataframe with word column
# Output: dataframe without stopwords
remove_stopwords <- function(data) {
  data("stop_words")
  data %>%
    anti_join(stop_words, by = "word")
}

# Add sentiment labels to words
# Input: dataframe with word column
# Output: dataframe with sentiment column (positive/negative)
add_sentiment <- function(data) {
  bing_sentiments <- get_sentiments("bing")
  data %>%
    inner_join(bing_sentiments, by = "word")
}

# Calculate sentiment score for each document
# Input: dataframe with line, sentiment columns
# Output: dataframe with sentiment_score per line
calculate_sentiment_score <- function(data) {
  data %>%
    count(line, sentiment) %>%
    pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>%
    mutate(sentiment_score = positive - negative)
}

# Aggregate sentiment by customer
# Input: dataframe with customerID and sentiment_score
# Output: dataframe with aggregated scores per customer
aggregate_customer_sentiment <- function(data) {
  data %>%
    group_by(customerID) %>%
    summarise(
      score = sum(sentiment_score),
      n_complaints = n()
    ) %>%
    mutate(weighted_score = (score * n_complaints) / sum(n_complaints) * 100)
}

# 5. MODEL EVALUATION FUNCTIONS

# Compute misclassification error
# Input: predicted values, true values
# Output: misclassification error rate
misclassification_error <- function(predictions, actual) {
  mean(predictions != actual)
}

# Create folds for k-fold cross-validation
# Input: number of observations, number of folds
# Output: vector of fold assignments
create_folds <- function(n, k) {
  sample(1:k, n, replace = TRUE)
}

# 6. TOPIC MODELING FUNCTIONS

# Perform SVD for topic modeling
# Input: term-document matrix
# Output: SVD decomposition
perform_svd <- function(td_matrix) {
  svd(td_matrix)
}

# Calculate explained variance by number of topics
# Input: SVD output
# Output: vector of explained variance proportions
explained_variance <- function(svd_output) {
  svd_output$d^2 / sum(svd_output$d^2)
}

# Find number of components for desired variance threshold
# Input: SVD output, variance threshold (default 0.7)
# Output: number of components needed
components_for_variance <- function(svd_output, threshold = 0.7) {
  var_of_compon <- explained_variance(svd_output)
  cum_var <- cumsum(var_of_compon)
  which(cum_var >= threshold)[1]
}

# 7. UTILITY FUNCTIONS

# Safe merge with NA replacement
# Input: two dataframes, key column, value to replace NA
# Output: merged dataframe with NA replaced
safe_merge <- function(df1, df2, by, na_value = 0) {
  merged <- df1 %>%
    full_join(df2, by = by) %>%
    mutate(across(where(is.numeric), ~replace_na(., na_value)))
  return(merged)
}

# Print model results summary
# Input: vector of cross-validation errors, model name
# Output: printed summary statistics
print_cv_results <- function(cv_errors, model_name) {
  cat("\n", model_name, "Results:\n")
  cat("Mean CV Error:", round(mean(cv_errors), 4), "\n")
  cat("SD CV Error:", round(sd(cv_errors), 4), "\n")
  cat("Min CV Error:", round(min(cv_errors), 4), "\n")
  cat("Max CV Error:", round(max(cv_errors), 4), "\n")
}

# 8. DATA VALIDATION FUNCTIONS

# Check for missing values
# Input: dataframe
# Output: dataframe of variables with NA counts
check_missing <- function(data) {
  na_counts <- colSums(is.na(data))
  na_df <- data.frame(
    variable = names(na_counts),
    na_count = na_counts,
    na_percentage = round(na_counts / nrow(data) * 100, 2)
  )
  na_df <- na_df[na_df$na_count > 0, ]
  return(na_df)
}

# Verify data types
# Input: dataframe
# Output: dataframe of variable names and types
check_data_types <- function(data) {
  type_df <- data.frame(
    variable = names(data),
    type = sapply(data, class)
  )
  return(type_df)
}
