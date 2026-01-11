# BCG X Customer Churn Prediction - Complete Analysis
# Authors: Group 9 - Six Sigma
# Egle Caudullo, Andrea Bianco, Federico Baio, Alessio Pistone, Alice Bianchi, Festus Udealor
# Competition: BCG X @ University of Bologna
# Date: April 2024

# 1. Libraries
library(tidyverse)
library(rstatix)
library(cluster)
library(gridExtra)
library(grid)
library(caret)
library(knitr)
library(DescTools)
library(rmarkdown)
library(scales)
library(gbm)
library(glmnet)
library(randomForest)
library(tree)
library(tidytext)
library(tm)
library(RColorBrewer)
library(wordcloud)
library(xgboost)
library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(e1071)
library(kableExtra)
library(colorspace)
library(lsa)

# 1. DATA LOADING AND PREPROCESSING

# Load clients dataset
clients <- read.csv("ABCTelco_clients.csv")

# Convert character variables to factor
convert_chr_to_factor <- function(data) {
  data %>%
    mutate(across(where(is.character), as.factor))
}

clients <- convert_chr_to_factor(clients)
clients$SeniorCitizen <- as.factor(clients$SeniorCitizen)
clients$Churn <- ifelse(clients$Churn == "Yes", 1, 0)

# Remove NA values
clients <- clients %>% filter(!is.na(TotalCharges))

# 2. CORRELATION ANALYSIS

# Numeric variables correlation
num_var <- clients[, sapply(clients, is.numeric)]
cor_matrix <- round(cor(num_var, use = "complete.obs", method = "pearson"), 3)

# Cramér's V for categorical variables
VCramers <- function(data) {
  cat_data <- data[, sapply(data, function(x) is.factor(x))]
  n <- ncol(cat_data)
  mat <- matrix(NA, nrow = n, ncol = n)
  colnames(mat) <- rownames(mat) <- colnames(cat_data)
  
  for (i in 1:n) {
    for (j in 1:n) {
      if (i == j) {
        mat[i, j] <- 1
      } else {
        table <- table(cat_data[[i]], cat_data[[j]])
        mat[i, j] <- CramerV(table)
      }
    }
  }
  return(as.data.frame(mat))
}

mat_Cramer <- VCramers(clients)
mat_Cramer <- round(mat_Cramer, 3)

# 3. Feature Engineering

# Create phonelines variable
clients$phonelines <- factor(
  ifelse(clients$MultipleLines == "No phone service", 0,
         ifelse(clients$MultipleLines == "No", 1, 2))
)

# Create internet variable
clients$internet <- factor(
  ifelse(clients$InternetService == "No", "No", "Yes")
)

# Create streaming variable
clients$streaming <- ifelse(
  clients$StreamingTV == "No" & clients$StreamingMovies == "No", 0,
  ifelse(clients$StreamingTV == "Yes" & clients$StreamingMovies == "Yes", 2,
         ifelse(clients$StreamingTV == "No" & clients$StreamingMovies == "Yes", 1,
                ifelse(clients$StreamingTV == "Yes" & clients$StreamingMovies == "No", 1, 0)))
)

# Verify TotalCharges = MonthlyCharges * tenure
clients$new_TotalCharges <- round(clients$MonthlyCharges * clients$tenure, 2)

# 2. COMPLAINTS DATASET PROCESSING

complaints <- read.csv("ABCTelco_complaints.csv", stringsAsFactors = FALSE)

# Prepare text data
texts <- complaints$complaint
texts <- texts[!is.na(texts) & texts != ""]
complaints_df <- tibble(line = 1:length(texts), text = texts)

# Add customerID
comp_cust <- complaints_df %>%
  mutate(customerID = complaints$customerID)

# Tokenize text
complaints_words <- comp_cust %>%
  unnest_tokens(word, text)

# Remove stopwords
data("stop_words")
complaints_words_clean <- complaints_words %>%
  anti_join(stop_words, by = "word")

# Add sentiment
bing_sentiments <- get_sentiments("bing")
complaints_sentiment <- complaints_words_clean %>%
  inner_join(bing_sentiments, by = "word")

# Calculate sentiment scores
complaint_scores <- complaints_sentiment %>%
  count(line, sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>%
  mutate(sentiment_score = positive - negative)

# Merge with original complaints
final_sentiment <- complaints_df %>%
  left_join(complaint_scores, by = "line") %>%
  mutate(sentiment_score = ifelse(is.na(sentiment_score), 0, sentiment_score))

# 4. Merge Datasets
complaints1 <- complaints %>%
  rename(text = complaint) %>%
  select(-complaint_number)

sentiment_id <- final_sentiment %>%
  left_join(complaints1, by = "text")

sentiment_id1 <- sentiment_id %>%
  select(c(customerID, sentiment_score))

sentiment_group <- sentiment_id1 %>%
  group_by(customerID) %>%
  summarise(
    score = sum(sentiment_score),
    n_complaints = n()
  ) %>%
  mutate(weighted_score = (score * n_complaints) / sum(n_complaints) * 100)

# Merge with clients data
df <- clients %>%
  full_join(sentiment_group, by = "customerID") %>%
  mutate(
    score = replace_na(score, 0),
    n_complaints = replace_na(n_complaints, 0),
    weighted_score = replace_na(weighted_score, 0)
  )

# 5. Topic Modeling
texts <- complaints$complaint
corpus <- Corpus(VectorSource(texts))
corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, function(x) removeWords(x, stopwords("english")))
corpus <- tm_map(corpus, stemDocument, language = "english")

td.mat <- as.matrix(TermDocumentMatrix(corpus))
td.mat <- t(td.mat)
out <- svd(td.mat)

# Explained variance analysis
plot(out$d^2 / sum(out$d^2), type = "b",
     main = "Explained Variance by Topic",
     ylab = "Variance", xlab = "Number of Topics")
grid()

# 70% variance threshold
var_of_compon <- out$d^2 / sum(out$d^2)
cum_var <- cumsum(var_of_compon)
n_components <- which(cum_var >= 0.7)[1]
cat("Number of components for 70% variance:", n_components, "\n")

# 6. Model Training with 5-Fold Cross-Validation

set.seed(123)
k <- 5
n <- nrow(df)
folds <- sample(1:k, n, replace = TRUE)

# Initialize error vectors
err.cv.logistic <- NULL
err.cv.lasso <- NULL
err.cv.boosting <- NULL
err.cv.rf <- NULL

# Logistic Regression
for (i in 1:k) {
  x.train <- df[folds != i, ]
  x.test <- df[folds == i, ]
  
  out.log <- glm(Churn ~ ., data = x.train, family = "binomial")
  phat.log <- predict(out.log, newdata = x.test, type = "response")
  pred.log <- ifelse(phat.log > 0.5, 1, 0)
  
  err.cv.logistic[i] <- mean(pred.log != x.test$Churn)
}

cat("LOGISTIC CV Error:", mean(err.cv.logistic), "\n")

# Lasso Regression
for (i in 1:k) {
  x.train <- df[folds != i, ]
  x.test <- df[folds == i, ]
  
  x.train.mat <- model.matrix(Churn ~ ., data = x.train)[, -1]
  y.train <- x.train$Churn
  
  cv.lasso <- cv.glmnet(x.train.mat, y.train, family = "binomial", alpha = 1)
  
  x.test.mat <- model.matrix(Churn ~ ., data = x.test)[, -1]
  phat.lasso <- predict(cv.lasso, newx = x.test.mat, s = "lambda.min", type = "response")
  pred.lasso <- ifelse(phat.lasso > 0.5, 1, 0)
  
  err.cv.lasso[i] <- mean(pred.lasso != x.test$Churn)
}

cat("LASSO CV Error:", mean(err.cv.lasso), "\n")

# Gradient Boosting with Tuning
numb <- c(300, 400, 500, 600, 700)
depths <- c(1, 2, 3)
shrinkages <- c(0.01, 0.05, 0.1)
grid <- expand.grid(n.trees = numb, depth = depths, shrink = shrinkages)

err.grid.selection <- matrix(NA, nrow = k, ncol = nrow(grid))
df$Churn <- as.numeric(df$Churn) - 1

for (i in 1:k) {
  x.train <- df[folds != i, ]
  x.test <- df[folds == i, ]
  
  folds2 <- sample(1:k, nrow(x.train), replace = TRUE)
  
  for (s in 1:k) {
    cv.train <- x.train[folds2 != s, ]
    cv.test <- x.train[folds2 == s, ]
    
    for (j in 1:nrow(grid)) {
      boost.out <- gbm(Churn ~ ., data = cv.train, 
                      distribution = "bernoulli",
                      n.trees = grid$n.trees[j],
                      interaction.depth = grid$depth[j],
                      bag.fraction = 1, 
                      shrinkage = grid$shrink[j])
      
      phat.grid <- predict(boost.out, newdata = cv.test, 
                          n.tree = grid$n.trees[j], type = "response")
      pred.grid <- ifelse(phat.grid > 0.5, 1, 0)
      
      err.grid.selection[s, j] <- mean(pred.grid != cv.test$Churn)
    }
  }
  
  best.idx <- which.min(colMeans(err.grid.selection))
  best.params <- grid[best.idx, ]
  
  boost.out <- gbm(Churn ~ ., data = x.train, 
                  distribution = "bernoulli",
                  n.trees = best.params$n.trees,
                  interaction.depth = best.params$depth, 
                  bag.fraction = 1,
                  shrinkage = best.params$shrink)
  
  phat.boost <- predict(boost.out, newdata = x.test, 
                       n.tree = best.params$n.trees, type = "response")
  pred.boosting <- ifelse(phat.boost > 0.5, 1, 0)
  
  err.cv.boosting[i] <- mean(pred.boosting != x.test$Churn)
}

cat("BOOSTING CV Error:", mean(err.cv.boosting), "\n")

# Random Forest
df$Churn <- as.factor(df$Churn)

for (i in 1:k) {
  x.train <- df[folds != i, ]
  x.test <- df[folds == i, ]
  
  y.train <- df[folds != i, ]$Churn
  y.test <- df[folds == i, ]$Churn
  
  x.train$Churn <- as.factor(x.train$Churn)
  
  rf.clas <- randomForest(Churn ~ ., data = x.train, 
                         importance = TRUE, ntree = 300)
  pred.rf <- predict(rf.clas, newdata = x.test, type = "response")
  
  err.cv.rf[i] <- mean(pred.rf != y.test)
}

cat("RANDOM FOREST CV Error:", mean(err.cv.rf), "\n")

# Final Results Summary
results_summary <- data.frame(
  Model = c("Logistic", "Lasso", "Boosting", "Random Forest"),
  Misclassification_Error = c(
    mean(err.cv.logistic),
    mean(err.cv.lasso),
    mean(err.cv.boosting),
    mean(err.cv.rf)
  )
)

print(results_summary)
