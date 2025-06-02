This report presents the analysis conducted for the “Practical Machine Learning” course project. The primary objective of the project is to predict how participants perform a weightlifting exercise (represented by the "classe" variable) using data collected from accelerometers placed on the belt, forearm, upper arm, and dumbbell. The dataset used is publicly available from the Human Activity Recognition Using Smartphones Data Set and was generously provided by researchers at the University of California, Irvine. We gratefully acknowledge their contribution in sharing this valuable resource.
The project involves several key steps, including data exploration, preprocessing, feature selection, dimensionality reduction, model selection, training, evaluation, and making predictions on the test dataset. All analyses and visualizations are performed using the R programming language. 

# Load libraries
library(caret)
library(ggplot2)
library(corrplot)
library(randomForest)

# Load data
trainData <- read.csv("pml-training.csv")
testData <- read.csv("pml-testing.csv")
summary(trainData)

The original training dataset consisted of 160 variables. An initial review identified several issues, including irrelevant features, a high percentage of missing values in some columns, and variables with near-zero variance.
To prepare the data for modeling, the following preprocessing steps were applied:
Removal of Irrelevant Features: Seven columns that were clearly unrelated to the prediction objective were eliminated (r irrelevant_cols).
Handling of Missing Data: Any columns with more than 95% missing values were excluded from the dataset.
Elimination of Near-Zero Variance Features: Variables exhibiting minimal variance, which contribute little to predictive modeling, were removed using the nearZeroVar function from the caret package.
As a result of these preprocessing steps, the training dataset was reduced to 53 variables.

**Feature Selection and Dimensionality Reduction**

To enhance the dataset and potentially boost model performance, two techniques were applied: ANOVA for selecting relevant features and Principal Component Analysis (PCA) for reducing dimensionality.
ANOVA for Feature Selection: ANOVA was utilized to determine which variables exhibited statistically significant differences in mean values across the five exercise categories (A, B, C, D, and E). This approach aids in identifying features that are likely to be strong predictors of the target variable, *classe*.

# Univariate Feature Selection using ANOVA

# Store the p-values
p_values <- numeric(ncol(numeric_data))
names(p_values) <- names(numeric_data)

# Perform ANOVA for each numeric variable
for (i in 1:ncol(numeric_data)) {
  # Create a linear model
  model <- lm(numeric_data[, i] ~ trainData$classe)
  # Perform ANOVA
  anova_result <- anova(model)
  # Store the p-value
  p_values[i] <- anova_result$"Pr(>F)"[1]
}

# Sort variables by p-value (ascending order)
sorted_p_values <- sort(p_values)

# Select top N variables based on p-values (e.g., top 20)
N <- 20
selected_variables_anova <- names(sorted_p_values)[1:N]

# Print selected variables
print(selected_variables_anova)

The top 20 variables with the lowest p-values were selected for further analysis.

# Perform PCA
pca_result <- prcomp(numeric_data, scale. = TRUE)

# Scree plot
plot(pca_result, type = "l", main = "Scree Plot")

# Get the principal component scores
pca_scores <- as.data.frame(pca_result$x)

# Add the 'classe' variable to the PCA scores for visualization
pca_scores$classe <- trainData$classe
# Scatter plot of the first two principal components, colored by classe
ggplot(pca_scores, aes(x = PC1, y = PC2, color = classe)) +
    geom_point() +
    ggtitle("PCA: First Two Principal Components") +
    theme_minimal()

Based on the scree plot and the cumulative variance explained (not shown in the plot), the first 20 principal components were selected for further analysis.

**Model Selection**

Two machine learning models were chosen for this classification task: Random Forest (RF) and Gradient Boosting Machine (GBM).
Rationale for Model Choices:
Random Forest: Robust, handles non-linearity well, provides variable importance measures, and generally performs well on a variety of datasets. GBM: Often achieves very high accuracy but can be more prone to overfitting and requires careful tuning. Cross-Validation:
10-fold cross-validation was used to evaluate the models and tune their hyperparameters.

**Model Training and Evaluation**
Both the Random Forest and Gradient Boosting Machine (GBM) models were trained and assessed using three different versions of the dataset:

* **Full Dataset:** Included all 53 variables retained after the preprocessing stage.
* **ANOVA Dataset:** Comprised the top 20 features identified through ANOVA-based feature selection.
* **PCA Dataset:** Contained the top 20 principal components derived from Principal Component Analysis.

# 1. Prepare Datasets

# Full Dataset (already have trainData)

# ANOVA Dataset
trainData_anova <- trainData[, c(selected_variables_anova, "classe")]

# PCA Dataset (using first 20 PCs)
trainData_pca <- pca_scores[, c(paste0("PC", 1:20), "classe")]

# 2. Set up trainControl

# Define cross-validation method (10-fold CV)
train_control <- trainControl(method = "cv", number = 10)

# 3. Train and Tune Models

# --- Random Forest (rf) ---

# Define tuning grid for mtry
rf_tuneGrid <- expand.grid(mtry = c(2, 3, 5, 7, 9))

# Train RF on Full Dataset
set.seed(123)
rf_full <- train(classe ~ ., data = trainData, method = "rf", trControl = train_control, tuneGrid = rf_tuneGrid)

# Train RF on ANOVA Dataset
set.seed(123)
rf_anova <- train(classe ~ ., data = trainData_anova, method = "rf", trControl = train_control, tuneGrid = rf_tuneGrid)

# Train RF on PCA Dataset
set.seed(123)
rf_pca <- train(classe ~ ., data = trainData_pca, method = "rf", trControl = train_control, tuneGrid = rf_tuneGrid)

# --- Gradient Boosting Machine (gbm) ---

# Define tuning grid for GBM parameters
gbm_tuneGrid <- expand.grid(
  n.trees = c(100, 200, 300),           # Number of trees
  interaction.depth = c(1, 2, 3),     # Tree depth
  shrinkage = c(0.01, 0.1),             # Learning rate
  n.minobsinnode = c(5, 10)           # Minimum observations in terminal nodes
)

# Train GBM on Full Dataset
set.seed(123)
gbm_full <- train(classe ~ ., data = trainData, method = "gbm", trControl = train_control, tuneGrid = gbm_tuneGrid, verbose = FALSE)

# Train GBM on ANOVA Dataset
set.seed(123)
gbm_anova <- train(classe ~ ., data = trainData_anova, method = "gbm", trControl = train_control, tuneGrid = gbm_tuneGrid, verbose = FALSE)

# Train GBM on PCA Dataset
set.seed(123)
gbm_pca <- train(classe ~ ., data = trainData_pca, method = "gbm", trControl = train_control, tuneGrid = gbm_tuneGrid, verbose = FALSE)

# 4. Evaluate Performance

# Create a function to extract performance metrics
get_performance <- function(model) {
  best_tune <- model$bestTune
  performance <- model$results[which.max(model$results$Accuracy), ]
  return(performance)
}

# Get performance for each model and dataset
rf_full_performance <- get_performance(rf_full)
rf_anova_performance <- get_performance(rf_anova)
rf_pca_performance <- get_performance(rf_pca)

gbm_full_performance <- get_performance(gbm_full)
gbm_anova_performance <- get_performance(gbm_anova)
gbm_pca_performance <- get_performance(gbm_pca)

# 5. Compare Performance

# Create a data frame to store the results
performance_summary <- data.frame(
  Model = c("Random Forest", "Random Forest", "Random Forest", "GBM", "GBM", "GBM"),
  Dataset = c("Full", "ANOVA", "PCA", "Full", "ANOVA", "PCA"),
  Accuracy = c(rf_full_performance$Accuracy, rf_anova_performance$Accuracy, rf_pca_performance$Accuracy,
               gbm_full_performance$Accuracy, gbm_anova_performance$Accuracy, gbm_pca_performance$Accuracy),
  Kappa = c(rf_full_performance$Kappa, rf_anova_performance$Kappa, rf_pca_performance$Kappa,
            gbm_full_performance$Kappa, gbm_anova_performance$Kappa, gbm_pca_performance$Kappa)
)
print(performance_summary)
# Visualize performance comparison
comparison_plot <- ggplot(performance_summary, aes(x = Dataset, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  ggtitle("Model Performance Comparison") +
  theme_minimal()

# Display the plot using print()
print(comparison_plot)

**Final Model and Justification**

Based on cross-validation results, the Random Forest model trained on the full dataset was chosen as the final model.
Rationale:

It achieved exceptionally high accuracy (0.993) and Kappa (0.992) during cross-validation.
It was more computationally efficient than the GBM model.
The Random Forest model on the full dataset offered a simpler solution compared to using GBM with ANOVA-selected features.
The performance difference between Random Forest (full dataset) and GBM (ANOVA dataset) was negligible.
Estimated Out-of-Sample Error:
Using cross-validation as a basis, the final model is expected to reach around 0.993 in accuracy and approximately 0.992 in Kappa when applied to new, unseen data.

# --- Final Model Training and Test Set Prediction ---

# 1. Train Final Random Forest Model on the Entire Training Dataset

# Get the best mtry value from the cross-validation results of rf_full
best_mtry <- rf_full$bestTune$mtry

# Train the final Random Forest model on the entire training set
set.seed(123)  # For reproducibility
final_rf_model <- randomForest(classe ~ ., data = trainData, mtry = best_mtry)

##Test Set Prediction The final Random Forest model was trained on the entire training dataset using the optimal hyperparameter (mtry) found during cross-validation.
The test set (pml-testing.csv) was preprocessed using the exact same steps as the training data.

# 3. Make Predictions on the Test Set

# Make predictions using the final Random Forest model
final_predictions <- predict(final_rf_model, newdata = testData)

# 4. Format Predictions

# Print the predictions
print(final_predictions)


**Conclusion**

This project illustrated the end-to-end process of developing and evaluating machine learning models to classify exercise types using accelerometer data. Both Random Forest and GBM models were trained and compared, with the Random Forest model using the full dataset ultimately selected as the final model due to its strong accuracy, efficiency, and straightforward implementation. Techniques like ANOVA and PCA were employed to gain deeper insights into the data and to explore methods for feature selection and dimensionality reduction. The final model demonstrated excellent cross-validation performance, suggesting strong potential for accurately predicting new, unseen cases.







