This report presents the analysis conducted for the “Practical Machine Learning” course project. The primary objective of the project is to predict how participants perform a weightlifting exercise (represented by the "classe" variable) using data collected from accelerometers placed on the belt, forearm, upper arm, and dumbbell. The dataset used is publicly available from the Human Activity Recognition Using Smartphones Data Set and was generously provided by researchers at the University of California, Irvine. We gratefully acknowledge their contribution in sharing this valuable resource.
The project involves several key steps, including data exploration, preprocessing, feature selection, dimensionality reduction, model selection, training, evaluation, and making predictions on the test dataset. All analyses and visualizations are performed using the R programming language. 



![Screenshot 2025-06-03 051819](https://github.com/user-attachments/assets/95cd9250-cf3f-46a1-9573-32096760c6c9)


The original training dataset consisted of 160 variables. An initial review identified several issues, including irrelevant features, a high percentage of missing values in some columns, and variables with near-zero variance.
To prepare the data for modeling, the following preprocessing steps were applied:
Removal of Irrelevant Features: Seven columns that were clearly unrelated to the prediction objective were eliminated (r irrelevant_cols).
Handling of Missing Data: Any columns with more than 95% missing values were excluded from the dataset.
Elimination of Near-Zero Variance Features: Variables exhibiting minimal variance, which contribute little to predictive modeling, were removed using the nearZeroVar function from the caret package.
As a result of these preprocessing steps, the training dataset was reduced to 53 variables.

**Feature Selection and Dimensionality Reduction**

To enhance the dataset and potentially boost model performance, two techniques were applied: ANOVA for selecting relevant features and Principal Component Analysis (PCA) for reducing dimensionality.
ANOVA for Feature Selection: ANOVA was utilized to determine which variables exhibited statistically significant differences in mean values across the five exercise categories (A, B, C, D, and E). This approach aids in identifying features that are likely to be strong predictors of the target variable, *classe*.

![Screenshot 2025-06-03 052124](https://github.com/user-attachments/assets/8f9f270e-5371-4263-b4a1-c189f774cc4a)

**PCA for Dimensionality Reduction**
PCA was performed to reduce the dimensionality of the dataset while retaining most of the variance. The scree plot was used to determine the number of principal components to retain.
Based on the scree plot and the cumulative variance explained (not shown in the plot), the first 20 principal components were selected for further analysis.

![Screenshot 2025-06-03 052142](https://github.com/user-attachments/assets/8d85f258-9e62-45e2-b458-8518eab2628d)


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

![Screenshot 2025-06-03 052248](https://github.com/user-attachments/assets/2bd85ad1-bf6b-4cc8-9960-dc2c4e0af9d9)

![Screenshot 2025-06-03 052309](https://github.com/user-attachments/assets/59cd46ea-db83-41a7-a637-97a2e455f317)

![Screenshot 2025-06-03 052331](https://github.com/user-attachments/assets/d07256b7-63db-4683-8782-be82f169a94a)

![Screenshot 2025-06-03 052405](https://github.com/user-attachments/assets/78893adb-e7b3-42f2-a1c7-c74d4e82f59a)


**Final Model and Justification**

Based on cross-validation results, the Random Forest model trained on the full dataset was chosen as the final model.
Rationale:

It achieved exceptionally high accuracy (0.993) and Kappa (0.992) during cross-validation.
It was more computationally efficient than the GBM model.
The Random Forest model on the full dataset offered a simpler solution compared to using GBM with ANOVA-selected features.
The performance difference between Random Forest (full dataset) and GBM (ANOVA dataset) was negligible.
Estimated Out-of-Sample Error:
Using cross-validation as a basis, the final model is expected to reach around 0.993 in accuracy and approximately 0.992 in Kappa when applied to new, unseen data.

![Screenshot 2025-06-03 052420](https://github.com/user-attachments/assets/9d95c29c-d1c7-46bb-9a4e-3a5885717df7)


##Test Set Prediction The final Random Forest model was trained on the entire training dataset using the optimal hyperparameter (mtry) found during cross-validation.
The test set (pml-testing.csv) was preprocessed using the exact same steps as the training data.

![Screenshot 2025-06-03 052438](https://github.com/user-attachments/assets/d970eab8-b60d-4e7b-a4eb-2848534a1ddb)



**Conclusion**

This project illustrated the end-to-end process of developing and evaluating machine learning models to classify exercise types using accelerometer data. Both Random Forest and GBM models were trained and compared, with the Random Forest model using the full dataset ultimately selected as the final model due to its strong accuracy, efficiency, and straightforward implementation. Techniques like ANOVA and PCA were employed to gain deeper insights into the data and to explore methods for feature selection and dimensionality reduction. The final model demonstrated excellent cross-validation performance, suggesting strong potential for accurately predicting new, unseen cases.







