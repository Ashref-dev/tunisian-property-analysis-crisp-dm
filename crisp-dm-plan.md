# CRISP-DM Plan: Tunisian Property Price Prediction

This document outlines the plan for developing a property price prediction model using the Tunisian property dataset, following the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology.

## 1. Business Understanding [✓ Completed]

*   **Business Objectives:**
    *   Primary: Develop a model to accurately predict property prices in Tunisia based on features available in the dataset (category, room count, bathroom count, size, type, city, region).
    *   Secondary: Understand the key drivers influencing property prices in Tunisia. Identify which features are most predictive.
*   **Assess Situation:**
    *   Resources:
        *   Data: `Property Prices in Tunisia.csv` dataset scraped from Tayara.tn (available in the notebook environment).
        *   Personnel: User (domain expertise, guidance) and AI assistant (data analysis, modeling).
        *   Software: Python environment with libraries used in the notebook (pandas, numpy, matplotlib, seaborn) and potentially machine learning libraries (scikit-learn, xgboost, lightgbm).
    *   Requirements/Constraints: The model should provide reasonably accurate price predictions. The process needs to be documented. Data privacy is not a major concern as the data seems publicly scraped. Time constraints depend on the project scope.
    *   Risks: Data quality issues (missing values imputed as -1 might affect prediction), potential biases in scraped data, model interpretability vs. accuracy trade-offs.
    *   Terminology: Define terms like 'category', 'type', 'region' as used in the dataset. Define evaluation metrics (RMSE, MAE, R²).
*   **Data Mining Goals:**
    *   Predict the `price` (or `log_price`) for property listings.
    *   Achieve a specific level of predictive accuracy (e.g., minimize Root Mean Squared Error (RMSE) or Mean Absolute Error (MAE) on a held-out test set).
    *   Develop a profile of features strongly correlated with high/low prices.
*   **Project Plan:**
    *   Follow the six phases of CRISP-DM.
    *   Iterate between Data Preparation, Modeling, and Evaluation phases.
    *   Document findings and model performance at each stage.
    *   Use standard train/test splits and potentially cross-validation for evaluation.

## 2. Data Understanding [✓ Completed]

*   **Initial Data Collection:** The data is already collected and loaded in the `Tunisian Property Market.ipynb` notebook from `/kaggle/input/property-prices-in-tunisia/Property Prices in Tunisia.csv`.
*   **Data Description:**
    *   Source: Scraped from Tayara.tn.
    *   Size: Approx. 12,748 listings, 9 columns.
    *   Columns: `category`, `room_count`, `bathroom_count`, `size`, `type`, `price`, `city`, `region`, `log_price`.
    *   Data Types: Mix of categorical (category, type, city, region) and numerical (room_count, bathroom_count, size, price, log_price).
*   **Data Exploration:**
    *   The notebook contains initial exploration: value counts for categories, distribution plots for cities.
    *   Further exploration needed: Distribution of numerical features (histograms), relationship between features and price (scatter plots, box plots), correlation matrix analysis (partially done in the notebook).
*   **Data Quality:**
    *   Missing Values: Identified in `room_count`, `bathroom_count`, `size`. Currently imputed with -1 in the source CSV. This needs careful handling for predictive modeling.
    *   Outliers: Need to check for outliers in `price` and `size`, potentially visualized in the notebook or requiring further checks. The log transformation of price (`log_price`) likely helps mitigate price outliers.
    *   Consistency: Check for inconsistencies in categorical data (e.g., variations in spelling for cities/regions).

## 3. Data Preparation [✓ Completed]

*   **Data Selection:** Use all relevant columns for prediction initially (`category`, `room_count`, `bathroom_count`, `size`, `type`, `city`, `region`). The target variable will be `price` (or potentially `log_price`, requiring transformation back after prediction).
*   **Data Cleaning:**
    *   Handle Missing Values: Revisit the -1 imputation. Strategies to consider:
        *   Mean/Median/Mode imputation based on feature distribution.
        *   Model-based imputation (e.g., KNNImputer).
        *   Dropping rows/columns (if missingness is extensive and imputation is unreliable, but likely undesirable given dataset size).
    *   Outlier Treatment: Decide on handling outliers (e.g., capping, transformation, removal) based on exploration findings.
*   **Data Construction (Feature Engineering):**
    *   Create new features if hypotheses suggest they might be useful (e.g., `price_per_sq_meter` if `size` is available, interactions between features).
    *   Encode Categorical Features: Use techniques like One-Hot Encoding or Target Encoding for `category`, `type`, `city`, `region`.
*   **Data Integration:** Not applicable as data comes from a single source.
*   **Data Formatting:**
    *   Feature Scaling: Scale numerical features (e.g., using StandardScaler or MinMaxScaler) if required by the chosen modeling algorithm (e.g., linear models, SVM).
    *   Train/Test Split: Split the data into training and testing sets (e.g., 80%/20% split) to evaluate model performance on unseen data. Ensure shuffling if data has any inherent order.

## 4. Modeling [✓ Completed]

*   **Select Modeling Techniques:**
    *   Start with baseline models (e.g., Linear Regression, Ridge, Lasso).
    *   Explore tree-based models (Decision Tree, Random Forest).
    *   Consider gradient boosting models (XGBoost, LightGBM) for potentially higher accuracy.
    *   Assumptions: Linear models assume linearity and independence. Tree models are generally non-parametric. Note any specific data requirements for each model.
*   **Generate Test Design:**
    *   Use the held-out test set for final evaluation.
    *   Employ k-fold cross-validation on the training set for hyperparameter tuning and robust model assessment.
    *   Define evaluation metrics: RMSE, MAE, R-squared (R²).
*   **Build Model:**
    *   Train selected models on the prepared training data.
    *   Tune hyperparameters using techniques like GridSearchCV or RandomizedSearchCV with cross-validation.
*   **Assess Model:**
    *   Evaluate models based on cross-validation performance using chosen metrics.
    *   Analyze feature importance from models (especially tree-based ones).
    *   Compare models based on technical criteria and select the best performing one(s).
    *   Check for overfitting by comparing training set performance to validation/test set performance.

    In the modeling phase (step 4 of CRISP-DM), you typically explore and apply various data mining techniques based on the problem at hand. use these techniques.

Classification: Methods such as decision trees, support vector machines, logistic regression, and neural networks that help predict categorical outcomes.

Regression: Techniques like linear or polynomial regression to forecast continuous variables.

Clustering: Algorithms such as k-means or hierarchical clustering that group similar data points together.

Association Rule Mining: Approaches like the Apriori algorithm to uncover relationships between variables.

## 5. Evaluation [✓ Completed]

*   **Evaluate Results:**
    *   Assess the best model(s) performance on the final test set.
    *   Interpret results in the context of the business objective: Is the prediction accuracy sufficient? (e.g., is the average error acceptable?).
    *   Analyze prediction errors: Are there specific types of properties the model struggles with?
    *   Compare findings against business success criteria defined earlier.
*   **Review Process:**
    *   Review the entire project: Were all steps followed? Any steps missed? Data quality issues overlooked?
    *   Ensure the model uses only features available at prediction time.
    *   Document findings thoroughly.
*   **Determine Next Steps:**
    *   Decision: Proceed to deployment if results are satisfactory, or iterate back to previous phases (e.g., more feature engineering, try different models, collect more data if possible) if improvement is needed.
    *   List possible actions: Refine features, try ensemble methods, gather more granular location data if feasible.

## 6. Deployment [✓ Completed]

*   **Plan Deployment:**
    *   Determine how the model will be used (e.g., integrated into an application, used for batch predictions, simple prediction API).
    *   Document the steps needed to deploy the chosen model (saving the model object, preprocessing pipeline, prediction script).
*   **Plan Monitoring and Maintenance:**
    *   Strategy for monitoring model performance over time (concept drift).
    *   Plan for retraining the model periodically with new data.
    *   Define processes for updating the model if performance degrades.
*   **Produce Final Report:**
    *   Create a comprehensive report summarizing the project phases, methodology, findings, model evaluation, and deployment plan.
*   **Review Project:**
    *   Final project review: Lessons learned, successes, challenges.
    *   Document experiences for future projects.

---
*Reference: [CRISP-DM Methodology Guide](https://www.sv-europe.com/crisp-dm-methodology/)* 