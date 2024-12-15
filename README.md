# Loan-Approval-Prediction--Classification

## Objective
The objective of this project is to develop a predictive model for loan approvals based on applicant data such as income, credit history, and loan amount. This model aims to assist financial institutions in making data-driven decisions and streamlining the loan approval process.

## Methodology

### Data Exploration and Preprocessing
- **Data Overview**: The dataset includes features like applicant income, loan amount, loan term, credit history, marital status, and gender.
- **Data Cleaning**: Addressed missing values, encoded categorical variables, and standardized numerical features to ensure consistency.
- **Feature Engineering**: Created new features to capture interactions between applicant characteristics and loan conditions.

### Model Development
- **Baseline Model**: Initiated with a logistic regression model to establish a basic benchmark for classification.
- **Advanced Models**: Implemented and fine-tuned machine learning algorithms such as:
  - Decision Trees
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - Gradient Boosting Classifier (e.g., XGBoost, LightGBM)

### Evaluation Metrics
- Evaluated models using key classification metrics:
  - **Accuracy**: To measure overall prediction performance.
  - **Precision, Recall, and F1-Score**: To evaluate the balance between false positives and false negatives.
  - **ROC-AUC**: To analyze the model's capability to distinguish between classes.

## Results
- The final model achieved a substantial improvement over the baseline, providing high accuracy and balanced precision-recall performance.
- Insights:
  - Credit history and income level were the most significant predictors for loan approval.
  - Feature importance analysis revealed trends in applicant demographics influencing loan decisions.

## Conclusion
The project successfully built an efficient and interpretable loan approval prediction model. It highlights the importance of data preprocessing and advanced classification techniques for financial decision-making. Future improvements may include:
- Expanding the dataset to include macroeconomic factors.
- Exploring ensemble techniques for enhanced robustness.
- Deploying the model as an API for integration with banking systems.

---

## How to Run
1. Clone the repository: `git clone https://github.com/yourusername/loan-approval-prediction.git`
2. Open the Jupyter Notebook: `jupyter notebook Loan_Approval_Prediction.ipynb` or open the notebook in google drive
3. Run the notebook cells sequentially to reproduce results.

## Files in the Repository
- **Loan_Approval_Prediction.ipynb**: Contains the code and analysis for the project.
- **train.csv**: Input training data
- **test.csv**: Input training data
- **prediction_results.csv**: Output prediction results


## References
- Loan prediction datasets from Kaggle and other sources
- Documentation for scikit-learn, XGBoost, and other libraries used in the project.

Feel free to explore the repository and provide feedback or contributions!

