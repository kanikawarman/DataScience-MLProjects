# Loan Default Prediction

## Project Overview
This project aims to predict loan defaults using machine learning models. The dataset contains various financial features related to borrowers, and the objective is to identify high-risk applicants to minimize loan defaults while maintaining fairness in approvals.

## Dataset
The dataset used for this analysis is sourced from [Kaggle](https://www.kaggle.com/) under the directory `/kaggle/input/loan-data`. If you want to reproduce the results, you may need to download the dataset manually from Kaggle.
This dataset includes numerical and categorical features representing borrower financial history, credit behavior, and other relevant factors. Key features used in modeling include:
- **Credit Score**
- **Annual Income**
- **Debt-to-Income Ratio**
- **Loan Amount**
- **Interest Rate**
- **Revolving Balance**

## Data Preprocessing
Before training the models, the data was cleaned and transformed:
- Handled missing values appropriately.
- Scaled numerical features using StandardScaler.
- Encoded categorical features using one-hot encoding.
- Applied log transformation to `revol.bal` for better distribution.
- Addressed class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

## Models Used
Three machine learning models were trained and evaluated:
1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Gradient Boosting Machine (GBM)**

## Model Evaluation
Each model was evaluated using a confusion matrix, precision, recall, and F1-score. The key findings are:

- **Random Forest performed the best overall:**
  - It had the lowest False Positives (303), reducing unnecessary loan rejections.
  - While its True Positives (4) were low, it achieved a better balance between precision and recall compared to the other models.
- **Logistic Regression and GBM struggled:**
  - They had high False Positives (1605 and 1596, respectively), leading to excessive loan rejections.
  - Their True Positives (7 and 9) were also low, making them less effective in identifying actual defaulters.

### Final Conclusion
- **Random Forest is the best model for this problem.**
- However, further tuning is needed to improve recall for defaults, including techniques like hyperparameter tuning and advanced class imbalance handling.

## How to Run the Notebook
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd loan-default-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open and execute `loan_default_prediction.ipynb`.

## Future Improvements
- Implement hyperparameter tuning for Random Forest.
- Experiment with different resampling techniques for class imbalance.
- Explore deep learning approaches like neural networks.
- Incorporate additional borrower features for improved model accuracy.

## Repository Structure
```
├── requirements.txt       # Python dependencies
├── loan_default_prediction.ipynb  # Main notebook
└── README.md              # Project documentation
```
