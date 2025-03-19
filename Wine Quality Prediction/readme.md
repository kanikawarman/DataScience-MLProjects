# Wine Quality Prediction

## Project Overview
Wine is an alcoholic beverage made from fermented grapes, with its quality influenced by various physicochemical properties. This project aims to analyze these attributes and develop machine learning models to:  

1. **Classify wines as red or white** based on their chemical properties.  
2. **Predict wine quality** (low, medium, or high) using classification models.  

We follow a structured Machine Learning workflow inspired by **CRISP-DM** to preprocess data, perform exploratory analysis, build predictive models, and evaluate their effectiveness.

## Dataset Information
- **Source**: [Kaggle - Wine Quality Dataset](/kaggle/input/wine-quality/winequalityN.csv)  
- **Description**: The dataset consists of red and white **Vinho Verde** wines from Portugal and includes physicochemical measurements such as acidity, sugar levels, and alcohol content.  
- **Features**:
  - **Inputs (Physicochemical Properties)**: Fixed Acidity, Volatile Acidity, Citric Acid, Residual Sugar, Chlorides, Free Sulfur Dioxide, Total Sulfur Dioxide, Density, pH, Sulphates, Alcohol.  
  - **Outputs (Target Variables)**:
    - **Wine Type**: Red or White  
    - **Wine Quality**: Categorized as Low, Medium, or High  

## Methodology
### **1. Data Preprocessing**
- Handled missing values.
- Normalized and transformed skewed features.
- Addressed collinearity and multicollinearity.
- Applied feature selection and dimensionality reduction.

### **2. Exploratory Data Analysis (EDA)**
- Visualized distributions of features.
- Conducted hypothesis testing and validation.
- Used correlation heatmaps to identify key relationships.

### **3. Model Training & Evaluation**
- Implemented multiple classification models:
  - **Wine Type Prediction**: Achieved high accuracy using simple models.
  - **Wine Quality Prediction**: Used ensemble models due to class imbalances.
- **Models Evaluated**:
  - Random Forest (RF)
  - Gradient Boosting Classifier (GBC)
  - XGBoost (XGBC)
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Decision Trees (with and without PCA)
  - AdaBoost
  - Stacking Classifier (RF + GBC + XGBC)

### **4. Model Performance**
- **Best Model for Wine Type Prediction**: Achieved **81.98% accuracy** with Random Forest.
- **Best Model for Wine Quality Classification**: Ensembles performed best, particularly:
  - **Random Forest**: **82.13% precision**, **0.908 ROC AUC score**.
  - **Gradient Boosting Classifier**: **81.98% accuracy**.
  - **XGBoost**: Balanced performance with strong generalization.

#### **Confusion Matrix & Key Metrics**
| Model                     | CV Accuracy | Accuracy | ROC AUC Score |
|---------------------------|------------|----------|--------------|
| Random Forest             | 80.56%     | 82.13%   | 90.80%       |
| Gradient Boosting         | 78.93%     | 81.98%   | 87.47%       |
| XGBoost                   | 78.26%     | 81.90%   | 89.20%       |
| K-Nearest Neighbors       | 78.54%     | 80.90%   | 88.74%       |
| Logistic Regression       | 71.29%     | 72.70%   | 80.95%       |
| Decision Tree (PCA)       | 71.28%     | 71.00%   | 76.03%       |

## Key Insights
- **Simple models performed well** for wine type classification, achieving high accuracy at low computational cost.
- **Predicting wine quality was more complex**, requiring ensemble methods to capture nuances, especially for **high-quality wines**, which had fewer samples.
- **Feature importance analysis** revealed that **alcohol, volatile acidity, and sulphates** were key indicators of wine quality.
- The model is effective but has a **13% accuracy gap**, partly due to missing sensory and external factors like **grape variety, tannin levels, and harvest details**.

## Practical Applications
- **For Wine Distributors**: Predict quality of new wines before purchasing.
- **For Winemakers**: Gain insights into the physicochemical factors that influence wine quality.
- **For Consumers**: Assist in selecting wines based on quality ratings.

## Installation & Usage
To run the project locally:

### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/wine-quality-prediction.git
cd wine-quality-prediction
