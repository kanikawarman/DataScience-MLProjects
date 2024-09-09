# DataScience-MLProjects
This repository contains various case studies on various topics which use python, various python libraries for data wrangling, visualization, Ml models like KNN, Decision tree, Random Forest and more

Project1 : Deciphering KNN
KNN folder contains:
1) info file --> information about the content of the KNN folder.
2) Introduction to KNN --> This document explains the concepts of K-nn
3) Knn_diabetes.ipnyb --> This jupyter notebook contains a simple implementation of K-NN model. In the first part, the prediction is done without the use of any k-nn libraries. In the second part, it uses sckit knn-library to make predictions. Then the comparison is made on both the implementation.

Project2 : Deciphering Decision Tree
Decision Tree: Trained and Predicted the model on Iris dataset (continuous values), using the scikit libraries and making the model from scratch. Comparing the results for both

Project3 : ML Project _ Improved Heart Diseased Prediction
The folder contains:
1) Project presentation - .ppt file
2) Project Report
3) project implemented notebook 
The focus of this project is to develop prediction algorithm using machine learning that could help in predicting heart diseases and other heart related ailments in humans. The machine learning technique used in the paper focus on ‘Stacked Ensemble’ Technique. 
This machine learning project focuses on implementing an already published paper available at: https://arxiv.org/pdf/2304.06015.pdf
To further improve the algorithm, following methods have been implemented:
  1. Use feature selection to focus on attributes that contribute more to the target variable decision.
  2. Evaluate all the base models and use only the ones that have better result for the target variable compared to others.
  3. Use 2-level stacking
     
Project4 : A/B Testing on AdSmart dataset
Executed A/B testing on the AdSmart dataset employing statistical tools such as the z-test and proportion_confint. The results contributed to informed decision-making, optimizing advertising performance based on statistically significant findings.

Project5 : Bank Churn Binary Classification
Explore the Binary Classification with Bank Churn notebook! Predict customer churn, treating certain numerical features as categorical for improved accuracy. The project involves pipelines, data preprocessing, engineered features, and an ensemble learning approach using LGBM and XGBoost. A weighted voting classifier combines models for the final prediction on customer exit or retention.

Project 6: Implementing Neural Network without libraries / from scratch
  mnist_nn_from_scratch --> This notebook implements a neural network from scratch without using external libraries on the MNIST dataset. The process involves       loading and preprocessing the dataset, creating the neural network architecture, training the model, and evaluating its performance using various metrics,         including accuracy and a detailed analysis with a confusion matrix and classification report.

Project 7: Implementing Neural Network with libraries
  mnist-using-lib.ipynb --> This notebook implements a neural network using Tensorflow and Kera libraries on the MNIST dataset. The process involves loading and preprocessing the dataset, training the model, and evaluating its performance using various metrics, including accuracy and a detailed analysis with a confusion matrix and classification report.

Project 8: Movie Recommendation system
  Movie_Recommendation_system.ipynb --> This notebook implements a KNN based content recommendation system for the customers similar to previously high-rated items by them.

Project 9: OCR_handwritten digits using OpenCV
  OCR_Handwritten_openCV.ipynb --> Developed an OCR system for recognizing handwritten digits using OpenCV and k-Nearest Neighbors (kNN), achieving an initial accuracy of 9.43% on a dataset of 2500 samples. Implemented image preprocessing, block segmentation, and model training to classify digits from a custom dataset, demonstrating fundamental skills in computer vision and machine learning.

Project 10: Stroke Prediction Using Classification Models and SMOTE Sampling
 Stroke Prediction Using Classification Models and SMOTE Sampling.ipynb --> Utilized diverse classification models including Logistic Regression, K-Nearest Neighbors, Decision Trees, Random Forest, Support Vector Machine, Gradient Boosting, XGBoost, and LightGBM to predict stroke occurrence based on demographic and health parameters. Implemented SMOTE (Synthetic Minority Oversampling Technique) to address class imbalance, achieving up to 92.78% accuracy with XGBoost, 91.79% with Random Forest, and significant improvements in recall and precision metrics for each model.

Project 11: Credit Risk Modeling - Loan Payment Prediction model
 credit-risk-modeling.ipynb --> Developed a credit risk prediction model using a VotingClassifier with XGBoost, Logistic Regression, and Random Forest, achieving 97.5% accuracy and a mean squared error of 0.025.

 Project 12: Sarcasm Detection in News Headlines Using Logistic Regression
 - Notebook --> Sarcasm Detection in News Headlines.ipynb
 - Objective --> This project implements a Logistic Regression model to detect sarcasm in news headlines. Utilizing linguistic features and sentiment analysis, the model is trained on pre-processed text data to classify headlines as sarcastic or non-sarcastic. Key techniques include feature extraction with TextBlob for sentiment analysis, text pre-processing using NLTK for tokenization and stemming, and the application of Logistic Regression with scikit-learn. The workflow encompasses data collection, feature engineering, text cleaning, model training, and evaluation.
