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
