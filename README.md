# Credit Card Fraud Analysis
Includes mutliple feature selection algorithms & outlier detection methods, tried and tested, finally concluding on which method gave us the best results.

Dataset taken from `https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud`

## Introduction
This project aims to detect fraudulent credit card transactions using various machine learning techniques and algorithms. The dataset used in this project is the Credit Card Fraud Detection dataset, which contains transactions made by credit cards in September 2013 by European cardholders. The dataset is highly imbalanced, with only 0.172% of the transactions being fraudulent.

The dataset contains the following features:

- `Time`: Time elapsed between the first transaction in the dataset and the current transaction.
- `Amount`: Transaction amount.
- `Class`: Binary label indicating whether the transaction is fraudulent (1) or normal (0).
- `V1`, `V2`, ..., `V28`: Principal components obtained by applying PCA to the original features.

## Code Overview
The code is written in Python and uses various libraries such as pandas, numpy, scikit-learn, imbalanced-learn, and matplotlib. The main steps performed in the code are:

1. **Data Loading and Preprocessing**
   - Loading the dataset
   - Shuffling the data
   - Feature engineering (scaling the 'Amount' feature)
   - Handling missing values (if any)
   - Creating feature and target arrays

2. **Oversampling**
   - Applying the Synthetic Minority Over-sampling Technique (SMOTE) to balance the dataset.

3. **Fitness Function**
   - Defining a fitness function that evaluates the performance of a Random Forest Classifier on a given subset of features.

4. **Feature Selection Algorithms**
   - Implementing various feature selection algorithms to find the optimal subset of features:
     - Genetic Algorithm
     - Hill Climbing
     - Particle Swarm Optimization (PSO)
     - Simulated Annealing

5. **Outlier Detection**
   - Applying outlier detection algorithms to identify fraudulent transactions:
     - Isolation Forest
     - Local Outlier Factor
     - One-Class Support Vector Machine (SVM)
   - Evaluating the performance of these algorithms using accuracy, precision, recall, and F1-score.

6. **Data Visualization**
   - Plotting the distribution of normal and fraudulent transactions based on time and amount.
   - Visualizing the class distribution in the dataset.
   - Plotting the correlation matrix to identify highly correlated features.

## Usage
To run the code, follow these steps:

1. Clone the repository or download the code files.
2. Install the required libraries: pandas, numpy, scikit-learn, imbalanced-learn, and matplotlib.
3. Download the Credit Card Fraud Detection dataset from Kaggle and place it in the same directory as the code files.
4. Run the code file in your preferred Python environment.

Note: The code assumes that the dataset is named `creditcard.csv`. If your dataset has a different name, update the corresponding line in the code.

## Results
The code provides a detailed analysis of the performance of various outlier detection algorithms, including Isolation Forest, Local Outlier Factor, and One-Class SVM. The results indicate that the Isolation Forest algorithm performs better than the others in detecting fraudulent transactions, with an accuracy of 99.75% and a fraud detection rate of around 30%.

Additionally, the code explores different feature selection algorithms, such as Genetic Algorithm, Hill Climbing, Particle Swarm Optimization, and Simulated Annealing, to find the optimal subset of features for the Random Forest Classifier.
