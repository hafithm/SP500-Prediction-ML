# SP500-Prediction-ML

A Machine Learning-based approach to forecasting S&P 500 stock trends using financial indicators, historical data, and feature engineering.

## Project Overview
Stock market prediction has always been a significant challenge due to the complex and volatile nature of financial markets. This project leverages Machine Learning (ML) models to predict whether an S&P 500 stock will outperform or underperform based on historical price movements, technical indicators, and macroeconomic factors.

🔹 Goal: Predict the daily movement (Outperform vs. Underperform) of S&P 500 stocks
🔹 Dataset: Historical stock price data & fundamental metrics from Yahoo Finance API

🔹 Machine Learning Models Used:

Support Vector Machine (SVM)
Random Forest Classifier (alternative) 

🔹 Key Techniques: Data preprocessing, feature engineering, and model evaluation
🔹 Technologies Used: Python, Scikit-Learn, Pandas, Matplotlib, Yahoo Finance API

##  Dataset & Feature Engineering
The dataset consists of S&P 500 stock prices and financial indicators collected from Yahoo Finance API. Key features include:

- Stock Metrics: Opening price, closing price, high, low, adjusted close, volume
- Technical Indicators: Price change percentage, volatility, momentum
- Fundamental Analysis: P/E ratio, revenue growth, market cap, enterprise value
- Market Sentiment: Performance of S&P 500 index (SP500 Change)

Feature engineering plays a crucial role in improving model performance. The following transformations were applied:

🔹 Handling missing values using mean imputation
🔹 Scaling features with StandardScaler() for normalization
🔹 Encoding categorical variables (e.g., converting "Outperform" → 1, "Underperform" → 0)
🔹 Merging stock-specific and market-wide features into a single dataset

## Machine Learning Models
1️⃣ Support Vector Machine (SVM)
SVM is used as the primary model due to its ability to handle high-dimensional data and prevent overfitting.

Implementation Steps:

Preprocessed dataset with StandardScaler()
Trained SVM classifier on stock performance labels (Performance Label)
Evaluated accuracy using Stratified K-Fold Cross-Validation

''' 
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Initialize cross-validation
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
accuracies = []

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = LinearSVC(C=1.0, max_iter=5000, random_state=42)
    clf.fit(X_train, y_train)

    fold_accuracy = clf.score(X_test, y_test) * 100
    accuracies.append(fold_accuracy)

print(f"Cross-Validation Accuracy: {np.mean(accuracies):.2f}%")

''' 
✅ Final Accuracy: ~63.10%
✅ Key Findings: The model captures some market patterns but requires feature optimization for improvement.



## Challenges & Improvements

Stock market data is highly volatile and difficult to model
Some financial indicators had missing values, requiring imputation
SVM struggled with complex non-linear relationships

Future Improvements:

Hyperparameter tuning for SVM to optimize decision boundaries
Try deep learning models (LSTM, Transformer) for time-series forecasting
Expand feature set to include social media sentiment analysis
Automate data pipeline to fetch and update stock data in real-time



