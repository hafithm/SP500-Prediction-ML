# SP500-Prediction-ML

![image alt](https://github.com/hafithm/SP500-Prediction-ML/blob/0a5d4dc828236b7f1fd91ca7220d359dfa96abe0/SP500.webp)



## Project Overview

This project aims to predict S&P 500 stock performance using machine learning models based on historical stock prices, fundamental indicators, and technical indicators. 
The objective is to determine whether a stock outperforms or underperforms the broader market, using Support Vector Machines (SVM) for classification.

Project Goals
 Predict whether a stock will outperform or underperform the S&P 500 based on financial indicators.
 Utilize fundamental features, such as P/E ratios, revenue growth, and debt levels, to determine stock performance.
 Incorporate technical indicators and price movements into the analysis.
 Evaluate model performance and visualize key feature importances.
 
## Methodology
Data Preprocessing:
-    Stock Price Data via Yahoo Finance API (yfinance):
         Retrieved historical stock prices (Adj Close, High, Low, Volume, Open).
         Automated daily data updates using Python scripts.
-    Cleaned raw financial data, handling missing values using mean imputation.
-    Standardized numerical features for better SVM performance.
-    Merged price movements, fundamental metrics, and technical indicators into a final dataset.

Feature Selection:
-    Selected key fundamental & technical indicators that historically impact stock performance.
-    Feature importance analysis helped in understanding which metrics influence predictions.
-    Machine Learning Model:

Implemented Support Vector Machines (SVM) to classify stocks.
-    Used Stratified K-Fold Cross-Validation (4-fold) to ensure robustness.
-    Evaluated accuracy, precision, and recall to measure model effectiveness.
-    Visualization & Insights:

Plotted feature importance rankings to understand the driving factors behind stock performance.
Analyzed historical price movements & volatility trends to validate model predictions.

## Key Findings
Stock Fundamentals Matter:

-    Features like Debt-to-Equity Ratio, Revenue Growth, and Price-to-Book Ratio were highly influential.
-    Companies with strong financials tend to outperform the market in the long run.
-    Machine Learning Performance:

  The SVM model achieved ~63% accuracy in predicting whether a stock outperforms or underperforms.
  Despite market volatility, the model identified key factors that differentiate winning stocks.

  The Random Forest model achieved 67% accuracy, while SVM scored 63%.
  Tree-based methods captured complex non-linear relationships better than SVM.

Technical Indicators Play a Role:

Stock price momentum, market beta, and volume changes helped in short-term predictions.
Combining fundamental and technical analysis improved predictive power.
Volatility & Market Trends Impact Predictions:

Stocks with higher volatility were harder to classify due to rapid fluctuations.
The broader market movement (S&P 500 changes) significantly influenced stock performance.


##  Dataset & Feature Engineering
The dataset consists of S&P 500 stock prices and financial indicators collected from Yahoo Finance API. Key features include:

- Stock Metrics: Opening price, closing price, high, low, adjusted close, volume
- Technical Indicators: Price change percentage, volatility, momentum
- Fundamental Analysis: P/E ratio, revenue growth, market cap, enterprise value
- Market Sentiment: Performance of S&P 500 index (SP500 Change)

Feature engineering plays a crucial role in improving model performance. The following transformations were applied:

- Handling missing values using mean imputation
- Scaling features with StandardScaler() for normalization
- Encoding categorical variables (e.g., converting "Outperform" → 1, "Underperform" → 0)
- Merging stock-specific and market-wide features into a single dataset



## Challenges & Future Improvements
1. Stock Market Volatility & Modeling Complexity
Financial markets are highly unpredictable, with frequent fluctuations due to macroeconomic conditions, earnings reports, and geopolitical events.
Traditional ML models like SVM struggled with non-linear relationships, while Random Forest handled it better but still had room for improvement.
2. Missing Data in Financial Indicators
Some key financial ratios (Debt-to-Equity, Revenue Growth) had missing values for certain stocks.
Imputation methods (mean replacement) were used, but this could be improved with alternative data sources or predictive imputation models.
3. Future Model Enhancements
Random Forest provided better accuracy, but still lacked interpretability.
Future improvements could involve ensemble models (Gradient Boosting, XGBoost) or Deep Learning (LSTMs) to capture complex stock market relationships.



Despite these challenges, the SVM model was able to achieve a reasonable 63.1% accuracy, showing that financial indicators do play a role in stock performance. 
The next steps would involve incorporating alternative datasets (news sentiment, earnings surprises) and testing non-linear models to improve prediction performance. 



