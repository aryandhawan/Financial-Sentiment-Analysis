## Financial Sentiment Analysis: An Experiment in Predictive Modelling
This project is a rigorous, end-to-end machine learning experiment designed to test a fundamental hypothesis: Can the sentiment of a public figure's social media posts be a reliable predictor of short-term stock price movements?

Instead of just building a model, the goal was to construct a professional, scalable MLOps pipeline to investigate the limits of what classical machine learning algorithms can achieve on such a noisy and complex problem. The findings from this experiment serve as a data-driven justification for the necessity of more advanced Deep Learning techniques for nuanced language understanding.

#  Project Workflow & System Design
The entire system was designed as a modular, configuration-driven "model factory," following modern MLOps best practices.

Data Ingestion: A robust data pipeline was engineered to create a novel dataset by pulling from two separate, real-time sources:

Text Data: Historical tweets were sourced from the X (Twitter) API (via Tweepy) and historical Kaggle archives.

Financial Data: Corresponding historical stock prices were fetched from the Yahoo! Finance API (via yfinance).

Feature Engineering: A critical data processing phase focused on creating a feature-rich dataset. The key step was a time-series merge, aligning each tweet with the last known stock price at the moment of posting. New features were engineered to provide context, including:

A sentiment_score for each tweet using the VADER sentiment analyser.

An influence_score combining sentiment with engagement metrics like likeCount.

Time-based features like hour_of_day to capture market timing.

Model Benchmarking: To find the best possible model, a systematic benchmark was conducted. The pipeline was designed to train and evaluate four different classifiers:

Logistic Regression

Decision Tree

Random Forest

XGBoost

Handling Class Imbalance: Initial results revealed a severe class imbalance. The pipeline was upgraded to include the SMOTE (Synthetic Minority Over-sampling Technique) on the training data, forcing the models to learn the patterns of the minority "Up" and "Down" classes.

# 🛠 Tech Stack
Data Manipulation & Analysis: Pandas, NumPy

Machine Learning: Scikit-learn, XGBoost, Imbalanced-learn

Data Acquisition: Tweepy, yfinance

NLP: VADER Sentiment

Development: Python, Jupyter Notebook, Git

#  Results & Conclusion: Finding the Limit
After extensive feature engineering and handling class imbalance, the experiment yielded a clear and insightful conclusion.

While the models' ability to predict the "Up" and "Down" classes improved significantly after using SMOTE, the overall performance of even the best models (Random Forest and XGBoost) remained low.

This is not a failure of the models, but a successful finding of the experiment.

The results demonstrate that a classical machine learning approach, which relies on a "bag-of-words" representation of text and engineered features, is insufficient to capture the complex, nuanced, and often sarcastic nature of human language required to reliably predict stock market movements. The signal in this data is too weak and noisy for these methods to overcome.

#  Future Work: The Path to Deep Learning
This project successfully establishes the boundary of traditional ML for this problem and provides a powerful justification for the next logical step. Future work will involve replacing the classical models with Deep Learning architectures designed for sequence and context:

Recurrent Neural Networks (RNNs/LSTMs): To understand the order and context of words in a tweet.

Transformer Models (like BERT): To achieve a much deeper, contextual understanding of the language, potentially unlocking the subtle signals that classical models could not find.
