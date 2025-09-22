import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder  # --- FIX: Import LabelEncoder ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# --- 1. Load Data ---
df = pd.read_csv("./data/final_training_dataset.csv")

# --- 2. Define Features and Target ---
X = df.drop('Trend', axis=1)
y = df['Trend']


# This step is required because XGBoost cannot handle string labels.
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data using the NEWLY encoded target variable
x_train, x_test, y_train, y_test = train_test_split(
    X, y_encoded,  # Use y_encoded for the split
    test_size=0.2,
    random_state=42,
    stratify=y_encoded # Also stratify on the encoded labels
)

# --- Define Preprocessing Pipelines ---
numerical_cols = ['sentiment_score', 'likeCount', 'retweetCount']
text_col = 'fullText'

numerical_pipeline = Pipeline([('scaler', StandardScaler())])
text_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english', max_features=1000))])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('text', text_pipeline, text_col)
])

# Your model dictionary is perfect.
models = {
    'Logistic Regression': ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote',SMOTE(random_state=42)),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ]),
    'Decision Tree': ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote',SMOTE(random_state=42)),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ]),
    'Random Forest': ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote',SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),
    'XGBoost': ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote',SMOTE(random_state=42)),
        # You can add the use_label_encoder=False and eval_metric='mlogloss' for best practice
        ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'))
    ])
}

# --- 5. Train and Evaluate Each Model ---
for model_name, pipeline in models.items():
    print(f"--- Training {model_name} ---")
    pipeline.fit(x_train, y_train)
    print(f"✓ Training for {model_name} has been completed")

    print(f"--- Evaluating {model_name} ---")
    # The pipeline is evaluated on the original, imbalanced test set for an honest result.
    y_pred = pipeline.predict(x_test)

    target_names = le.classes_

    print(f'Classification Report for {model_name}:')
    print(classification_report(y_test, y_pred, target_names=target_names))
    print('--' * 30 + "\n")

print('Done')
