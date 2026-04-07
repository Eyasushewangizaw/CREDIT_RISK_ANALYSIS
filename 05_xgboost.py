import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

columns = [
    'checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_account', 'employment', 'installment_rate', 'personal_status',
    'other_debtors', 'residence_since', 'property', 'age', 'other_installments',
    'housing', 'existing_credits', 'job', 'dependents', 'telephone', 'foreign_worker',
    'target'
]

df = pd.read_csv('german.data', sep=' ', names=columns)

# Prepare data
df['target'] = df['target'].map({1: 0, 2: 1})
text_columns = df.select_dtypes(include='object').columns
le = LabelEncoder()
for col in text_columns:
    df[col] = le.fit_transform(df[col])

X = df.drop('target', axis=1)
y = df['target']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost
model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("XGBoost Results:")
print(classification_report(y_test, y_pred))
print("AUC Score:", round(roc_auc_score(y_test, y_prob), 3))

# Show top 5 most important features
print("\nTop 5 most important features:")
importance = pd.Series(model.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False).head(5))