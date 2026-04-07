import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost
model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)

# SHAP explanation
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Plot and save
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png')
print("SHAP chart saved as shap_summary.png")
