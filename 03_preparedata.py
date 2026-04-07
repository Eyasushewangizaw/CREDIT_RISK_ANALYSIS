import pandas as pd
from sklearn.preprocessing import LabelEncoder

columns = [
    'checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_account', 'employment', 'installment_rate', 'personal_status',
    'other_debtors', 'residence_since', 'property', 'age', 'other_installments',
    'housing', 'existing_credits', 'job', 'dependents', 'telephone', 'foreign_worker',
    'target'
]

df = pd.read_csv('german.data', sep=' ', names=columns)

# Convert target: 1=good becomes 0, 2=bad becomes 1
df['target'] = df['target'].map({1: 0, 2: 1})

# Separate text columns from number columns
text_columns = df.select_dtypes(include='object').columns

# Convert text columns to numbers
le = LabelEncoder()
for col in text_columns:
    df[col] = le.fit_transform(df[col])

# Split into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("\nDefault rate:", round(y.mean() * 100, 1), "%")
print("\nFirst 5 rows of prepared data:")
print(X.head())
