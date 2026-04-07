import pandas as pd

columns = [
    'checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_account', 'employment', 'installment_rate', 'personal_status',
    'other_debtors', 'residence_since', 'property', 'age', 'other_installments',
    'housing', 'existing_credits', 'job', 'dependents', 'telephone', 'foreign_worker',
    'target'
]

df = pd.read_csv('german.data', sep=' ', names=columns)

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())