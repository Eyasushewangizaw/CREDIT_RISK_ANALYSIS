import pandas as pd
import matplotlib.pyplot as plt

columns = [
    'checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_account', 'employment', 'installment_rate', 'personal_status',
    'other_debtors', 'residence_since', 'property', 'age', 'other_installments',
    'housing', 'existing_credits', 'job', 'dependents', 'telephone', 'foreign_worker',
    'target'
]

df = pd.read_csv('german.data', sep=' ', names=columns)

# How many defaulted vs not
print("Default counts (1=good, 2=bad):")
print(df['target'].value_counts())

# Any missing values?
print("\nMissing values:")
print(df.isnull().sum())

# Basic stats on numeric columns
print("\nBasic statistics:")
print(df[['age', 'credit_amount', 'duration']].describe())

# Average loan amount by default status
print("\nAverage credit amount by default status:")
print(df.groupby('target')['credit_amount'].mean())

# Average age by default status
print("\nAverage age by default status:")
print(df.groupby('target')['age'].mean())

# Average duration by default status
print("\nAverage loan duration by default status:")
print(df.groupby('target')['duration'].mean())

# Plot default rate
df['target'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Good vs Bad Credit')
plt.xlabel('1 = Good, 2 = Bad')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('default_rate.png')
print("\nChart saved as default_rate.png")