import pandas as pd

# Load and analyze data
df = pd.read_csv('data.csv')
print(f'Data shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print('\nColumn info:')
for col in df.columns:
    print(f'{col}: {df[col].dtype}, {df[col].notna().sum()} non-null')