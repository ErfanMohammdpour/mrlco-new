import csv

# Load and analyze data
with open('data.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    data = list(reader)

print(f'Data shape: ({len(data)}, {len(headers)})')
print(f'Columns: {headers}')
print('\nColumn info:')

# Analyze each column
for i, col in enumerate(headers):
    col_data = [row[i] for row in data if row[i]]
    # Determine data type
    try:
        # Try to convert to float
        float_vals = [float(val) for val in col_data[:10]]
        dtype = 'float64'
    except:
        dtype = 'string'
    
    print(f'{col}: {dtype}, {len(col_data)} non-null')