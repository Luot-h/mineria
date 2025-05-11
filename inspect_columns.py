import pandas as pd
import os

input_dir = "./raw-data"
first_file = next(f for f in os.listdir(input_dir) if f.endswith('.parquet'))
path = os.path.join(input_dir, first_file)
df = pd.read_parquet(path)
print(f"Columns in {first_file}:")
print(df.columns.tolist())

