import pandas as pd

df = pd.read_pickle("./data/data.pkl")

print(len(df))

# for col in df.columns:
#     print(f"Column: {col}")
#     print(df[col].dropna().head(200))
#     print("\n")