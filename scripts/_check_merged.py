"""Quick check of the rebuilt merged dataset."""
from src.data.data_loader import load_merged_dataset

df = load_merged_dataset()
print(f"Shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Tickers: {df['ticker'].nunique()}")
print(f"Trading days per ticker: ~{len(df) // df['ticker'].nunique()}")
print()
for col in ["return_direction", "return_5d", "return_20d"]:
    valid = df[col].notna().sum()
    print(f"  {col}: {valid} valid labels ({valid/len(df)*100:.1f}%)")
