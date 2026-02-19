"""Quick script to rebuild merged dataset and show stats."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.data_loader import load_merged_dataset

df = load_merged_dataset(save_processed=True)
print(f"\nFinal shape: {df.shape}")
tickers = sorted(df["ticker"].unique())
print(f"Tickers: {tickers}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

if "article_count" in df.columns:
    counts = df.groupby("ticker")["article_count"].sum()
    print("\nArticle counts per ticker:")
    for t, c in counts.items():
        print(f"  {t}: {int(c)}")
    print(f"\n  TOTAL: {int(counts.sum())}")
