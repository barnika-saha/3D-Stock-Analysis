import pandas as pd

df = pd.read_csv(r"D:\AI_ML_SE\ML\Sentiment\data\news_clean_rss.csv")
print(df.columns.tolist())
print("Number of features:", len(df.columns))