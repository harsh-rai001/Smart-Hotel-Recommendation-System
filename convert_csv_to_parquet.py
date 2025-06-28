import pandas as pd
import ast

df = pd.read_csv("Hotel_data_with_clusters.csv")


df["tfidf_features"] = df["tfidf_features"].apply(ast.literal_eval)

df["price"] = df["price"].astype("float")
df["cluster_id"] = df["cluster_id"].astype("int")

df.to_parquet("hotels.parquet", engine="pyarrow", index=False)

print("Successfully converted to hotels.parquet")
