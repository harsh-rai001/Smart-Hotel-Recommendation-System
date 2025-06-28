from pyspark.sql.functions import col
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def recommend_similar_hotels(spark_df, hotel_name, top_n=5):
    hotel_row = spark_df.filter(col("property_name") == hotel_name).collect()[0]
    cluster_id = hotel_row["cluster_id"]

    cluster_df = spark_df.filter(col("cluster_id") == cluster_id).select("property_name", "tfidf_features")
    pandas_df = cluster_df.toPandas()

    tfidf_matrix = np.vstack(pandas_df["tfidf_features"].values)
    similarities = cosine_similarity([hotel_row["tfidf_features"]], tfidf_matrix)[0]
    pandas_df["similarity"] = similarities

    results = pandas_df[pandas_df["property_name"] != hotel_name].sort_values("similarity", ascending=False).head(top_n)
    return results[["property_name", "similarity"]]


def recommend_filtered_hotels(spark_df, hotel_name, city=None, min_price=None, max_price=None, amenity_keywords=None, top_n=5):
    hotel_row = spark_df.filter(col("property_name") == hotel_name).collect()[0]
    cluster_id = hotel_row["cluster_id"]
    filtered_df = spark_df.filter(col("cluster_id") == cluster_id)

    if city:
        filtered_df = filtered_df.filter(col("city") == city)
    if min_price is not None:
        filtered_df = filtered_df.filter(col("price") >= min_price)
    if max_price is not None:
        filtered_df = filtered_df.filter(col("price") <= max_price)
    if amenity_keywords:
        for keyword in amenity_keywords:
            filtered_df = filtered_df.filter(col("in_your_room").contains(keyword))

    
    pandas_df = filtered_df.select("property_name", "tfidf_features", "price", "area").toPandas()
    if pandas_df.empty:
        return "No results match the filters."

    tfidf_matrix = np.vstack(pandas_df["tfidf_features"].values)
    similarities = cosine_similarity([hotel_row["tfidf_features"]], tfidf_matrix)[0]
    pandas_df["similarity"] = similarities
    pandas_df = pandas_df[pandas_df["property_name"] != hotel_name]

    # Return 
    return pandas_df.sort_values("similarity", ascending=False).head(top_n)[
        ["property_name", "area", "price", "similarity"]
    ]
