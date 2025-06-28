def get_unique_values(spark_df, column):
    """
    Returns a list of unique values from a Spark DataFrame column.
    """
    return [row[column] for row in spark_df.select(column).distinct().collect()]
