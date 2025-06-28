import streamlit as st
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np

from recommender import recommend_similar_hotels, recommend_filtered_hotels

#  Spark
spark = SparkSession.builder \
    .appName("Hotel Recommendation") \
    .getOrCreate()


df = spark.read.parquet("hotels.parquet") ## Loading file

# Extract unique hotel names and cities
hotel_names = [row["property_name"] for row in df.select("property_name").distinct().collect()]
city_names = [row["city"] for row in df.select("city").distinct().collect()]

### Streamlit UI setup  ###
st.set_page_config(page_title="Smart Hotel Recommendation", layout="wide")

st.markdown("<h1 style='text-align: center; color: #ff6347;'>Smart Hotel Recommendation</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: grey;'>Every need has a Recommendation</h4>", unsafe_allow_html=True)
st.markdown("---")

# Similar Hotels
st.subheader(" Get Similar Hotel Recommendations")
selected_hotel = st.selectbox("Select a Hotel", sorted(hotel_names) if hotel_names else ["No hotels found"])

if st.button("Search Similar Hotels"):
    with st.spinner("Finding similar hotels..."):
        try:
            results = recommend_similar_hotels(df, selected_hotel)
            st.success("Here are the similar hotels:")
            st.table(results)
        except Exception as e:
            st.error(f"Error: {e}")


# Filtering 
st.markdown("---")
st.subheader(" Find Hotels")

col1, col2, col3 = st.columns([1, 2, 1])  # Center col2

with col2:
    selected_city = st.selectbox("City Name", sorted(city_names) if city_names else ["No cities found"])
    price_range = st.slider("Select Price Range", min_value=0, max_value=10000, value=(1000, 5000), step=100)
    amenities_input = st.text_input("Amenities (comma separated)", placeholder="e.g. wifi, TV, minibar")

    # Center-aligned button
    button_placeholder = st.empty()
    button_clicked = button_placeholder.button("🔍 Search with Filters", use_container_width=True)

if button_clicked:
    with st.spinner("Searching based on filters..."):
        amenities = [a.strip() for a in amenities_input.split(",")] if amenities_input else []
        try:
            results = recommend_filtered_hotels(
                spark_df=df,
                hotel_name=selected_hotel,
                city=selected_city,
                min_price=price_range[0],
                max_price=price_range[1],
                amenity_keywords=amenities,
                top_n=10
            )

            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            if isinstance(results, str):
                st.warning(results)
            else:
                # Rename columns for better readability
                results = results.rename(columns={
                    "property_name": "Hotel Name",
                    "area": "Location",
                    "price": "Price (₹)",
                    "similarity": "Match %"
                })
                results["Match %"] = (results["Match %"] * 100).round(2)
                st.success("Here are the filtered hotel recommendations:")
                st.table(results)
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")


# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>Powered by Smart Recommendations Engine</div>", unsafe_allow_html=True)
