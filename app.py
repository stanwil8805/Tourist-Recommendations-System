import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load("tourist_model.pkl")
category_encoder = joblib.load(
    "category_encoder.pkl"
)
district_encoder = joblib.load(
    "district_encoder.pkl"
)

df = pd.read_csv("Tourist.csv")

st.set_page_config(
    page_title="Tamil Nadu Tourism Recommendation",
    page_icon="🏖️",
    layout="wide"
)

st.title(
    "🏖️ Tamil Nadu Tourism Recommendation System"
)

st.markdown(
    "AI Powered Tourist Place Recommendation"
)

category = st.selectbox(
    "Category",
    sorted(df["Category"].unique())
)

district = st.selectbox(
    "District",
    sorted(df["District"].unique())
)

rating = st.slider(
    "Rating",
    1,
    5,
    4
)

budget = st.slider(
    "Budget",
    100,
    10000,
    3000
)

distance = st.slider(
    "Travel Distance",
    1,
    500,
    100
)

if st.button("Recommend Place"):

    category_encoded = category_encoder.transform(
        [category]
    )[0]

    district_encoded = district_encoder.transform(
        [district]
    )[0]

    prediction = model.predict(
        [[
            category_encoded,
            district_encoded,
            rating,
            budget,
            distance
        ]]
    )

    if prediction[0] == 1:

        st.success(
            "✅ Recommended Tourist Destination"
        )

        result = df[
            (df["Category"] == category)
            &
            (df["District"] == district)
        ]

        st.dataframe(result)

    else:

        st.error(
            "❌ Not Recommended"
        )

st.markdown("---")

st.subheader("Available Tourist Places")

st.dataframe(df)