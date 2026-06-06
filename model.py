import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("Tourist.csv")

# Convert Yes/No to numbers
df["would_recommend"] = df["would_recommend"].map({
    "Yes":1,
    "No":0
})

# Encode text columns
le_category = LabelEncoder()
le_district = LabelEncoder()

df["Category"] = le_category.fit_transform(df["Category"])
df["District"] = le_district.fit_transform(df["District"])

# Features
X = df[
    [
        "Category",
        "District",
        "Rating",
        "Budget",
        "Travel_Distance"
    ]
]

# Target
y = df["would_recommend"]

# Train model
model = RandomForestClassifier()

model.fit(X, y)

# Save model
joblib.dump(model, "tourist_model.pkl")

joblib.dump(le_category, "category_encoder.pkl")
joblib.dump(le_district, "district_encoder.pkl")

print("Model Saved Successfully!")