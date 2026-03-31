import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("electricity_data.csv")

data.columns = data.columns.str.strip()

data["Dates"] = pd.to_datetime(data["Dates"], dayfirst=True)

# Extract year and month
data["Year"] = data["Dates"].dt.year
data["Month"] = data["Dates"].dt.month

# NEW FEATURE (helps future prediction)
data["Year_index"] = data["Year"] - data["Year"].min()

# Encode
state_encoder = LabelEncoder()
region_encoder = LabelEncoder()

data["State_encoded"] = state_encoder.fit_transform(data["States"])
data["Region_encoded"] = region_encoder.fit_transform(data["Regions"])

# Features
X = data[[
"Year",
"Month",
"latitude",
"longitude",
"State_encoded",
"Region_encoded",
"Year_index"
]]

# Target
y = data["Usage"]

model = RandomForestRegressor(n_estimators=200)

model.fit(X, y)

# Save
joblib.dump(model,"electricity_model.pkl")
joblib.dump(state_encoder,"state_encoder.pkl")
joblib.dump(region_encoder,"region_encoder.pkl")

print("Model trained successfully!")