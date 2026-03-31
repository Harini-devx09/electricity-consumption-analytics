import joblib
import pandas as pd

# Load trained model
model = joblib.load("electricity_model.pkl")

# Example future input
data = pd.DataFrame({
    "Year": [2025],
    "Month": [6]
})

# Predict electricity usage
prediction = model.predict(data)

print("Predicted Electricity Consumption:", prediction[0])