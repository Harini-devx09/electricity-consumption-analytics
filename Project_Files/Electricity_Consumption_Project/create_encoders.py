import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# load dataset
df = pd.read_csv("electricity_data.csv")

# create encoders
region_encoder = LabelEncoder()
state_encoder = LabelEncoder()

df["Region_encoded"] = region_encoder.fit_transform(df["Regions"])
df["State_encoded"] = state_encoder.fit_transform(df["States"])

# save encoders
pickle.dump(region_encoder, open("region_encoder.pkl", "wb"))
pickle.dump(state_encoder, open("state_encoder.pkl", "wb"))

print("Encoders saved successfully!")