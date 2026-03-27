import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_excel("reduced_dataset/RecommendationDataset.xlsx")

print("Columns:", df.columns)

# Drop ID column
df = df.drop(columns=["Pothole_ID"])

# Encode categorical input columns
le_road = LabelEncoder()
le_traffic = LabelEncoder()
le_weather = LabelEncoder()
le_rain = LabelEncoder()

df["Road_Type"] = le_road.fit_transform(df["Road_Type"])
df["Traffic_Intensity"] = le_traffic.fit_transform(df["Traffic_Intensity"])
df["Weather_Condition"] = le_weather.fit_transform(df["Weather_Condition"])
df["Rainfall_Level"] = le_rain.fit_transform(df["Rainfall_Level"])

# Encode outputs
le_material = LabelEncoder()
le_method = LabelEncoder()

df["Recommended_Material"] = le_material.fit_transform(df["Recommended_Material"])
df["Repair_Method"] = le_method.fit_transform(df["Repair_Method"])

# Features (X)
X = df[[
    "Road_Type",
    "Pothole_Depth_cm",
    "Pothole_Diameter_cm",
    "Traffic_Intensity",
    "Weather_Condition",
    "Rainfall_Level"
]]

# Targets
y_material = df["Recommended_Material"]
y_method = df["Repair_Method"]
y_durability = df["Expected_Durability_Months"]

# Split
X_train, X_test, y_material_train, y_material_test = train_test_split(
    X, y_material, test_size=0.2, random_state=42
)

_, _, y_method_train, y_method_test = train_test_split(
    X, y_method, test_size=0.2, random_state=42
)

_, _, y_durability_train, y_durability_test = train_test_split(
    X, y_durability, test_size=0.2, random_state=42
)

# Models
material_model = RandomForestClassifier()
method_model = RandomForestClassifier()
durability_model = RandomForestRegressor()

# Train
material_model.fit(X_train, y_material_train)
method_model.fit(X_train, y_method_train)
durability_model.fit(X_train, y_durability_train)

# Save everything
model_data = {
    "material_model": material_model,
    "method_model": method_model,
    "durability_model": durability_model,
    "le_road": le_road,
    "le_traffic": le_traffic,
    "le_weather": le_weather,
    "le_rain": le_rain,
    "le_material": le_material,
    "le_method": le_method
}

with open("pothole_recommender_v2.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("Model trained and saved successfully!")
