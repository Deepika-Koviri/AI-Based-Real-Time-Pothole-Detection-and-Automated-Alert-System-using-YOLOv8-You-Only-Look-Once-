import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

class PotholeRepairRecommender:
    def __init__(self):
        self.material_encoder = LabelEncoder()
        self.equipment_encoder = LabelEncoder()
        self.model_cost = None
        self.model_labor = None
        self.model_time = None
        
    def train(self, filepath='pothole_repair_data.csv'):
        df = pd.read_csv(filepath)
        
        # Encode categorical features
        df['severity_encoded'] = LabelEncoder().fit_transform(df['severity'])
        df['road_type_encoded'] = LabelEncoder().fit_transform(df['road_type'])
        df['traffic_level_encoded'] = LabelEncoder().fit_transform(df['traffic_level'])
        
        features = ['volume_cm3', 'severity_encoded', 'pothole_count', 
                   'road_type_encoded', 'traffic_level_encoded']
        
        # Train models
        self.model_cost_min = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_cost_max = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_unskilled = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_skilled = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_time = RandomForestRegressor(n_estimators=100, random_state=42)
        
        X = df[features]
        self.model_cost_min.fit(X, df['cost_per_m3_min'])
        self.model_cost_max.fit(X, df['cost_per_m3_max'])
        self.model_unskilled.fit(X, df['unskilled_labor'])
        self.model_skilled.fit(X, df['skilled_labor'])
        self.model_time.fit(X, df['repair_time_hours'])
        
        # Store encoders
        self.material_encoder.fit(df['optimal_material'])
        self.equipment_encoder.fit(df['optimal_equipment'])
        
        joblib.dump(self, 'pothole_recommender.pkl')
        print("✅ Model trained and saved!")
    
    def recommend(self, volume_cm3, severity, pothole_count=1, road_type='urban', traffic_level='medium'):
        # Encode inputs
        severity_map = {'Small': 0, 'Medium': 1, 'Large': 2}
        road_map = {'urban': 0, 'rural': 1, 'highway': 2}
        traffic_map = {'low': 0, 'medium': 1, 'high': 2}
        
        features = np.array([[volume_cm3, severity_map[severity], pothole_count,
                            road_map[road_type], traffic_map[traffic_level]]])
        
        # Predictions
        cost_min = int(self.model_cost_min.predict(features)[0])
        cost_max = int(self.model_cost_max.predict(features)[0])
        unskilled = int(self.model_unskilled.predict(features)[0])
        skilled = int(self.model_skilled.predict(features)[0])
        time_hours = round(self.model_time.predict(features)[0], 1)
        
        total_volume_m3 = volume_cm3 / 1e6
        
        return {
            'pothole_type': severity,
            'volume_cm3': volume_cm3,
            'total_volume_m3': round(total_volume_m3, 2),
            'recommended_unskilled': unskilled,
            'recommended_skilled': skilled,
            'recommended_supervisors': 1 if unskilled+skilled > 3 else 0,
            'cost_per_m3_min': cost_min,
            'cost_per_m3_max': cost_max,
            'total_cost_min': int(total_volume_m3 * cost_min),
            'total_cost_max': int(total_volume_m3 * cost_max),
            'estimated_time_hours': time_hours,
            'optimal_material': self.material_encoder.inverse_transform([1])[0],  # Hot mix
            'optimal_equipment': self.equipment_encoder.inverse_transform([1])[0]  # Compactor
        }

# Train the model (run once)
if __name__ == "__main__":
    recommender = PotholeRepairRecommender()
    recommender.train()
