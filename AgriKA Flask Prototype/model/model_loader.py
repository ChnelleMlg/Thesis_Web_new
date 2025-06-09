import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from model.db import store_prediction_result

class ModelLoader:
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    def __init__(self, merged_data):
        self.merged_data = merged_data
        self.weather_df = []
        self.yield_results = []

        # Load Model
        modelpath = r"C:\Users\ASUS\OneDrive\Documents\GitHub\Thesis_Web_new\AgriKA Flask Prototype\model\model_v2.keras"
        #modelpath = os.path.join(os.getcwd(), "model", "model_v2.keras")

        self.model = load_model(modelpath)
        self.weather_features = ["Temperature (Celsius)", "Day", "Month"]
        self.scaler = MinMaxScaler()

    def fit_scaler(self):
        """Fits the scaler to the temperature data in merged_data."""

        merged_df = pd.DataFrame(self.merged_data)
        print("🔍 DEBUG: Available columns:", merged_df.columns.tolist())

        # ✅ Check if expected columns are missing
        missing_cols = [col for col in self.weather_features if col not in merged_df.columns]
        if missing_cols:
            print(f"❌ ERROR: Missing required weather columns: {missing_cols}")
            return  
        
        ######
        weather_df = merged_df[self.weather_features]
        self.scaler.fit(weather_df)

        for entry in self.merged_data:
            city = entry["City/Municipality"]
            day = entry["Day"]
            month = entry["Month"]
            weather_data = [entry.get(feature, 0) for feature in self.weather_features]
            image_features = entry["Image_Features"]

            try:
                features = self.preprocess_input(city, day, month, weather_data, image_features)
                print("features: ", features)
                predicted_yield = self.model.predict(features)[0][0]
                print("PREDICTED YIELD: ", predicted_yield)

                result = {
                    "City": city,
                    "Day": day,
                    "Month": month,
                    "Predicted Yield": predicted_yield
                }
                self.yield_results.append(result)
                store_prediction_result(result)  # Store in database

            except Exception as e:
                print(f"Error processing entry {entry}: {e}")

            print(f"Yield Results: {self.yield_results}")
            print("✅ Yield prediction process completed.")

    def preprocess_input(self, city, day, month, weather_data, image_features):
        """
        Prepares a new input sample for yield prediction.
        """
        # Convert temperature data into a DataFrame
        weather_df = pd.DataFrame([weather_data], columns=self.weather_features)

        # Scale the temperature
        weather_scaled = self.scaler.transform(weather_df).flatten()  
        
        if (month == 3 and day >= 16) or (4 <= month <= 9 and (month != 9 or day <= 15)):
            # Second Cycle
            if (month == 3 and day >= 16) or month == 4 or (month == 5 and day <= 15):
                phase = 1  # Phase 1
            elif (month == 5 and day >= 16) or month == 6 or (month == 7 and day <= 15):
                phase = 2  # Phase 2
            elif (month == 7 and day >= 16) or month == 8 or (month == 9 and day <= 15):
                phase = 3  # Phase 3
        else:
            # First Cycle (September 16 – March 15)
            if (month == 9 and day >= 16) or month == 10 or (month == 11 and day <= 15):
                phase = 1  # Phase 1
            elif (month == 11 and day >= 16) or month == 12 or (month == 1 and day <= 15):
                phase = 2  # Phase 2
            elif (month == 1 and day >= 16) or month == 2 or (month == 3 and day <= 15):
                phase = 3  # Phase 3

        green_ratio = next((entry["Green_Ratio"] for entry in self.merged_data 
                            if entry["City/Municipality"] == city and 
                            entry["Day"] == day and 
                            entry["Month"] == month), 0)

        
        print("phase: ", phase)
        # final_features = np.append(weather_scaled, green_ratio, phase)
        # final_features = np.append(weather_scaled, [green_ratio, phase])
        extra_features = np.array([green_ratio, phase])
        print("Extra Features: ", extra_features)
        final_features = np.concatenate([weather_scaled, extra_features])
        print("final features: ", final_features)

        # Combine image features and scaled temperature
        X_new = np.hstack([image_features, final_features])

        # Reshape into sequence format expected by LSTM
        num_timesteps = 3
        feature_dim = X_new.shape[0]
        X_seq = np.zeros((1, num_timesteps, feature_dim))
        X_seq[0, -1, :] = X_new  # Only last timestep has data

        return X_seq