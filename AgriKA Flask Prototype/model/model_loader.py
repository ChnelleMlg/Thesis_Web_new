import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from model.db import store_prediction_result

class ModelLoader:

    def __init__(self, merged_data):
        self.merged_data = merged_data
        self.weather_df = []
        self.yield_results = []

        # Load Model
        modelpath = r"C:\Users\perli\Desktop\AgriKA Web\AgriKA\Thesis_Web_new\AgriKA Flask Prototype\model\model_v2.keras"
        #modelpath = os.path.join(os.getcwd(), "model", "model_v2.keras")

        self.model = load_model(modelpath)
        self.weather_features = ["Temperature (Celsius)", "Rainfall (mm)", "Humidity (%)"]
        self.scaler = MinMaxScaler()

    def fit_scaler(self):
        """Fits the scaler to the temperature data in merged_data."""

        merged_df = pd.DataFrame(self.merged_data)
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
        weather_scaled = self.scaler.transform(weather_df).flatten()  # ✅ No mismatch errors!

        green_ratio = next((entry["Green_Ratio"] for entry in self.merged_data 
                        if entry["City/Municipality"] == city and 
                           entry["Day"] == day and 
                           entry["Month"] == month), 0)
        
        final_features = np.append(weather_scaled, green_ratio)
        print(final_features)

        # Combine image features and scaled temperature
        X_new = np.hstack([image_features, final_features])

        # Reshape into sequence format expected by LSTM
        num_timesteps = 3
        feature_dim = X_new.shape[0]
        X_seq = np.zeros((1, num_timesteps, feature_dim))
        X_seq[0, -1, :] = X_new  # Only last timestep has data

        return X_seq