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
        modelpath = r"C:\Users\perli\Desktop\AgriKA Web\AgriKA\Thesis_Web_new\AgriKA Flask Prototype\model\model_v2.keras"
        #modelpath = os.path.join(os.getcwd(), "model", "model_v2.keras")
        self.municipalities = {"ALAMINOS": 1,"BAY": 2,"CABUYAO": 3,"CALAUAN": 4,"CAVINTI": 5,"BINAN": 6,"CALAMBA": 7,
                              "FAMY": 8,"KALAYAAN": 9,"LILIW": 10,"LOS BANOS": 11,"LUISIANA": 12,"LUMBAN": 13,
                               "MABITAC": 14,"MAGDALENA": 15,"MAJAYJAY": 16,"NAGCARLAN": 17,"PAETE": 18,
                               "PAGSANJAN": 19,"PAKIL": 20,"PANGIL": 21,"PILA": 22,"RIZAL": 23,"SAN PABLO": 24,
                               "SANTA CRUZ": 25,"SANTA MARIA": 26,"SANTA ROSA": 27,"SINILOAN": 28,"VICTORIA": 29
                              }

        self.model = load_model(modelpath)
        self.weather_features = ["Temperature (Celsius)", "Day", "Month"]
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

            raw_city = entry["City/Municipality"]
            print("raw city: ", raw_city)

            city_key = str(raw_city).strip().upper()
            city_id = self.municipalities.get(city_key, 0)

            try:
                features = self.preprocess_input(city, city_id, day, month, weather_data, image_features)
                predicted_yield = self.model.predict(features)[0][0]

                result = {
                    "City": raw_city,
                    "Day": day,
                    "Month": month,
                    "Predicted Yield": predicted_yield
                }
                self.yield_results.append(result)
                store_prediction_result(result)

            except Exception as e:
                print(f"Error processing entry {entry}: {e}")

            print(f"Yield Results: {self.yield_results}")

    def preprocess_input(self, city, city_id, day, month, weather_data, image_features):
          weather_df = pd.DataFrame([weather_data], columns=self.weather_features)

          # Scale the temperature
          weather_scaled = self.scaler.transform(weather_df).flatten()

          # Determine Phase
          if (month == 3 and day >= 16) or (4 <= month <= 9 and (month != 9 or day <= 15)):
              # Second Cycle
              if (month == 3 and day >= 16) or month == 4 or (month == 5 and day <= 15):
                  phase = 1
              elif (month == 5 and day >= 16) or month == 6 or (month == 7 and day <= 15):
                  phase = 2
              elif (month == 7 and day >= 16) or month == 8 or (month == 9 and day <= 15):
                  phase = 3
          else:
              # First Cycle
              if (month == 9 and day >= 16) or month == 10 or (month == 11 and day <= 15):
                  phase = 1
              elif (month == 11 and day >= 16) or month == 12 or (month == 1 and day <= 15):
                  phase = 2
              elif (month == 1 and day >= 16) or month == 2 or (month == 3 and day <= 15):
                  phase = 3

          # Get green ratio
          green_ratio = next((entry["Green_Ratio"] for entry in self.merged_data
                              if entry["City/Municipality"] == city and
                              entry["Day"] == day and
                              entry["Month"] == month), 0)

          green_ratio = np.array([green_ratio])
          city_id = np.array([city_id])      # already converted to integer
          phase_array = np.array([phase]) # also needs to be added

          # Combine all features
          final_features = np.concatenate([image_features, weather_scaled, green_ratio, city_id, phase_array])
          print("\n\n\nfinal features: ", image_features, weather_scaled, green_ratio, city_id, phase_array)

          # Reshape into sequence format expected by LSTM
          X_seq = np.zeros((1, 3, final_features.shape[0]))
          X_seq[0, -1, :] = final_features  # only last timestep has data

          return X_seq