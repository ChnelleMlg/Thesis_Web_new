import numpy as np
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime


class variableCollector:
    def __init__(self, ndvi_images, selected_dates):
        self.ndvi_images = ndvi_images
        self.selected_dates = selected_dates
        self.weather_data = {}
        self.image_features = []
        self.merged_data = []
        self.resnet = ResNet50(weights="imagenet", include_top=False, pooling='avg')
        self.city_coordinates = self._initialize_city_coordinates()

    def _initialize_city_coordinates(self):
        return {
            "Alaminos": {"latitude": 14.0616, "longitude": 121.2604},
            "Bay": {"latitude": 14.1320, "longitude": 121.2569},
            "Cabuyao": {"latitude": 14.2471, "longitude": 121.1367},
            "Calauan": {"latitude": 14.1384, "longitude": 121.3198},
            "Cavinti": {"latitude": 14.2647, "longitude": 121.5455},
            "Binan": {"latitude": 14.3036, "longitude": 121.0781},
            "Calamba": {"latitude": 14.2127, "longitude": 121.1639},
            "Santa Rosa": {"latitude": 14.2843, "longitude": 121.0889},
            "Famy": {"latitude": 14.4730, "longitude": 121.4842},
            "Kalayaan": {"latitude": 14.3313, "longitude": 121.5484},
            "Liliw": {"latitude": 14.1364, "longitude": 121.4399},
            "Los Banos": {"latitude": 14.1699, "longitude": 121.2441},
            "Luisiana": {"latitude": 14.1908, "longitude": 121.5256},
            "Lumban": {"latitude": 14.2956, "longitude": 121.4962},
            "Mabitac": {"latitude": 14.4338, "longitude": 121.4113},
            "Magdalena": {"latitude": 14.2041, "longitude": 121.4342},
            "Majayjay": {"latitude": 14.1447, "longitude": 121.4723},
            "Nagcarlan": {"latitude": 14.1490, "longitude": 121.3885},
            "Paete": {"latitude": 14.3675, "longitude": 121.5300},
            "Pagsanjan": {"latitude": 14.2624, "longitude": 121.4570},
            "Pakil": {"latitude": 14.3800, "longitude": 121.4765},
            "Pangil": {"latitude": 14.4074, "longitude": 121.4856},
            "Pila": {"latitude": 14.2346, "longitude": 121.3656},
            "Rizal": {"latitude": 14.0841, "longitude": 121.4113},
            "San Pablo": {"latitude": 14.0642, "longitude": 121.3233},
            "Santa Cruz": {"latitude": 14.2691, "longitude": 121.4113},
            "Santa Maria": {"latitude": 14.5129, "longitude": 121.4342},
            "Siniloan": {"latitude": 14.4383, "longitude": 121.4856},
            "Victoria": {"latitude": 14.2028, "longitude": 121.3370},
            }

    def extract_green_area(self, image):
        # Remove black background pixels
        non_black_mask = np.any(image != [0, 0, 0], axis=-1)

        # Check if image is already RGB (skip conversion if so)
        if image.shape[-1] == 3:
            img_rgb = image
        else:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        lower_green = np.array([0, 60, 0])
        upper_green = np.array([190, 210, 100])

        # Create mask for green areas
        green_mask = cv2.inRange(img_rgb, lower_green, upper_green)

        lower_dgreen = np.array([0, 60, 0])
        upper_dgreen = np.array([5, 75, 5])
        dgreen_mask = cv2.inRange(img_rgb, lower_dgreen, upper_dgreen)

        lower_pink = np.array([248, 153, 151])
        upper_pink = np.array([237, 13, 182])
        pink_mask = cv2.inRange(img_rgb, lower_pink, upper_pink)

        lower_yellow = np.array([251, 207, 159])
        upper_yellow = np.array([254, 255, 255])
        yellow_mask = cv2.inRange(img_rgb, lower_yellow, upper_yellow)

        exclusion_mask = cv2.bitwise_or(dgreen_mask, pink_mask)
        exclusion_mask = cv2.bitwise_or(exclusion_mask, yellow_mask)
        refined_mask = cv2.bitwise_and(green_mask, cv2.bitwise_not(exclusion_mask))

        valid_pixels = np.sum(non_black_mask)
        if valid_pixels == 0:
            return 0.0  # Avoid division by zero

        green_ratio = np.sum(refined_mask == 255) / valid_pixels
        print(f"Green Ratio: {green_ratio:.2%}")

        return green_ratio

    def extract_features(self):
        for entry in self.ndvi_images:
            img = entry["Image_Array"]

            img_resized = cv2.resize(img, (224, 224))
            if img.shape[-1] == 4:
                img_resized = img_resized[:, :, :3]
            green_ratio = self.extract_green_area(img_resized)

            img_array = preprocess_input(img_to_array(img_resized))
            features = self.resnet.predict(np.expand_dims(img_array, axis=0), verbose=0)

            self.image_features.append({
                "City/Municipality": entry["City/Municipality"],
                "Date": entry["Date"],
                "Green_Ratio": green_ratio,
                "Image_Features": features.flatten()
            })
        self.fetch_weather_data()
        self.merge_data()

    def fetch_weather_data(self):
        self.weather_data.clear()

        self.city_coordinates = {k.upper(): v for k, v in self.city_coordinates.items()}

        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        for city, date_str in self.selected_dates.items():
            print(city, date_str)
            city_info = self.city_coordinates.get(city.upper())
            print(city_info)
            if not city_info:
                continue

            latitude, longitude = city_info["latitude"], city_info["longitude"]
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": date_str,
                "end_date": date_str,
                "hourly": ["relative_humidity_2m"],
                "daily": ["temperature_2m_mean", "precipitation_sum"],
                "timezone": "Asia/Singapore"
            }

            try:
                response = openmeteo.weather_api(url, params=params)[0]
                daily = response.Daily()
                hourly = response.Hourly()

                self.weather_data[city] = {
                    "date": date_str,
                    "temperature": daily.Variables(0).ValuesAsNumpy()[0],
                    "rainfall": daily.Variables(1).ValuesAsNumpy()[0],
                    "humidity": np.mean(hourly.Variables(0).ValuesAsNumpy())
                }
            except Exception as e:
                self.weather_data[city] = {"date": date_str, "temperature": None, "rainfall": None, "humidity": None}

    def merge_data(self):
        for img_data in self.image_features:
            city, date_str = img_data['City/Municipality'], img_data['Date']
            if city in self.weather_data and self.weather_data[city]['date'] == date_str:
                weather = self.weather_data[city]
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")

                self.merged_data.append({
                    'City/Municipality': city,
                    'Temperature (Celsius)': round(weather['temperature'], 2),
                    'Day': date_obj.timetuple().tm_yday,
                    'Month': date_obj.month,
                    'Green_Ratio': img_data['Green_Ratio'],
                    'Image_Features': img_data['Image_Features']
                })
            else:
                print('not work')

    def get_merged_data(self):
        return self.merged_data
