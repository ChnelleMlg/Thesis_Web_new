# Install required packages (run these in the terminal or notebook separately)
# Standard library imports
import os
import json
import re
from datetime import datetime, timedelta

# Third-party imports
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.image import imsave
from PIL import Image
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.validation import explain_validity

# SentinelHub imports
from sentinelhub import (
    SHConfig, SentinelHubCatalog, SentinelHubRequest, DataCollection, MimeType,
    Geometry, bbox_to_dimensions, CRS, BBox
)

# OpenMeteo imports
import openmeteo_requests
import requests_cache
from retry_requests import retry

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input


def load_and_process_geojson(filepath):
    """Loads a GeoJSON file, normalizes city names, and organizes polygons by municipality."""

    # Load GeoJSON data
    with open(filepath, "r") as f:
        geojson_data = json.load(f)

    # Define municipalities
    municipalities = {
        "ALAMINOS", "BAY", "CABUYAO", "CALAUAN", "CAVINTI", "BIÑAN", "CALAMBA",
        "SAN PEDRO", "SANTA ROSA", "FAMY", "KALAYAAN", "LILIW", "LOS BAÑOS", "LUISIANA",
        "LUMBAN", "MABITAC", "MAGDALENA", "MAJAYJAY", "NAGCARLAN", "PAETE", "PAGSANJAN", "PAKIL", "PANGIL",
        "PILA", "RIZAL", "SAN PABLO", "SANTA CRUZ", "SANTA MARIA", "SINILOAN", "VICTORIA"
    }

    # Collect polygons per municipality
    municipality_polygons = {m: [] for m in municipalities}

    # Normalize city names to uppercase
    for feature in geojson_data["features"]:
        feature["properties"]["city"] = feature["properties"]["city"].upper()

    for feature in geojson_data["features"]:
        city = feature["properties"]["city"]

        try:
            geometry = feature["geometry"]

            # Ensure coordinates are properly formatted
            if geometry["type"] == "Polygon" and isinstance(geometry["coordinates"], list):
                if isinstance(geometry["coordinates"][0], list) and isinstance(geometry["coordinates"][0][0], float):
                    geometry["coordinates"] = [geometry["coordinates"]]  # Wrap in a list

            # Convert to Shapely object and store if valid
            geom = shape(geometry)
            municipality_polygons[city].append(geom)

        except Exception as e:
            print(f"Error processing geometry for {city}: {e}")

    return municipality_polygons

# Example usage
filepath = r"C:\Users\ASUS\OneDrive\Documents\GitHub\Thesis_Web\AgriKA Flask Prototype\static\fields_coordinates.geojson"

# Update path if necessary
municipality_polygons = load_and_process_geojson(filepath)


# Sentinel Hub Configuration
config = SHConfig()
config.instance_id = '34fbb6f7-0161-4322-8fe7-87a3f49a95c4'
config.sh_client_id = '59b27267-9361-4628-865a-03ba9d0667aa'
config.sh_client_secret = 'GmLNYkGAgdd6h0x8xtR397N6AUzsluhx'

catalog = SentinelHubCatalog(config=config)

# Get the latest available date for the Sentinel-2 image
def get_latest_sentinel_date(geometry):
    search_iterator = catalog.search(
        DataCollection.SENTINEL2_L2A,
        geometry=geometry,
        time=("2023-01-01", datetime.utcnow().strftime("%Y-%m-%d")),  # Search up to today
        fields={"include": ["id", "properties.datetime"], "exclude": []}
    )

    results = list(search_iterator)

    if results:
        # Extract available dates and sort them manually
        dates = [result["properties"]["datetime"][:10] for result in results]  # Extract YYYY-MM-DD
        latest_date = max(dates)  # Get the most recent date
        return latest_date
    else:
        return None

selected_dates = {}

def apply_image_filters(image):
    """Enhance the image by applying brightness and sharpening filters."""
    brightness_filter = np.array([[0, 0, 0], [0, 4, 0], [0, 0, 0]])
    sharpness_filter = np.array([[0, -1, 0], [-1, 5.2, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, brightness_filter)
    return cv2.filter2D(image, -1, sharpness_filter)

def display_image(image, title):
    """Display an image with a title."""
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

def compute_black_pixel_ratio(image):
    """Calculate the ratio of black pixels in the image."""
    black_threshold = 30

    if image.shape[-1] == 4:
        alpha_channel = image[:, :, 3]
        visible_pixels = image[:, :, :3][alpha_channel > 0]
    else:
        visible_pixels = image[:, :, :3] if len(image.shape) == 3 else None

    if len(visible_pixels.shape) == 3:
        black_pixels = np.sum(
            (visible_pixels[:, :, 0] < black_threshold) &
            (visible_pixels[:, :, 1] < black_threshold) &
            (visible_pixels[:, :, 2] < black_threshold)
        )
        total_pixels = visible_pixels.shape[0] * visible_pixels.shape[1]
    else:
        black_pixels = np.sum(visible_pixels < black_threshold)
        total_pixels = visible_pixels.size

    return black_pixels / total_pixels if total_pixels > 0 else 0

def create_image_request(aoi_geometry, time_interval, config):
    """Generate a SentinelHubRequest for fetching satellite images."""
    evalscript = """
        //VERSION=3
        function setup() {
          return { input: ["B02", "B03", "B04", "SCL", "dataMask"], output: { bands: 4 } };
        }
        function evaluatePixel(sample) {
          if (sample.dataMask === 0) {
            return [0, 0, 0, 0];
          }
          if (sample.SCL === 7 || sample.SCL === 8 || sample.SCL === 9) {
            return [0, 0, 0, 255];
          }
          return [sample.B04, sample.B03, sample.B02, 255];
        }
    """
    return SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=time_interval,
            other_args={"dataFilter": {"mosaickingOrder": "leastCC"}}
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        geometry=aoi_geometry,
        config=config
    )

def get_DateOfImage(municipality, polygons, config):
    """Fetch the latest valid satellite image for a given municipality."""
    if not polygons:
        print(f"No polygons found for {municipality}")
        return

    aoi_geometry = Geometry(MultiPolygon(polygons), CRS.WGS84)
    latest_date = get_latest_sentinel_date(aoi_geometry)

    if not latest_date:
        print(f"No recent Sentinel-2 image found for {municipality}")
        return

    for attempt in range(60):
        start_date = (datetime.strptime(latest_date, "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")
        time_interval = (start_date, latest_date)
        #print(f"📅 Fetching image for {municipality} from {start_date} to {latest_date}")

        request = create_image_request(aoi_geometry, time_interval, config)
        raw_images = request.get_data()

        if raw_images:
            image = apply_image_filters(raw_images[0])
            #display_image(image, f"Fetched Image - {municipality} ({latest_date})")
            black_ratio = compute_black_pixel_ratio(image)
            #print(f"Black pixel ratio: {black_ratio:.2%}")

            if black_ratio < 0.30:
                for day in range(6):
                    check_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=day)).strftime("%Y-%m-%d")
                    #print(f"🔍 Checking image for {municipality} on {check_date}")

                    request_specific = create_image_request(aoi_geometry, (check_date, check_date), config)
                    specific_image = request_specific.get_data()

                    if specific_image:
                        image = apply_image_filters(specific_image[0])
                        #display_image(image, f"Checking Image - {municipality} ({check_date})")
                        black_ratio_new = compute_black_pixel_ratio(image)
                        #print(f"Black pixel ratio on {check_date}: {black_ratio_new:.2%}")

                        if abs(black_ratio_new - black_ratio) < 1e-3:
                            #print(f"✅ Using image from {check_date} (Same black pixel ratio)")
                            selected_dates[municipality] = check_date
                            #display_image(image, f"Final Image - {municipality} ({check_date})")
                            return selected_dates
                #print(f"⚠️ No valid image found within the range {start_date} to {latest_date}")

        latest_date = (datetime.strptime(latest_date, "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")

    #print(f"❌ No suitable image found for {municipality} after 60 attempts.")

'''
city = "BAY"  # Ensure the city name is in uppercase to match your dictionary keys

if city in municipality_polygons and municipality_polygons[city]:  # Check if it exists and has polygons
    print(f"✅ Found polygons for {city}, fetching image...")
    get_DateOfImage(city, municipality_polygons[city])
else:
    print(f"❌ No polygon data found for {city}. Check if the GeoJSON contains valid coordinates.")
'''
ndvi_images = []  # Store NDVI image data

def get_NDVI_Images(selected_dates, municipality_polygons):

    for city, date in selected_dates.items():
        print(f"Processing NDVI image for {city} on {date}")

        # Retrieve the correct polygon for the city
        polygons = municipality_polygons.get(city, [])
        if not polygons:
            print(f"⚠️ No polygons found for {city}, skipping NDVI processing.")
            continue

        # Convert polygons to SentinelHub Geometry
        geometry = Geometry(MultiPolygon(polygons), crs=CRS.WGS84)

        evalscript_ndvi = """
        //VERSION=3
        function setup() {
            return {
                input: ["B04", "B08", "dataMask"],
                output: { bands: 4 }
            };
        }

        function evaluatePixel(sample) {
            let val = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
            let imgVals = null;
            if (val < -1.1) imgVals = [0, 0, 0];
            else if (val < -0.2) imgVals = [0.75, 0.75, 0.75];
            else if (val < -0.1) imgVals = [0.86, 0.86, 0.86];
            else if (val < 0) imgVals = [1, 1, 0.88];
            else if (val < 0.025) imgVals = [1, 1, 0.5];
            else if (val < 0.05) imgVals = [0.93, 0.1, 0.71];
            else if (val < 0.075) imgVals = [0.87, 0.85, 0.61];
            else if (val < 0.1) imgVals = [0.8, 0.78, 0.51];
            else if (val < 0.125) imgVals = [0.74, 0.72, 0.42];
            else if (val < 0.15) imgVals = [0.69, 0.76, 0.38];
            else if (val < 0.175) imgVals = [0.64, 0.8, 0.35];
            else if (val < 0.2) imgVals = [0.57, 0.75, 0.32];
            else if (val < 0.25) imgVals = [0.5, 0.7, 0.28];
            else if (val < 0.3) imgVals = [0.44, 0.64, 0.25];
            else if (val < 0.35) imgVals = [0.38, 0.59, 0.21];
            else if (val < 0.4) imgVals = [0.31, 0.54, 0.18];
            else if (val < 0.45) imgVals = [0.25, 0.49, 0.14];
            else if (val < 0.5) imgVals = [0.19, 0.43, 0.11];
            else if (val < 0.55) imgVals = [0.13, 0.38, 0.07];
            else if (val < 0.6) imgVals = [0.06, 0.33, 0.04];
            else imgVals = [0, 0.27, 0];

            imgVals.push(sample.dataMask);
            return imgVals;
        }
        """

        request_ndvi_image = SentinelHubRequest(
            evalscript=evalscript_ndvi,
            input_data=[SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(date, date),
                other_args={"dataFilter": {"mosaickingOrder": "leastCC"}}
            )],
            responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
            geometry=geometry,
            size=(224, 224),  # Match ResNet50 input size
            config=config
        )

        ndvi_images_batch = request_ndvi_image.get_data()

        if not ndvi_images_batch:
            print(f"⚠️ No NDVI image available for {city} on {date}")
            continue

        ndvi_image = ndvi_images_batch[0]  # Extract the first image

        # Convert to NumPy array
        ndvi_image = np.array(ndvi_image)

        # Store data in structured format
        ndvi_images.append({
            "City/Municipality": city,
            "Date": date,
            "Image_Array": ndvi_image  # Store the actual image array
        })

        # Show the NDVI image
        '''plt.figure(figsize=(5, 5))
        plt.imshow(ndvi_image)
        plt.axis("off")
        plt.title(f"NDVI Image - {city} ({date})")
        plt.show()'''

    return ndvi_images  # List of dictionaries with city name, date, and image

for city, polygons in municipality_polygons.items():
 get_DateOfImage(city, polygons, config)

# Call function and store results
ndvi_images = get_NDVI_Images(selected_dates, municipality_polygons)

# Print collected data
print(f"Collected {len(ndvi_images)} NDVI images.")
print(selected_dates)

image_features = []

resnet = ResNet50(weights=r'C:\Users\ASUS\OneDrive\Documents\GitHub\Thesis_Web\AgriKA Flask Prototype\model\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                  include_top=False, pooling='avg')


def extract_green_area(image):
    #print(image)
    """Extracts the green area ratio from an image array."""
    img = cv2.resize(image, (224, 224))

    non_black_mask = np.any(img != [0, 0, 0], axis=-1)

    # Ensure image is in RGB format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # **NEW: Check unique colors in the image**
    unique_colors = np.unique(img_rgb.reshape(-1, 3), axis=0)
    #print("Unique Colors in Image:", unique_colors[:20])  # Show first 20 unique colors for debugging

    # **NEW: Adjusted RGB Range for Greens**
    lower_green = np.array([0, 60, 0])    # Loosen lower bound
    upper_green = np.array([176, 204, 100])  # Loosen upper bound
    green_mask = cv2.inRange(img_rgb, lower_green, upper_green)

    # Dark green exclusion
    lower_dgreen = np.array([0, 40, 0])
    upper_dgreen = np.array([10, 80, 10])
    dgreen_mask = cv2.inRange(img_rgb, lower_dgreen, upper_dgreen)

    # Exclude pink (soil/stressed crops)
    lower_pink = np.array([200, 0, 100])
    upper_pink = np.array([255, 150, 200])
    pink_mask = cv2.inRange(img_rgb, lower_pink, upper_pink)

    # Exclude yellow (dry areas)
    lower_yellow = np.array([200, 150, 0])
    upper_yellow = np.array([255, 255, 100])
    yellow_mask = cv2.inRange(img_rgb, lower_yellow, upper_yellow)

    # **NEW: Debug individual masks**
    '''plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(green_mask, cmap="gray")
    plt.title("Green Mask")

    plt.subplot(1, 3, 2)
    plt.imshow(dgreen_mask, cmap="gray")
    plt.title("Dark Green Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(pink_mask, cmap="gray")
    plt.title("Pink Mask")
    plt.show()'''

    # **NEW: Ensure exclusions are not removing all greens**
    exclusion_mask = cv2.bitwise_or(dgreen_mask, pink_mask)
    exclusion_mask = cv2.bitwise_or(exclusion_mask, yellow_mask)
    refined_mask = cv2.bitwise_and(green_mask, cv2.bitwise_not(exclusion_mask))

    # Count valid pixels
    valid_pixels = np.sum(non_black_mask)
    if valid_pixels == 0:
        return 0  # Avoid division by zero

    green_ratio = np.sum(refined_mask == 255) / valid_pixels

    # **NEW: Display final refined mask**
    '''plt.imshow(refined_mask, cmap="gray")
    plt.title("Refined Green Mask")
    plt.show()
    print(f"Valid Pixels: {valid_pixels}")'''

    return green_ratio


def batch_extract_features(image_entries, batch_size=32):
    """Extracts features and green area ratios for a batch of images."""
    features = []
    green_ratios = []

    for i in range(0, len(image_entries), batch_size):
        batch_images = [entry["Image_Array"] for entry in image_entries[i : i + batch_size]]  # ✅ Extract only the image data
        batch_preprocessed = []
        batch_green_areas = []

        for img_array in batch_images:
            img_array_resized = cv2.resize(img_array, (224, 224))  # ✅ Resize for consistency

            # ✅ Extract green ratio BEFORE preprocessing
            green_ratio = extract_green_area(img_array_resized)
            batch_green_areas.append(green_ratio)
            green_ratios.append(green_ratio)

            # ✅ Convert for ResNet AFTER green extraction
            img_array_resized = img_to_array(img_array_resized)
            img_array_resized = preprocess_input(img_array_resized)

            batch_preprocessed.append(img_array_resized)

        batch_preprocessed = np.array(batch_preprocessed)
        batch_features = resnet.predict(batch_preprocessed, verbose=0)
        batch_green_areas = np.array(batch_green_areas).reshape(-1, 1)

        features.extend(batch_features)

    return np.array(features), np.array(green_ratios)


# Retrieve NDVI images with city names and dates
ndvi_images=[]
ndvi_images.clear()
ndvi_images = get_NDVI_Images(selected_dates, municipality_polygons)

print("Before clearing:", len(image_features))  # Check length before clearing
image_features.clear()
print("After clearing:", len(image_features))   # Should be 0

if ndvi_images:
    # Ensure images have 3 channels (ignore alpha channel if present)
    for entry in ndvi_images:
        entry["Image_Array"] = entry["Image_Array"][..., :3]

    # Extract features and green ratios
    features, green_ratios = batch_extract_features(ndvi_images)

    # Store data in the required format
    for i, entry in enumerate(ndvi_images):
        image_features.append({
            "City/Municipality": entry["City/Municipality"],  # ✅ Use stored city name
            "Date": entry["Date"],  # ✅ Use stored date
            "Green_Ratio": green_ratios[i],
            "Image_Features": features[i].tolist()
        })

    '''print("\nFeature extraction complete. Here is the collected data:\n")
    for entry in image_features[:5]:  # Print first 5 entries for debugging
        print(entry)'''
else:
    print("No NDVI images to process.")

#print(image_features)

# Dictionary to store weather data
weather_data = {}

def fetch_weather_data(selected_dates):
    weather_data.clear()
    """
    Fetches weather data (temperature, rainfall, humidity) for each municipality and date.

    Args:
        selected_dates (dict): Dictionary with municipality names as keys and dates as values.

    Returns:
        dict: Weather data for each municipality and date.
    """

    # Define city coordinates
    city_coordinates = {
        "Alaminos": {"latitude": 14.0616, "longitude": 121.2604},
        "Bay": {"latitude": 14.1320, "longitude": 121.2569},
        "Cabuyao": {"latitude": 14.2471, "longitude": 121.1367},
        "Calauan": {"latitude": 14.1384, "longitude": 121.3198},
        "Cavinti": {"latitude": 14.2647, "longitude": 121.5455},
        "Biñan": {"latitude": 14.3036, "longitude": 121.0781},
        "Calamba": {"latitude": 14.2127, "longitude": 121.1639},
        "Santa Rosa": {"latitude": 14.2843, "longitude": 121.0889},
        "Famy": {"latitude": 14.4730, "longitude": 121.4842},
        "Kalayaan": {"latitude": 14.3313, "longitude": 121.5484},
        "Liliw": {"latitude": 14.1364, "longitude": 121.4399},
        "Los Baños": {"latitude": 14.1699, "longitude": 121.2441},
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

    # Convert municipality names to uppercase for matching
    city_coordinates = {k.upper(): v for k, v in city_coordinates.items()}

    # Initialize API session
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    for municipality, date_str in selected_dates.items():
        try:
            city_info = city_coordinates.get(municipality.upper())
            if not city_info:
                print(f"⚠️ No coordinates found for {municipality}")
                continue

            latitude = city_info["latitude"]
            longitude = city_info["longitude"]

            # Open-Meteo API request
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

            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]  # Extract first response

            # Extract weather data
            daily = response.Daily()
            temperature = daily.Variables(0).ValuesAsNumpy()[0]  # Temperature
            rainfall = daily.Variables(1).ValuesAsNumpy()[0]  # Precipitation

            hourly = response.Hourly()
            humidity_values = hourly.Variables(0).ValuesAsNumpy()
            mean_humidity = sum(humidity_values) / len(humidity_values)  # Avg humidity

            # Store in dictionary
            weather_data[municipality] = {
                "date": date_str,
                "temperature": temperature,
                "rainfall": rainfall,
                "humidity": mean_humidity
            }

        except Exception as e:
            print(f"❌ Error fetching weather for {municipality} on {date_str}: {e}")
            weather_data[municipality] = {
                "date": date_str,
                "temperature": None,
                "rainfall": None,
                "humidity": None
            }


    return weather_data

weather_info = fetch_weather_data(selected_dates)
print(weather_data)

# Process and merge data
merged_data = []

def merge_image_weather_data(image_features, weather_data):
    """Merges image features with weather data based on city and date."""
    merged_data.clear()

    for img_data in image_features:
        city = img_data['City/Municipality']
        date_str = img_data['Date']

        if city in weather_data and weather_data[city]['date'] == date_str:
            weather = weather_data[city]
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")

            city = img_data['City/Municipality'].title()
            merged_data.append({
                'City/Municipality': city,
                'Temperature (Celcius)': round(weather['temperature'], 2),
                'Rainfall (mm)': round(weather['rainfall'], 2),
                'Humidity (%)': round(weather['humidity'], 2),
                'Day': date_obj.day,
                'Month': date_obj.month,
                'Green_Ratio': img_data['Green_Ratio'],
                'Image_Features': img_data['Image_Features']  # Keep as list or convert if needed
            })

    return merged_data

merge_image_weather_data(image_features, weather_data)
# Print the result
print("Total merged entries:", len(merged_data))
print(merged_data)


# Load Model
model = load_model(r"C:\Users\ASUS\OneDrive\Documents\GitHub\Thesis_Web\AgriKA Flask Prototype\model\cnn_lstm_yield_prediction_model.h5")

# Define weather feature (only Temperature)
weather_features = ["Temperature (Celcius)"]

# Convert merged_data into a DataFrame
merged_df = pd.DataFrame(merged_data)

# Extract only Temperature
weather_df = merged_df[weather_features]

# Initialize and fit the scaler on the temperature data
scaler = MinMaxScaler()
scaler.fit(weather_df)  # Fit only on Temperature ✅

def preprocess_input(city, day, month, weather_data, image_features):
    """
    Prepares a new input sample for yield prediction.
    """
    # Convert temperature data into a DataFrame
    weather_df = pd.DataFrame([[weather_data]], columns=weather_features)

    # Scale the temperature
    weather_scaled = scaler.transform(weather_df).flatten()  # ✅ No mismatch errors!

    # Combine image features and scaled temperature
    X_new = np.hstack([image_features, weather_scaled])

    # Reshape into sequence format expected by LSTM
    num_timesteps = 3
    feature_dim = X_new.shape[0]
    X_seq = np.zeros((1, num_timesteps, feature_dim))
    X_seq[0, -1, :] = X_new  # Only last timestep has data

    return X_seq

# List to store prediction results
yield_results = []

for entry in merged_data:
    city = entry["City/Municipality"]
    day = entry["Day"]
    month = entry["Month"]

    # Extract only Temperature
    weather_data = entry.get("Temperature (Celcius)", 0)  # Default to 0 if missing

    # Extract image features
    image_features = entry["Image_Features"]

    # Preprocess the input
    try:
        features = preprocess_input(city, day, month, weather_data, image_features)
        print("Final feature shape:", features.shape)  # Should print (1, 3, image_dim + 1)

        # Predict
        predicted_yield = model.predict(features)[0][0]

        # Store result
        yield_results.append({
            "City": city,
            "Day": day,
            "Month": month,
            "Predicted Yield": predicted_yield
        })
    except Exception as e:
        print(f"Error processing entry {entry}: {e}")

# Print results
for result in yield_results:
    print(result)

