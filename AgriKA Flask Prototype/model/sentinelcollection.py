import json
from model.db import get_phase
from shapely.geometry import shape, MultiPolygon
from sentinelhub import (
    SHConfig, SentinelHubCatalog, SentinelHubRequest, DataCollection, MimeType,
    Geometry, bbox_to_dimensions, CRS, BBox
)
from datetime import datetime, timedelta
import numpy as np
import cv2

class SentinelImageGet:
    def __init__(self, geojson_path, config):
        self.geojson_path = geojson_path
        self.config = config
        self.catalog = SentinelHubCatalog(self.config)
        self.ndvi_images = []
        self.selected_dates = {}

    def load_and_process_geojson(self):
        with open(self.geojson_path, "r") as f:
            geojson_data = json.load(f)

        # Define municipalities
        # municipalities = {
            # "ALAMINOS", "BAY", "CABUYAO", "CALAUAN", "CAVINTI", "BINAN", "CALAMBA",
            # "SAN PEDRO", "SANTA ROSA", "FAMY", "KALAYAAN", "LILIW", "LOS BANOS", "LUISIANA",
            # "LUMBAN", "MABITAC", "MAGDALENA", "MAJAYJAY", "NAGCARLAN", "PAETE", "PAGSANJAN", "PAKIL", "PANGIL",
            # "PILA", "RIZAL", "SAN PABLO", "SANTA CRUZ", "SANTA MARIA", "SINILOAN", "VICTORIA"
        # }

        municipalities = {
           "VICTORIA"
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
                    if isinstance(geometry["coordinates"][0], list) and isinstance(geometry["coordinates"][0][0],
                                                                                   float):
                        geometry["coordinates"] = [geometry["coordinates"]]  # Wrap in a list

                # Convert to Shapely object and store if valid
                geom = shape(geometry)
                municipality_polygons[city].append(geom)

            except Exception as e:
                print(f"Error processing geometry for {city}: {e}")

        municipality_polygons = dict(sorted(municipality_polygons.items()))

        return municipality_polygons

    # def get_latest_sentinel_date(self, geometry):
        # search_iterator = self.catalog.search(
        #     DataCollection.SENTINEL2_L2A,
        #     geometry=geometry,
        #     time=("2023-01-01", datetime.utcnow().strftime("%Y-%m-%d")),  # Search up to today
        #     fields={"include": ["id", "properties.datetime"], "exclude": []}
        # )
    #     results = list(search_iterator)

    #     if results:
    #         # Extract available dates and sort them manually
    #         dates = [result["properties"]["datetime"][:10] for result in results]  # Extract YYYY-MM-DD
    #         latest_date = max(dates)  # Get the most recent date (assumed to be a string)
    #         latest_date = datetime.strptime(latest_date, "%Y-%m-%d")  # Convert to datetime
    #         latest_date -= timedelta(days=1)  # Now you can subtract
    #         latest_date = latest_date.strftime("%Y-%m-%d")  # Convert back to string if needed
    #         return latest_date
    #     else:
    #         return None

    def get_all_valid_sentinel_image(self, geometry):

        # Season2_2024 = [
        #     ('2024-03-16', '2024-03-20'), ('2024-03-21', '2024-03-25'), ('2024-03-26', '2024-03-30'),
        #     ('2024-03-31', '2024-04-04'), ('2024-04-05', '2024-04-09'), ('2024-04-10', '2024-04-14'),
        #     ('2024-04-15', '2024-04-19'), ('2024-04-20', '2024-04-24'), ('2024-04-25', '2024-04-29'),
        #     ('2024-04-30', '2024-05-04'), ('2024-05-05', '2024-05-09'), ('2024-05-10', '2024-05-14'),
        #     ('2024-05-15', '2024-05-19'), ('2024-05-20', '2024-05-24'), ('2024-05-25', '2024-05-29'),
        #     ('2024-05-30', '2024-06-03'), ('2024-06-04', '2024-06-08'), ('2024-06-09', '2024-06-13'),
        #     ('2024-06-14', '2024-06-18'), ('2024-06-19', '2024-06-23'), ('2024-06-24', '2024-06-28'),
        #     ('2024-06-29', '2024-07-03'), ('2024-07-04', '2024-07-08'), ('2024-07-09', '2024-07-13'),
        #     ('2024-07-14', '2024-07-18'), ('2024-07-19', '2024-07-23'), ('2024-07-24', '2024-07-28'),
        #     ('2024-07-29', '2024-08-02'), ('2024-08-03', '2024-08-07'), ('2024-08-08', '2024-08-12'),
        #     ('2024-08-13', '2024-08-17'), ('2024-08-18', '2024-08-22'), ('2024-08-23', '2024-08-27'),
        #     ('2024-08-28', '2024-09-01'), ('2024-09-02', '2024-09-06'), ('2024-09-07', '2024-09-11'),
        #     ('2024-09-12', '2024-09-15')
        # ]

        Season2_2024 = [
            ('2024-03-16', '2024-03-20'), ('2024-03-21', '2024-03-25'), ('2024-03-26', '2024-03-30'),
            ('2024-03-31', '2024-04-04'), ('2024-04-05', '2024-04-09'), ('2024-04-10', '2024-04-14'),
            ('2024-04-15', '2024-04-19'), ('2024-04-20', '2024-04-24'), ('2024-04-25', '2024-04-29'),
            ('2024-04-30', '2024-05-04'), ('2024-05-05', '2024-05-09'), ('2024-05-10', '2024-05-14'),
            ('2024-05-15')
        ]

        # # Check from latest to earliest for early exit
        # for start, end in reversed(Season2_2024):
        #     search_iterator = self.catalog.search(
        #         DataCollection.SENTINEL2_L2A,
        #         geometry=geometry,
        #         time=(start, end),
        #         fields={"include": ["id", "properties.datetime"], "exclude": []}
        #     )
        
        search_iterator = self.catalog.search(
            DataCollection.SENTINEL2_L2A,
            geometry=geometry,
            time=("2024-07-16", '2024-09-15'),  # Search up to today
            fields={"include": ["id", "properties.datetime"], "exclude": []}
        )
        results = list(search_iterator)
        print("results: ", results)
        if results:
            # Extract available dates and sort them manually
                dates = [result["properties"]["datetime"][:10] for result in results]  # Extract YYYY-MM-DD
                latest_date = max(dates)  # Get the most recent date (assumed to be a string)
                latest_date = datetime.strptime(latest_date, "%Y-%m-%d")  # Convert to datetime
                latest_date -= timedelta(days=1)  # Now you can subtract
                latest_date = latest_date.strftime("%Y-%m-%d")
                return latest_date

        return None

    # def get_all_valid_sentinel_images(self, geometry):
    #     """
    #     Retrieves all valid Sentinel-2 images from March 16 to September 15 (2024),
    #     labeled with their phase and season-end status.
        
    #     Returns:
    #         A list of dictionaries with keys: date, phase, is_season_end, id.
    #     """
    #     all_images = []

    #     # Define the season time range
    #     time_range = ("2024-03-16", "2024-09-15")

    #     # Perform the catalog search
    #     search_iterator = self.catalog.search(
    #         DataCollection.SENTINEL2_L2A,
    #         geometry=geometry,
    #         time=time_range,
    #         fields={"include": ["id", "properties.datetime"], "exclude": []},
    #         limit=100  # Adjust as needed
    #     )

    #     # Process results
    #     for count, result in enumerate(search_iterator, start=1):
    #         date_str = result["properties"]["datetime"][:10]

    #         try:
    #             date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    #             phase, is_season_end = get_phase(date_obj.day, date_obj.month)

    #             print(f"[{count}] Found image: {date_str} → Phase {phase}, End: {is_season_end}")

    #             if phase:
    #                 all_images.append({
    #                     "date": date_str,
    #                     "phase": phase,
    #                     "is_season_end": is_season_end,
    #                     "id": result["id"]
    #                 })

    #         except Exception as e:
    #             print(f"⚠️ Skipping result {count} due to error: {e}")

    #     # Optionally sort by date
    #     all_images.sort(key=lambda x: x["date"])

    #     print(f"✅ Total valid images collected: {len(all_images)}")
    #     return all_images


    def apply_image_filters(self, image):
        """Enhance the image by applying brightness and sharpening filters."""
        brightness_filter = np.array([[0, 0, 0], [0, 4, 0], [0, 0, 0]])
        sharpness_filter = np.array([[0, -1, 0], [-1, 5.2, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, brightness_filter)
        return cv2.filter2D(image, -1, sharpness_filter)

    def compute_black_pixel_ratio(self, image):
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

    def create_image_request(self, aoi_geometry, time_interval):
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
            config=self.config
        )

    def get_DateOfImage(self, municipality, polygons):
        if not polygons:
            print(f"No polygons found for {municipality}")
            return

        aoi_geometry = Geometry(MultiPolygon(polygons), CRS.WGS84)
        latest_date = self.get_all_valid_sentinel_image(aoi_geometry)

        if not latest_date:
            print(f"No recent Sentinel-2 image found for {municipality}")
            return

        for attempt in range(60):
            start_date = (datetime.strptime(latest_date, "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")
            time_interval = (start_date, latest_date)
            print(f"📅 Fetching image for {municipality} from {start_date} to {latest_date}")
            request = self.create_image_request(aoi_geometry, time_interval)
            raw_images = request.get_data()

            if raw_images:
                image = self.apply_image_filters(raw_images[0])
                black_ratio = self.compute_black_pixel_ratio(image)
                print(f"Black pixel ratio: {black_ratio:.2%}")

                if black_ratio < 0.30:
                    for day in range(6):
                        check_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=day)).strftime(
                            "%Y-%m-%d")
                        print(f"🔍 Checking image for {municipality} on {check_date}")

                        request_specific = self.create_image_request(aoi_geometry, (check_date, check_date))
                        specific_image = request_specific.get_data()

                        if specific_image:
                            image = self.apply_image_filters(specific_image[0])
                            black_ratio_new = self.compute_black_pixel_ratio(image)
                            print(f"Black pixel ratio on {check_date}: {black_ratio_new:.2%}")

                            if abs(black_ratio_new - black_ratio) < 1e-3:
                                print(f"✅ Using image from {check_date} (Same black pixel ratio)")
                                self.selected_dates[municipality] = check_date
                                return self.selected_dates
                    print(f"⚠️ No valid image found within the range {start_date} to {latest_date}")
            latest_date = (datetime.strptime(latest_date, "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")

    # def get_DateOfImage(self, municipality, polygons):
    #     if not polygons:
    #         return

    #     aoi_geometry = Geometry(MultiPolygon(polygons), CRS.WGS84)
    #     latest_date = self.get_all_valid_sentinel_images(aoi_geometry)

    #     if not latest_date:
    #         print(f"⚠️ No image found for {municipality} in Season2_2024")
    #         return

    #     time_interval = (latest_date, latest_date)
    #     print(f"📅 Fetching image for {municipality} on {latest_date}")
    #     request = self.create_image_request(aoi_geometry, time_interval)
    #     raw_images = request.get_data()

    #     if raw_images:
    #         image = self.apply_image_filters(raw_images[0])
    #         black_ratio = self.compute_black_pixel_ratio(image)

    #         if black_ratio < 0.30:
    #             self.selected_dates[municipality] = latest_date
    #         else:
    #             print(f"⚠️ Image for {municipality} on {latest_date} is too cloudy (black pixel ratio: {black_ratio:.2%})")
        
    #     if municipality in self.selected_dates:
    #         print(self.selected_dates[municipality])

    # def get_DateOfImage(self, municipality, polygons):
    #     if not polygons:
    #         return

    #     aoi_geometry = Geometry(MultiPolygon(polygons), CRS.WGS84)

    #     # Get all valid Sentinel-2 images
    #     all_images = self.get_all_valid_sentinel_images(aoi_geometry)
    #     if not all_images:
    #         print(f"⚠️ No valid images found for {municipality} in Season2_2024")
    #         return

    #     print(f"📦 Total valid image dates to check for {municipality}: {len(all_images)}")

    #     best_images = {}  # phase → { "date": ..., "black_ratio": ... }

    #     for idx, entry in enumerate(all_images, start=1):
    #         date_str = entry["date"]
    #         phase = entry["phase"]
    #         time_interval = (date_str, date_str)

    #         print(f"🛰️ [{idx}] Checking image for {municipality} on {date_str} (Phase {phase})")
    #         request = self.create_image_request(aoi_geometry, time_interval)
    #         raw_images = request.get_data()

    #         if raw_images:
    #             image = self.apply_image_filters(raw_images[0])
    #             black_ratio = self.compute_black_pixel_ratio(image)

    #             if black_ratio < 0.30:
    #                 if phase not in best_images or black_ratio < best_images[phase]["black_ratio"]:
    #                     best_images[phase] = {
    #                         "date": date_str,
    #                         "black_ratio": black_ratio
    #                     }
    #                     print(f"✅ New best image for Phase {phase} on {date_str} (black ratio: {black_ratio:.2%})")
    #                 else:
    #                     print(f"➖ Skipping {date_str}: Better image already selected for Phase {phase}")
    #             else:
    #                 print(f"☁️ Rejected image on {date_str} (black ratio: {black_ratio:.2%})")
    #         else:
    #             print(f"🚫 No image data returned on {date_str}")

    #     if best_images:
    #         selected_dates = [info["date"] for info in best_images.values()]
    #         self.selected_dates[municipality] = selected_dates
    #         print(f"📌 Best image dates for {municipality}: {selected_dates}")
    #     else:
    #         print(f"⚠️ No acceptable images found for {municipality} after filtering.")




    def get_NDVI_Images(self, municipality_polygons):

        for city, date in self.selected_dates.items():
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
                config=self.config
            )

            ndvi_images_batch = request_ndvi_image.get_data()

            if not ndvi_images_batch:
                print(f"⚠️ No NDVI image available for {city} on {date}")
                continue

            ndvi_image = ndvi_images_batch[0]  # Extract the first image

            # Convert to NumPy array
            ndvi_image = np.array(ndvi_image)

            print(ndvi_image)
            # Store data in structured format
            self.ndvi_images.append({
                "City/Municipality": city,
                "Date": date,
                "Image_Array": ndvi_image  # Store the actual image array
            })

    # def get_NDVI_Images(self, municipality_polygons):
    #     for city, date_list in self.selected_dates.items():
    #         print(f"📍 Processing NDVI images for {city}")

    #         # Retrieve the correct polygon for the city
    #         polygons = municipality_polygons.get(city, [])
    #         if not polygons:
    #             print(f"⚠️ No polygons found for {city}, skipping NDVI processing.")
    #             continue

    #         geometry = Geometry(MultiPolygon(polygons), crs=CRS.WGS84)

    #         for date in date_list:
    #             print(f"🛰️ Fetching NDVI image for {city} on {date}")

    #             evalscript_ndvi = """
    #                         //VERSION=3
    #                         function setup() {
    #                             return {
    #                                 input: ["B04", "B08", "dataMask"],
    #                                 output: { bands: 4 }
    #                             };
    #                         }

    #                         function evaluatePixel(sample) {
    #                             let val = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
    #                             let imgVals = null;
    #                             if (val < -1.1) imgVals = [0, 0, 0];
    #                             else if (val < -0.2) imgVals = [0.75, 0.75, 0.75];
    #                             else if (val < -0.1) imgVals = [0.86, 0.86, 0.86];
    #                             else if (val < 0) imgVals = [1, 1, 0.88];
    #                             else if (val < 0.025) imgVals = [1, 1, 0.5];
    #                             else if (val < 0.05) imgVals = [0.93, 0.1, 0.71];
    #                             else if (val < 0.075) imgVals = [0.87, 0.85, 0.61];
    #                             else if (val < 0.1) imgVals = [0.8, 0.78, 0.51];
    #                             else if (val < 0.125) imgVals = [0.74, 0.72, 0.42];
    #                             else if (val < 0.15) imgVals = [0.69, 0.76, 0.38];
    #                             else if (val < 0.175) imgVals = [0.64, 0.8, 0.35];
    #                             else if (val < 0.2) imgVals = [0.57, 0.75, 0.32];
    #                             else if (val < 0.25) imgVals = [0.5, 0.7, 0.28];
    #                             else if (val < 0.3) imgVals = [0.44, 0.64, 0.25];
    #                             else if (val < 0.35) imgVals = [0.38, 0.59, 0.21];
    #                             else if (val < 0.4) imgVals = [0.31, 0.54, 0.18];
    #                             else if (val < 0.45) imgVals = [0.25, 0.49, 0.14];
    #                             else if (val < 0.5) imgVals = [0.19, 0.43, 0.11];
    #                             else if (val < 0.55) imgVals = [0.13, 0.38, 0.07];
    #                             else if (val < 0.6) imgVals = [0.06, 0.33, 0.04];
    #                             else imgVals = [0, 0.27, 0];

    #                             imgVals.push(sample.dataMask);
    #                             return imgVals;
    #                         }
    #                         """

    #             request_ndvi_image = SentinelHubRequest(
    #                 evalscript=evalscript_ndvi,
    #                 input_data=[SentinelHubRequest.input_data(
    #                     data_collection=DataCollection.SENTINEL2_L2A,
    #                     time_interval=(date, date),
    #                     other_args={"dataFilter": {"mosaickingOrder": "leastCC"}}
    #                 )],
    #                 responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
    #                 geometry=geometry,
    #                 size=(224, 224),
    #                 config=self.config
    #             )

    #             ndvi_images_batch = request_ndvi_image.get_data()

    #             if not ndvi_images_batch:
    #                 print(f"⚠️ No NDVI image available for {city} on {date}")
    #                 continue

    #             ndvi_image = np.array(ndvi_images_batch[0])

    #             # Append the image and metadata
    #             self.ndvi_images.append({
    #                 "City/Municipality": city,
    #                 "Date": date,
    #                 "Image_Array": ndvi_image
    #             })

    #             print(f"✅ NDVI image stored for {city} on {date}")


    def get_ndvi_images(self):
        return self.ndvi_images

    def get_selected_dates(self):
        return self.selected_dates
