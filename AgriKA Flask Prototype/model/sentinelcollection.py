import json
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
        municipalities = {
            "ALAMINOS", "BAY", "CABUYAO", "CALAUAN", "CAVINTI", "BINAN", "CALAMBA",
            "SAN PEDRO", "SANTA ROSA", "FAMY", "KALAYAAN", "LILIW", "LOS BANOS", "LUISIANA",
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

    def get_latest_sentinel_date(self, geometry):
        search_iterator = self.catalog.search(
            DataCollection.SENTINEL2_L2A,
            geometry=geometry,
            time=("2023-01-01", datetime.utcnow().strftime("%Y-%m-%d")),  # Search up to today
            fields={"include": ["id", "properties.datetime"], "exclude": []}
        )
        results = list(search_iterator)

        if results:
            # Extract available dates and sort them manually
            dates = [result["properties"]["datetime"][:10] for result in results]  # Extract YYYY-MM-DD
            latest_date = max(dates)  # Get the most recent date (assumed to be a string)
            latest_date = datetime.strptime(latest_date, "%Y-%m-%d")  # Convert to datetime
            latest_date -= timedelta(days=1)  # Now you can subtract
            latest_date = latest_date.strftime("%Y-%m-%d")  # Convert back to string if needed
            return latest_date
        else:
            return None

    def apply_image_filters(self, image):
        truecolor_image = image

        brightness_filter = np.array([[0, 0, 0], [0, 4, 0], [0, 0, 0]])
        truecolor_image = cv2.filter2D(truecolor_image, -1, brightness_filter)

        sharpness_filter = np.array([[0, -1, 0], [-1, 5.2, -1], [0, -1, 0]])
        truecolor_image = cv2.filter2D(truecolor_image, -1, sharpness_filter)

        brightness_factor = 1.5
        truecolor_image_float = truecolor_image.astype(np.float32) * brightness_factor
        truecolor_image_float = np.clip(truecolor_image_float, 0, 255)
        truecolor_bright = truecolor_image_float.astype(np.uint8)

        return truecolor_bright

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
            #print(f"No polygons found for {municipality}")
            return

        aoi_geometry = Geometry(MultiPolygon(polygons), CRS.WGS84)
        latest_date = self.get_latest_sentinel_date(aoi_geometry)

        if not latest_date:
            #print(f"No recent Sentinel-2 image found for {municipality}")
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
                # print(f"Black pixel ratio: {black_ratio:.2%}")

                if black_ratio < 0.30:
                    for day in range(6):
                        check_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=day)).strftime(
                            "%Y-%m-%d")
                        # print(f"🔍 Checking image for {municipality} on {check_date}")

                        request_specific = self.create_image_request(aoi_geometry, (check_date, check_date))
                        specific_image = request_specific.get_data()

                        if specific_image:
                            image = self.apply_image_filters(specific_image[0])
                            black_ratio_new = self.compute_black_pixel_ratio(image)
                            # print(f"Black pixel ratio on {check_date}: {black_ratio_new:.2%}")

                            if abs(black_ratio_new - black_ratio) < 1e-3:
                                # print(f"✅ Using image from {check_date} (Same black pixel ratio)")
                                self.selected_dates[municipality] = check_date
                                return self.selected_dates
                    # print(f"⚠️ No valid image found within the range {start_date} to {latest_date}")
            latest_date = (datetime.strptime(latest_date, "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")


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
            # Store data in structured format
            self.ndvi_images.append({
                "City/Municipality": city,
                "Date": date,
                "Image_Array": ndvi_image  # Store the actual image array
            })

    def get_ndvi_images(self):
        return self.ndvi_images

    def get_selected_dates(self):
        return self.selected_dates
