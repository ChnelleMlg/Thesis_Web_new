from model.db import get_db_connection, get_realtime_yield_data
import folium
import os
import json
import matplotlib.pyplot as plt
from io import BytesIO
import base64

municipality_coords = {
    "alaminos": {"lat": 14.0616, "lng": 121.2604, "zoom": 12},
    "bay": {"lat": 14.1320, "lng": 121.2569, "zoom": 12},
    "cabuyao": {"lat": 14.2471, "lng": 121.1367, "zoom": 12},
    "calauan": {"lat": 14.1384, "lng": 121.3198, "zoom": 12},
    "cavinti": {"lat": 14.2647, "lng": 121.5455, "zoom": 12},
    "biñan": {"lat": 14.3036, "lng": 121.0781, "zoom": 12},
    "calamba": {"lat": 14.2127, "lng": 121.1639, "zoom": 12},
    "santa Rosa": {"lat": 14.2843, "lng": 121.0889, "zoom": 12},
    "famy": {"lat": 14.4730, "lng": 121.4842, "zoom": 12},
    "kalayaan": {"lat": 14.3313, "lng": 121.5484, "zoom": 12},
    "liliw": {"lat": 14.1364, "lng": 121.4399, "zoom": 12},
    "los baños": {"lat": 14.1699, "lng": 121.2441, "zoom": 12},
    "luisiana": {"lat": 14.1908, "lng": 121.5256, "zoom": 12},
    "lumban": {"lat": 14.2956, "lng": 121.4962, "zoom": 12},
    "mabitac": {"lat": 14.4338, "lng": 121.4113, "zoom": 12},
    "magdalena": {"lat": 14.2041, "lng": 121.4342, "zoom": 12},
    "majayjay": {"lat": 14.1447, "lng": 121.4723, "zoom": 12},
    "nagcarlan": {"lat": 14.1490, "lng": 121.3885, "zoom": 12},
    "paete": {"lat": 14.3675, "lng": 121.5300, "zoom": 12},
    "pagsanjan": {"lat": 14.2624, "lng": 121.4570, "zoom": 12},
    "pakil": {"lat": 14.3800, "lng": 121.4765, "zoom": 12},
    "pangil": {"lat": 14.4074, "lng": 121.4856, "zoom": 12},
    "pila": {"lat": 14.2346, "lng": 121.3656, "zoom": 12},
    "rizal": {"lat": 14.0841, "lng": 121.4113, "zoom": 12},
    "san pablo": {"lat": 14.0642, "lng": 121.3233, "zoom": 12},
    "san pedro": {"lat": 14.3562, "lng": 121.0553, "zoom": 12},
    "santa cruz": {"lat": 14.2691, "lng": 121.4113, "zoom": 12},
    "santa maria": {"lat": 14.5129, "lng": 121.4342, "zoom": 12},
    "siniloan": {"lat": 14.4383, "lng": 121.4856, "zoom": 12},
    "victoria": {"lat": 14.2028, "lng": 121.3370, "zoom": 12},
}

def get_color(yield_value):
    if yield_value == "No data" or yield_value == 0:
        return "#808080"
    elif yield_value < 3:
        return "#d13237"
    elif 3 <= yield_value < 4:
        return "#ffc91f"
    elif 4 <= yield_value < 5:
        return "#69a436"
    else:
        return "#1b499f"
    
def get_color_realtime(yield_value):
    if yield_value == "No data" or yield_value == 0:
        return "#808080"
    elif yield_value < 0.75:
        return "#d13237"
    elif 0.75 <= yield_value < 1.5:
        return "#ffc91f"
    elif 1.5 <= yield_value < 2.25:
        return "#69a436"
    else:
        return "#1b499f"

def create_map():
    """
    Generates a Folium map with municipalities colored based on yield values.
    """
    m = folium.Map(
        location=[14.16667, 121.33333],
        zoom_start=10,
        tiles="CartoDB Positron",
        attr="© OpenStreetMap contributors, © CartoDB"
    )

    geojson_files = [f"data/{file}" for file in os.listdir("data") if file.endswith(".geojson")]

    municipalities, yields, yield_data = get_realtime_yield_data()
    yield_dict = {m.lower(): v for m, v in yield_data.items()}

    for file in geojson_files:
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                geojson_data = json.load(f)

                geojson_data["features"] = [
                    feature for feature in geojson_data["features"]
                    if feature["geometry"]["type"] in ["Polygon", "MultiPolygon"]
                ]

                for feature in geojson_data["features"]:
                    municipality_name = feature["properties"].get("name", "Unknown Municipality").strip().lower()

                    if municipality_name not in yield_dict:
                        print(f"❌ No yield data for: {municipality_name}")
                    else:
                        print(f"✅ Found: {municipality_name} -> {yield_dict[municipality_name]}")

                    yield_value = yield_dict.get(municipality_name, "No data")

                    tooltip_html = folium.Tooltip(
                        f"""
                        <div style="font-size: 14px; font-weight: bold;">
                            🌾 {municipality_name.title()}
                        </div>
                        <div style="font-size: 12px;">
                            <b>Yield:</b> {yield_value if isinstance(yield_value, (int, float)) else '<span style="color: red;">No data</span>'}
                        </div>
                        """,
                        sticky=True
                    )

                    folium.GeoJson(
                        feature,
                        name=municipality_name,
                        style_function=lambda feature, y=yield_value: {
                            "fillColor": get_color_realtime(y),
                            "color": "black",
                            "weight": 2,
                            "fillOpacity": 0.7,
                        },
                        tooltip=tooltip_html
                    ).add_to(m)

    # --- Embed a JavaScript Function to Highlight a Municipality ---
    # Note: Folium creates a map variable with a generated name.
    # You can get that name from m.get_name(). For example:
    map_var = m.get_name()  # This returns a string like "map_a8f68228628dd4abeda66c8b1b11129b"
    highlight_script = f"""
                        <script>
                        function highlightMunicipality(selected) {{
                        for (var i in {map_var}._layers) {{
                            var layer = {map_var}._layers[i];
                            if (layer.feature && layer.feature.properties && layer.feature.properties.name) {{
                                var munName = layer.feature.properties.name.trim().toLowerCase();
                                if (munName === selected.trim().toLowerCase()) {{
                                    layer.setStyle({{fillOpacity: 0.9, color: 'red', weight: 3}});
                                    if (layer.bringToFront) {{
                                        layer.bringToFront();
                                    }}
                                }} else {{
                                    layer.setStyle({{fillOpacity: 0.7, color: 'black', weight: 2}});
                                }}
                            }}
                        }}
                        }}
                        </script>
                        """
    m.get_root().html.add_child(folium.Element(highlight_script))
    # --- End Embed Script ---

    if not os.path.exists("static"):
        os.makedirs("static")
    m.save("static/map.html")

def create_historical_map(year, season):
    """
    Generates a Folium map with municipalities colored based on historical yield values.
    """
    m = folium.Map(
        location=[14.16667, 121.33333],
        zoom_start=10,
        tiles="CartoDB Positron",
        attr="© OpenStreetMap contributors, © CartoDB"
    )

    geojson_files = [f"data/{file}" for file in os.listdir("data") if file.endswith(".geojson")]

    # Fetch historical data
    historical_data = get_historical_data(year, season)
    yield_dict = {entry["municipality"].strip().lower(): entry["yield"] for entry in historical_data}

    for file in geojson_files:
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                geojson_data = json.load(f)

                geojson_data["features"] = [
                    feature for feature in geojson_data["features"]
                    if feature["geometry"]["type"] in ["Polygon", "MultiPolygon"]
                ]

                for feature in geojson_data["features"]:
                    municipality_name = feature["properties"].get("name", "Unknown Municipality").strip().lower()

                    if municipality_name not in yield_dict:
                        print(f"❌ No historical yield data for: {municipality_name}")
                    else:
                        print(f"✅ Found: {municipality_name} -> {yield_dict[municipality_name]}")

                    yield_value = yield_dict.get(municipality_name, "No data")

                    tooltip_html = folium.Tooltip(
                        f"""
                        <div style="font-size: 14px; font-weight: bold;">
                            🌾 {municipality_name.title()}
                        </div>
                        <div style="font-size: 12px;">
                            <b>Year:</b> {year}<br>
                            <b>Season:</b> {season}<br>
                            <b>Yield:</b> {yield_value if isinstance(yield_value, (int, float)) else '<span style="color: red;">No data</span>'}
                        </div>
                        """,
                        sticky=True
                    )

                    folium.GeoJson(
                        feature,
                        name=municipality_name,
                        style_function=lambda feature, y=yield_value: {
                            "fillColor": get_color(y),
                            "color": "black",
                            "weight": 2,
                            "fillOpacity": 0.7,
                        },
                        tooltip=tooltip_html
                    ).add_to(m)

    if not os.path.exists("static"):
        os.makedirs("static")
    m.save("static/historical_map.html")

def create_all_historical_maps():
    """
    Generates a Folium map for each season in each year based on historical yield data.
    """
    # Get all unique year-season combinations
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT DISTINCT year, season FROM rice_field")
    year_season_combinations = cursor.fetchall()
    cursor.close()
    conn.close()

    for entry in year_season_combinations:
        year, season = entry["year"], entry["season"]
        print(f"Generating map for {year} - {season}")
        create_historical_map(year, season)
        
        # Save each map with a unique filename
        filename = f"static/historical_map_{year}_{season}.html"
        # ✅ Delete existing file if it already exists
        if os.path.exists(filename):
            os.remove(filename)  

        os.rename("static/historical_map.html", filename)  # Rename safely

def create_maps_per_municipality():
    """
    Generates a separate Folium map for each municipality with a uniform green color.
    """
    if not os.path.exists("static"):
        os.makedirs("static")

    geojson_files = [f"data/{file}" for file in os.listdir("data") if file.endswith(".geojson")]

    for file in geojson_files:
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                geojson_data = json.load(f)

                for municipality_name, coords in municipality_coords.items():
                    municipality_name = municipality_name.lower()  # Ensure matching case
                    
                    # Filter GeoJSON to only include this municipality
                    filtered_geojson = {
                        "type": "FeatureCollection",
                        "features": [
                            feature for feature in geojson_data["features"]
                            if feature["properties"].get("name", "").strip().lower() == municipality_name
                        ],
                    }

                    if not filtered_geojson["features"]:
                        print(f"⚠️ No GeoJSON data found for {municipality_name}")
                        continue  # Skip if no features match

                    # Create a map centered on the municipality
                    m = folium.Map(
                        location=[coords["lat"], coords["lng"]],
                        zoom_start=coords["zoom"],
                        tiles="CartoDB Positron"
                    )

                    # Add filtered municipality boundary with green color
                    folium.GeoJson(
                        filtered_geojson,
                        name=municipality_name,
                        style_function=lambda feature: {
                            "fillColor": "green",
                            "color": "black",
                            "weight": 2,
                            "fillOpacity": 0.7,
                        },
                    ).add_to(m)

                    # Save the map
                    filename = f"static/map_{municipality_name.replace(' ', '_')}.html"
                    m.save(filename)
                    print(f"✅ Map saved: {filename}")

def generate_yield_chart(municipalities, yields):
    """
    Generates a bar chart for yield data and encodes it as a base64 image.
    """
    plt.figure(figsize=(10, 5))
    plt.bar(municipalities, yields, color="#1b499f")
    plt.xlabel("Municipalities")
    plt.ylabel("Yield (tons per hectare)")
    plt.xticks(rotation=90)
    plt.title("Crop Yield Per Municipality")
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return chart_url


# Fetch historical data from MySQL
def get_historical_data(year=None, season=None):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Base query with JOIN
    query = """
        SELECT rf.municipality, rf.year, rf.season, h.yield
        FROM historical h
        JOIN rice_field rf ON h.ID_rice = rf.ID_rice
    """
    params = []

    # Add filtering if parameters are provided
    if year is not None and season is not None:
        query += " WHERE rf.year = %s AND rf.season = %s"
        params.extend([year, season])

    cursor.execute(query, params)
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return data