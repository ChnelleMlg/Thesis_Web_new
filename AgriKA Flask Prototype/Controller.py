from flask import Flask, render_template
from model.db import get_db_connection 
from datetime import datetime
import folium
import os
import json
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

def get_realtime_yield_data():
    municipalities = [
        "Alaminos", "Bay", "Cabuyao", "Calauan", "Cavinti", "Biñan", "Calamba",
        "San Pedro", "Santa Rosa", "Famy", "Kalayaan", "Liliw", "Los Baños", "Luisiana",
        "Lumban", "Mabitac", "Magdalena", "Majayjay", "Nagcarlan", "Paete", "Pagsanjan", "Pakil", "Pangil",
        "Pila", "Rizal", "San Pablo", "Santa Cruz", "Santa Maria", "Siniloan", "Victoria"
    ]

    yields = [
        5.838, 4.694, 5.681, 4.716, 4.420, 5.603, 5.033, 0, 5.467, 4.452,
        3.959, 5.128, 5.313, 4.394, 4.043, 4.351, 5.412, 4.100, 4.588, 4.041,
        4.050, 4.485, 3.940, 4.417, 3.949, 4.614, 4.757, 4.678, 3.963, 4.583
    ]

    yield_data = [(m.strip().lower(), y if y > 0 else "No data") for m, y in zip(municipalities, yields)]
    
    return municipalities, yields, yield_data

def get_color(yield_value):

    if yield_value == "No data":
        return "#808080" 
    elif yield_value < 3:
        return "#d13237"  
    elif 3 <= yield_value < 4:
        return "#ffc91f"  
    elif 4 <= yield_value < 5:
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
    yield_dict = {m: v for m, v in yield_data}

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
                            "fillColor": get_color(y),
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

    if not os.path.exists("templates"):
        os.makedirs("templates")
    m.save("templates/map.html")


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


#### REAL TIME PART #####
def get_phase(day, month):
    """
    Determine phase based on day and month.
    
    For the first cycle (dates not between March 16 and September 15):
        - Phase 1: September 16 to October 31
        - Phase 2: November, December, and January
        - Phase 3: February to March 15
        
    For the second cycle (dates between March 16 and September 15):
        - Phase 1: March 16 to April 30
        - Phase 2: May, June, and July
        - Phase 3: August to September 15
    """
    # First determine which cycle the date falls into.
    # We'll assume that if the date is between March 16 and September 15, it's in the second cycle.
    if (month > 3 or (month == 3 and day >= 16)) and (month < 9 or (month == 9 and day <= 15)):
        # Second cycle
        if (month == 3 and day >= 16) or (month == 4):
            return 1
        elif month in [5, 6, 7]:
            return 2
        elif month == 8 or (month == 9 and day <= 15):
            return 3
    else:
        # First cycle: either the date is before March 16 or after September 15.
        # Note: This cycle spans from September 16 through March 15.
        if (month == 9 and day >= 16) or (month == 10):
            return 1
        elif month in [11, 12, 1]:
            return 2
        elif month == 2 or (month == 3 and day <= 15):
            return 3

    # Fallback (shouldn't happen if date is valid)
    return None


def store_prediction_result(result):
    city = result.get('City')
    day = int(result.get('Day'))
    month = int(result.get('Month'))
    predicted_yield = float(result.get('Predicted Yield'))  # ✅ Convert numpy.float32 to Python float
    phase = get_phase(day, month)
    current_year = datetime.now().year
    date_str = f"{current_year}-{month:02d}-{day:02d}"

    conn = get_db_connection()
    
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT ID_rice FROM rice_field WHERE municipality = %s", (city,))
            result_rows = cursor.fetchall()  # ✅ Fetch all results
            if not result_rows:
                print(f"Warning: No rice_field record found for '{city}'. Skipping insertion.")
                return
            id_rice = result_rows[0][0]

            insert_query = """
                INSERT INTO real_time (ID_rice, date, phase, yield)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_query, (id_rice, date_str, phase, predicted_yield))
            conn.commit()
            print("Prediction result inserted successfully.")
    except Exception as e:
        conn.rollback()
        print("Error inserting prediction result:", e)
    finally:
        conn.close()



##### HISTORICAL #####
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


@app.route('/')
def home():
    return render_template('HomePage.html')


@app.route('/view')
def view():
    create_map()
    
    # Fetch real-time data
    municipalities, yields, yield_data = get_realtime_yield_data()
    yield_chart = generate_yield_chart(municipalities, yields)

    # Fetch historical data
    historical_yield_data = get_historical_data()
    print("🔹 Historical Yield Data:", historical_yield_data)  # Debugging output

    return render_template(
        'View.html', 
        municipalities=municipalities, 
        yields=yields, 
        yield_data=yield_data, 
        yield_chart=yield_chart,  
        table_container_id="yield-table-container",
        historical_yield_data=historical_yield_data
    )



if __name__ == '__main__':
    app.run(debug=True)
