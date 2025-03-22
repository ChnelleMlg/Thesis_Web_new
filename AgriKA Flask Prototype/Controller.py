# FINAL IMPORTS
import sys
from flask import Flask, render_template, jsonify
from flask_apscheduler import APScheduler
from model.webloader import *
from model.sentinelcollection import *
from model.variablescollection import *
from model.model_loader import *
from model.db import *

sys.path.insert(0, r"C:\Users\perli\Desktop\AgriKA Web\AgriKA\Thesis_Web_new\AgriKA Flask Prototype")

app = Flask(__name__)
scheduler = APScheduler()

@scheduler.task('interval', id='sentinel_get', minutes=20)
def sentinel_get():

    print("\n\n\nSENTINEL WORKING\n\n\n")
    filepath = r"C:\Users\perli\Desktop\AgriKA Web\AgriKA\Thesis_Web_new\AgriKA Flask Prototype\static\fields_coordinates.geojson"
    
    # Sentinel acc ni Robby
    config = SHConfig()
    config.instance_id = '6225b6fd-f1b1-4ca9-bc6f-0669c6addf11'
    config.sh_client_id = '1f87b8fd-5426-4cc6-b509-a22ac3026f0b'
    config.sh_client_secret = 'nEyOclApQZvF6WwZMDvNI53OayeD5IRb'

    ndvi_retriever = SentinelImageGet(filepath, config)

    municipality_polygons = ndvi_retriever.load_and_process_geojson()

    for city, polygons in municipality_polygons.items():
        ndvi_retriever.get_DateOfImage(city, polygons)

    ndvi_retriever.get_NDVI_Images(municipality_polygons)

    ndvi_images = ndvi_retriever.get_ndvi_images()
    selected_dates = ndvi_retriever.get_selected_dates()

    datafor_database = variableCollector(ndvi_images, selected_dates)
    datafor_database.extract_features()

    merged_data = datafor_database.get_merged_data() #ito gamitin for prediction

    cnn_model_instance = ModelLoader(merged_data)

    # Fit the scaler once on the merged dataset
    cnn_model_instance.fit_scaler()


@app.route('/get_real_time_data')
def get_real_time_data():
    """Fetch real-time yield data dynamically via AJAX."""
    try:
        municipalities, yields, yield_data = get_realtime_yield_data()  # ✅ Correctly unpacking 3 values

        print("✅ Real-time Data Fetched:", municipalities, yields, yield_data)  # Debugging Output

        response = jsonify({
            "municipalities": municipalities,
            "yields": yields,
            "yield_data": yield_data  # ✅ Include yield_data in JSON response
        })
        #print("🔹 JSON Response:", response.get_data(as_text=True))  # Debugging output
        return response

    except Exception as e:
        print(f"❌ Error fetching real-time data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return render_template('HomePage.html')


@app.route('/view')
def view():
    create_map()
    create_all_historical_maps()
    create_maps_per_municipality()

    try:
        municipalities, yields, yield_data = get_realtime_yield_data()  #Unpacking three values ✅ 
    except Exception as e:
        print("❌ Error in get_realtime_yield_data:", e)
        municipalities, yields, yield_data = [], [], {}

    try:
        yield_chart = generate_yield_chart(municipalities, yields) if municipalities and yields else None
    except Exception as e:
        print("❌ Error in generate_yield_chart:", e)
        yield_chart = None

    try:
        historical_yield_data = get_historical_data()
    except Exception as e:
        print("❌ Error in get_historical_data:", e)
        historical_yield_data = []
    
    return render_template(
        'View.html', 
        municipalities=municipalities, 
        yields=yields, 
        yield_data=yield_data,  # ✅ Pass yield_data to template
        yield_chart=yield_chart,  
        table_container_id="yield-table-container",
        historical_yield_data=historical_yield_data
    )

if __name__ == '__main__':
    scheduler.init_app(app)
    scheduler.start()
    app.run(debug=True)