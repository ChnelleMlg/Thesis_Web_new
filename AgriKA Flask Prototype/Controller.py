# FINAL IMPORTS
import sys
from flask import Flask, render_template, jsonify, request, session
from collections import defaultdict
import random
from flask_apscheduler import APScheduler
from model.webloader import *
from model.sentinelcollection import *
from model.variablescollection import *
from model.model_loader import *
from model.db import *
#from services.yield_analytics import *

sys.path.insert(0, r"C:\Users\ASUS\OneDrive\Documents\GitHub\Thesis_Web_new\AgriKA Flask Prototype")

app = Flask(__name__)
scheduler = APScheduler()
app.secret_key = 'your_secret_key'

@scheduler.task('interval', id='sentinel_get', minutes=10)
def sentinel_get():

    print("\n\n\nSENTINEL WORKING\n\n\n")
    filepath = r"C:\Users\ASUS\OneDrive\Documents\GitHub\Thesis_Web_new\AgriKA Flask Prototype\static\fields_coordinates.geojson"
    #filepath = os.path.join(os.getcwd(), "static", "fields_coordinates.geojson")
    
    # Latest Sentinel acc ni MeChuchi 
    config = SHConfig()
    config.instance_id = 'ad3c8cf4-14a0-4dce-87fe-8c395e005cb5'
    config.sh_client_id = 'df5f87e9-0650-4e56-b680-15f78d0a0b4a'
    config.sh_client_secret = '68U2DFtPR3t1BNLj8McFvFzkikFsWbTT'

    ndvi_retriever = SentinelImageGet(filepath, config)

    municipality_polygons = ndvi_retriever.load_and_process_geojson()

    for city, polygons in municipality_polygons.items():
        ndvi_retriever.get_DateOfImage(city, polygons)

    ndvi_retriever.get_NDVI_Images(municipality_polygons)

    ndvi_images = ndvi_retriever.get_ndvi_images()
    selected_dates = ndvi_retriever.get_selected_dates()

    print(ndvi_images, selected_dates)

    datafor_database = variableCollector(ndvi_images, selected_dates)
    datafor_database.extract_features()
    merged_data = datafor_database.get_merged_data()
    

    cnn_model_instance = ModelLoader(merged_data)

    # Fit the scaler once on the merged dataset
    cnn_model_instance.fit_scaler()

#changes ni perl
@app.route('/handle_click', methods=['POST'])
def handle_click():
    data = request.get_json()  # Get the incoming JSON data
    municipality_clicked = data['municipality'].upper() #CHANGE 04-27
    year = data['year']
    season = data['season']
    yield_value = data['yield']

    session['municipality_clicked'] = municipality_clicked
    
    # Do something with the data (e.g., store it in a database, log it, etc.)
    print(f"Received data: {municipality_clicked}, {year}, {season}, {yield_value}")
    
    
    return jsonify({'status': 'success', 'message': 'Data received successfully'})
    #return redirect(url_for('view'))

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
    create_map()
    create_all_historical_maps()
    create_maps_per_municipality()
    return render_template('HomePage.html')

# Zoe's Dashboard Update
@app.route("/dashboard")
def dashboard():
    try:
        municipalities, real_time_yields, real_time_yield_data = get_realtime_yield_data()  #Unpacking three values ✅ 
    except Exception as e:
        print("❌ Error in get_realtime_yield_data:", e)
        municipalities, yields, yield_data = [], [], {}
    
    historical_yield_data = get_historical_data()
    municipality_clicked = session.get('municipality_clicked', 'Not clicked')  # Default to 'Not clicked' if not found

    #yearly_trends = process_yearly_trends(historical_yield_data)
    #municipality_averages = process_municipality_averages(historical_yield_data)
    #seasonal_data = process_seasonal_yield(historical_yield_data)

    return render_template(
        "dashboard.html",
        # Pass the real-time yield data to the template
        municipalities=municipalities,
        real_time_yields=real_time_yields,
        real_time_yield_data=real_time_yield_data,

        # Pass the historical yield data to the template
        historical_yield_data=historical_yield_data,
        municipality_clicked=municipality_clicked
        #yearly_trends=yearly_trends,
        #municipality_averages=municipality_averages,
        #seasonal_data=seasonal_data,
    )

@app.route('/view')
def view():

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
    
    # Get the clicked municipality data from the session
    municipality_clicked = session.get('municipality_clicked', 'Not clicked')  # Default to 'Not clicked' if not found

    # Pass the data to the template
    return render_template(
        'View.html', 
        municipalities=municipalities, 
        yields=yields, 
        yield_data=yield_data,  # Pass yield_data to template
        yield_chart=yield_chart,  
        table_container_id="yield-table-container",
        historical_yield_data=historical_yield_data,
        municipality_clicked=municipality_clicked,  # Add clicked data to template
    )

def get_color_for_muni(muni):
        random.seed(muni)  
        r = random.randint(50, 200)
        g = random.randint(50, 200)
        b = random.randint(50, 200)
        solid = f'rgba({r}, {g}, {b}, 1)'
        faded = f'rgba({r}, {g}, {b}, 0.5)'  
        return solid, faded


@app.route("/multi_year")
def multi_year():
    season = request.args.get("season")
    municipalities = request.args.getlist("municipality")

    # Convert season to int or None
    try:
        season = int(season) if season else None
    except ValueError:
        season = None

    all_munis = get_all_municipalities()

    if not municipalities:
        municipalities = all_munis

    # Always fetch all season data to allow conditional processing
    historical_data = get_multi_year(season=None, municipalities=municipalities)

    # chart_data[municipality][season][year] = yield
    chart_data = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    years_set = set()

    for row in historical_data:
        municipality = row['municipality']
        year = row['year']
        season_val = row['season']
        yield_value = row['yield']

        chart_data[municipality][season_val][year] += yield_value
        years_set.add(year)

    sorted_years = sorted(years_set)
    formatted_chart_data = []

    for municipality, seasons in chart_data.items():
        color_solid, color_faded = get_color_for_muni(municipality)

        if season:  # Single season selected
            year_data = seasons.get(season, {})
            dataset = {
                'label': f"{municipality} - Season {season}",
                'data': [year_data.get(year, 0) for year in sorted_years],
                'fill': False,
                'borderColor': color_solid if season == 1 else color_faded,
                'backgroundColor': color_solid if season == 1 else color_faded,  # <- Add this
                'borderDash': [] if season == 1 else [5, 5],
                'tension': 0.3,
                'borderWidth': 2
            }
            formatted_chart_data.append(dataset)

        else:  # All seasons
            for season_val in [1, 2]:
                year_data = seasons.get(season_val, {})
                dataset = {
                    'label': f"{municipality} - Season {season_val}",
                    'data': [year_data.get(year, 0) for year in sorted_years],
                    'fill': False,
                    'borderColor': color_solid if season_val == 1 else color_faded,
                    'backgroundColor': color_solid if season_val == 1 else color_faded,  # <- Add this
                    'borderDash': [] if season_val == 1 else [5, 5],
                    'tension': 0.3,
                    'borderWidth': 2
                }
                formatted_chart_data.append(dataset)

    return render_template(
        "multi_year.html",
        years=sorted_years,
        chart_datasets=formatted_chart_data,
        selected_season=str(season) if season else '',
        municipalities=all_munis,
        selected_municipalities=municipalities
    )


if __name__ == '__main__':
    scheduler.init_app(app)
    scheduler.start()
    app.run(debug=True)