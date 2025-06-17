# models/db.py
import mysql.connector
from config import DB_CONFIG
from datetime import datetime
import pandas as pd

def get_db_connection():
    """Create and return a new database connection."""
    conn = mysql.connector.connect(
        host=DB_CONFIG['host'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        database=DB_CONFIG['database']
    )
    return conn

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

#### REAL TIME PART #####
def get_phase(day, month):
    """
    Determines the phase based on the given day and month.

    First cycle (September 16 - March 15):
        
    Phase 1: September 16 - November 15
    Phase 2: November 16 - January 15
    Phase 3: January 16 - March 15

        Second cycle (March 16 - September 15):
            
    Phase 1: March 16 - May 15
    Phase 2: May 16 - July 15
    Phase 3: July 16 - September 15
    """

    # Determine if the date falls into the second cycle (March 16 – September 15)
    if (month == 3 and day >= 16) or (4 <= month <= 9 and (month != 9 or day <= 15)):
        # Second Cycle
        if (month == 3 and day >= 16) or month == 4 or (month == 5 and day <= 15):
            return 1  # Phase 1
        elif (month == 5 and day >= 16) or month == 6 or (month == 7 and day <= 15):
            return 2  # Phase 2
        elif (month == 7 and day >= 16) or month == 8 or (month == 9 and day <= 15):
            return 3  # Phase 3

    else:
        # First Cycle (September 16 – March 15)
        if (month == 9 and day >= 16) or month == 10 or (month == 11 and day <= 15):
            return 1  # Phase 1
        elif (month == 11 and day >= 16) or month == 12 or (month == 1 and day <= 15):
            return 2  # Phase 2
        elif (month == 1 and day >= 16) or month == 2 or (month == 3 and day <= 15):
            return 3  # Phase 3

    return None  # Fallback for invalid inputs

def store_prediction_result(result):
    city = result.get('City')
    day = int(result.get('Day'))
    month = int(result.get('Month'))
    predicted_yield = float(result.get('Predicted Yield'))  # Convert numpy.float32 to Python float
    phase = get_phase(day, month)

    # 🔄 Determine Season: 1 for first cycle, 2 for second cycle
    season = 1 if (month >= 9 or month <= 3) else 2

    current_year = datetime.now().year

    # Convert DOY to a proper date
    correct_date = datetime(current_year, 1, 1) + pd.Timedelta(days=day - 1)
    correct_day = correct_date.day

    date_str = f"{current_year}-{month:02d}-{correct_day:02d}"

    conn = get_db_connection()

    try:
        with conn.cursor() as cursor:
            # Get ID_rice for the given city
            cursor.execute("SELECT ID_rice FROM rice_field WHERE municipality = %s", (city,))
            result_rows = cursor.fetchall()
            if not result_rows:
                print(f"Warning: No rice_field record found for '{city}'. Skipping insertion.")
                return
            id_rice = result_rows[0][0]

            # 🔍 Check if record already exists
            check_query = """
                SELECT COUNT(*) FROM real_time 
                WHERE ID_rice = %s AND date = %s
            """
            cursor.execute(check_query, (id_rice, date_str))
            count = cursor.fetchone()[0]

            if count > 0:
                print(f"Skipping insertion: Data for '{city}' on {date_str} already exists.")
                return

            # ✅ Insert only if no duplicate exists
            insert_query = """
                INSERT INTO real_time (ID_rice, date, phase, season, yield)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (id_rice, date_str, phase, season, predicted_yield))
            conn.commit()
            print("Prediction result inserted successfully.")

    except Exception as e:
        conn.rollback()
        print("Error inserting prediction result:", e)

    finally:
        conn.close()
        
# Fetching real time data from database to display
def get_realtime_yield_data():
    """Fetch real-time yield data (Municipality, Yield, and Yield Data Dictionary)."""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        query = """
        SELECT rf.municipality, rt.yield
        FROM real_time rt
        JOIN rice_field rf ON rt.ID_rice = rf.ID_rice
        WHERE rt.date = (SELECT MAX(date) FROM real_time WHERE ID_rice = rt.ID_rice)  # Get latest yield per municipality
        ORDER BY rf.municipality ASC
        """
        cursor.execute(query)
        results = cursor.fetchall()

        print("🔹 Raw Database Results:", results)  # Debugging log

        if not results:
            return [], [], {}  # Return empty if no data found

        municipalities = [row[0] for row in results]
        yields = [row[1] if row[1] is not None else "No data" for row in results]  # Handle None values
        yield_data = {row[0]: row[1] if row[1] is not None else "No data" for row in results}  

        print("✅ Parsed Data:", municipalities, yields, yield_data)  # Debugging log
        return municipalities, yields, yield_data  

    except Exception as e:
        print(f"❌ Error fetching real-time data: {e}")
        return [], [], {}  

    finally:
        cursor.close()
        conn.close()

def get_multi_year(season=None, municipalities=None):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT rf.municipality, rf.year, rf.season, h.yield
        FROM historical h
        JOIN rice_field rf ON h.ID_rice = rf.ID_rice
        WHERE 1=1
    """
    params = []

    if season is not None:
        query += " AND rf.season = %s"
        params.append(season)

    if municipalities:
        placeholders = ','.join(['%s'] * len(municipalities))
        query += f" AND rf.municipality IN ({placeholders})"
        params.extend(municipalities)

    query += " ORDER BY rf.municipality, rf.year"

    cursor.execute(query, params)
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return data

def get_all_municipalities():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT municipality FROM rice_field ORDER BY municipality")
    municipalities = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return municipalities

def get_latest_realtime_date():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)  # Adjust if needed
    
    query = """
        SELECT rt.date
        FROM real_time rt
        ORDER BY rt.date DESC
        LIMIT 1
    """
    
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return data

