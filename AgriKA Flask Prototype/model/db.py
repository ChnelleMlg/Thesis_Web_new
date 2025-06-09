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




def get_phase(day, month):
    """
    Determines the phase based on the given day and month,
    and whether the date is the end of the season.

    First cycle (September 16 - March 15):
        Phase 1: September 16 - November 15
        Phase 2: November 16 - January 15
        Phase 3: January 16 - March 15 

    Second cycle (March 16 - September 15):
        Phase 1: March 16 - May 15
        Phase 2: May 16 - July 15
        Phase 3: July 16 - September 15 

    Returns:
        (phase: int, is_season_end: bool)
    """
    print(f"📅 DEBUG: Evaluating phase for {month=}, {day=}")

    # Second cycle: March 16 – September 15
    if (month == 3 and day >= 16) or (4 <= month <= 9 and (month != 9 or day <= 15)):
        if (month == 3 and day >= 16) or month == 4 or (month == 5 and day <= 15):
            return 1, False
        elif (month == 5 and day >= 16) or month == 6 or (month == 7 and day <= 15):
            return 2, False
        elif (month == 7 and day >= 16) or month == 8 or (month == 9 and day <= 15):
            return 3, True

    else:
        # First cycle: September 16 – March 15
        if (month == 9 and day >= 16) or month == 10 or (month == 11 and day <= 15):
            return 1, False
        elif (month == 11 and day >= 16) or month == 12 or (month == 1 and day <= 15):
            return 2, False
        elif (month == 1 and day >= 16) or month == 2 or (month == 3 and day <= 15):
            return 3, True

    return None, False  # Fallback for invalid inputs



def store_prediction_result(result):
    print("📥 Incoming Result:", result)

    city = result.get('City')
    day_of_year = int(result.get('Day'))  # Assuming this is 1–365
    predicted_yield = float(result.get('Predicted Yield'))
    current_year = 2024

    # Convert day-of-year to full date
    correct_date = datetime(current_year, 1, 1) + pd.Timedelta(days=day_of_year - 1)
    date_str = correct_date.strftime("%Y-%m-%d")

    # Extract day and month
    day = correct_date.day
    month = correct_date.month

    phase, is_season_end = get_phase(day, month)

    # Determine season
    if (month == 3 and day >= 16) or (4 <= month <= 8) or (month == 9 and day <= 15):
        season = 2
    else:
        season = 1

    print(f"🧾 Parsed Values → City: {city}, Date: {date_str}, Phase: {phase}, Season: {season}, Yield: {predicted_yield}")

    conn = get_db_connection()

    try:
        city = city.strip().upper()

        with conn.cursor() as cursor:
            # Ensure rice_field exists
            cursor.execute("""
                INSERT INTO rice_field (municipality, year, season)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE ID_rice = ID_rice
            """, (city, current_year, season))
            conn.commit()

            # Get ID_rice
            cursor.execute("""
                SELECT ID_rice FROM rice_field 
                WHERE municipality = %s AND year = %s AND season = %s
            """, (city, current_year, season))
            result_rows = cursor.fetchall()

            if not result_rows:
                print(f"⚠️ Error: Could not find or create rice_field for '{city}', {current_year}, Season {season}")
                return

            id_rice = result_rows[0][0]

            # Check if entry for this date already exists
            cursor.execute("""
                SELECT COUNT(*) FROM real_time 
                WHERE ID_rice = %s AND date = %s
            """, (id_rice, date_str))
            count = cursor.fetchone()[0]

            if count > 0:
                print(f"⛔ Skipping: Data for '{city}' on {date_str} already exists.")
                return

            # Insert real_time entry
            cursor.execute("""
                INSERT INTO real_time (ID_rice, date, phase, season, yield)
                VALUES (%s, %s, %s, %s, %s)
            """, (id_rice, date_str, phase, season, predicted_yield))
            conn.commit()
            print("✅ Prediction result inserted into real_time.")

            # Check if all three phases for this ID_rice and season are now available
            cursor.execute("""
                SELECT COUNT(DISTINCT phase)
                FROM real_time
                WHERE ID_rice = %s AND season = %s AND phase IN (1, 2, 3)
            """, (id_rice, season))
            phase_count = cursor.fetchone()[0]

            if phase_count == 3:
                # Check if historical entry already exists
                cursor.execute("""
                    SELECT COUNT(*) FROM historical
                    WHERE ID_rice = %s
                """, (id_rice,))
                hist_count = cursor.fetchone()[0]

                if hist_count == 0:
                    cursor.execute("""
                        SELECT phase, AVG(yield)
                        FROM real_time
                        WHERE ID_rice = %s AND season = %s AND phase IN (1, 2, 3)
                        GROUP BY phase
                        ORDER BY phase
                    """, (id_rice, season))
                    phase_averages = cursor.fetchall()

                    if len(phase_averages) == 3:
                        seasonal_avg = sum(row[1] for row in phase_averages) / 3.0
                        cursor.execute("""
                            INSERT INTO historical (ID_rice, yield)
                            VALUES (%s, %s)
                        """, (id_rice, seasonal_avg))
                        conn.commit()
                        print(f"📦 Seasonal average yield inserted into historical: {seasonal_avg:.2f}")
                else:
                    print("ℹ️ Seasonal average already exists in historical. Skipping insert.")

    except Exception as e:
        conn.rollback()
        print("❌ Error inserting prediction result:", e)

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