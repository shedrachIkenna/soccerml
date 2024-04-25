import psycopg2
import pandas as pd 
from config import Config

def db_connect():
        # Define your database connection parameters
    db_params = {
        "host": Config.DB_HOST,
        "database": Config.DB_NAME,
        "user": Config.DB_USER,
        "password": Config.DB_PASSWORD,
    }

    # Establish a connection to the PostgreSQL database
    conn = psycopg2.connect(**db_params)

    # Define your SQL query to fetch the required columns from the soccer_data table
    query = """
    SELECT date,venue,team, sh, sot, gf, ga, dist, xg, xga, fk, pk, pkatt, sca, gca, result, opponent
    FROM "PremierLeague";
    """

    # Fetch the data from the database into a DataFrame
    matches = pd.read_sql(query, conn)
    # Close the database connection
    conn.close()

    return matches 