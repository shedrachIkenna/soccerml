from flask import Flask, render_template
import joblib
import pandas as pd
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta
import numpy as np
import db_connect

class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {
    'LeicesterCity': 'Leicester City',
    'CrystalPalace': 'Crystal Palace',
    'NorwichCity': 'Norwich City',
    'AstonVilla': 'Aston Villa',
    'BrightonandHoveAlbion': 'Brighton',
    'ManchesterUnited': 'Manchester Utd',
    'Man Utd': 'Manchester Utd',
    'LeedsUnited': 'Leeds United',
    'ManchesterCity': 'Manchester City',
    'Man City': 'Manchester City',
    'NewcastleUnited': 'Newcastle Utd',
    'Newcastle': 'Newcastle Utd',
    'TottenhamHotspur': 'Tottenham',
    'Spurs': 'Tottenham',
    'WestHamUnited': 'West Ham', 
    'WolverhamptonWanderers': 'Wolves',
    'LutonTown': 'Luton Town',
    'Luton': 'Luton Town',
    'NottinghamForest': 'Nottingham Forest',
    "Nott'ham Forest": 'Nottingham Forest',
    "SheffieldUnited": 'Sheffield Utd',
    "LeedsUnited": "Leeds United",
    "NorwichCity": "Norwich City",
    "WestBromwichAlbion": "West Brom",
}



app = Flask(__name__)
model = joblib.load(r'C:\Users\DELL\Desktop\MLapp\app\model.pkl')
preprocessor = joblib.load(r'C:\Users\DELL\Desktop\MLapp\app\preprocessor.pkl')

@app.route('/')
def index():
    return render_template('predictions.html')  

@app.route('/predict')
def predict():
    processed_data = process_data()
    predicted_outcomes = model.predict(processed_data)
    predicted_probabilities = model.predict_proba(processed_data)

    # Map numerical predictions to text labels
    prediction_labels = {1: 'Home 2+ goals', 0: 'Home no 2+ goals'}
    predictions_text = [prediction_labels[pred] for pred in predicted_outcomes]
    
    # Format probabilities as percentages
    formatted_probabilities = [f"{prob*100:.0f}%" for prob in predicted_probabilities[:, 1]]

    # Construct DataFrame for predictions similar to the Jupyter notebook approach
    predictions_df = pd.DataFrame({
        #'Date': processed_data['date'],
        'Home': processed_data['home'],
        'Away': processed_data['away'],
        'Prediction': predictions_text,
        'Probability home 2+ goals': formatted_probabilities,  
        'Probability home no 2+ goals': [f"{(1-prob)*100:.0f}%" for prob in predicted_probabilities[:, 1]]   
    })
    
    # Calculate predicted result based on highest probability
    #predictions_df['Predicted Result'] = predictions_df[['Home Win', 'Draw', 'Away Win']].idxmax(axis=1)
    
    # Convert DataFrame to HTML or to a dictionary that can be easily passed to the template
    predictions_html = predictions_df.to_html(classes='table table-striped', index=False)
    
    # Pass the predictions as HTML table to the template
    # Alternatively, pass the DataFrame as a dictionary if you wish to format it in the template
    return render_template('predictions.html', predictions_table=predictions_html)

def get_matches():
    matches = db_connect()

    matches['date'] = pd.to_datetime(matches['date'], format='%Y-%m-%d')
    matches['year'] = matches['date'].dt.year
    mapping = MissingDict(**map_values)
    matches['team'] = matches['team'].map(mapping)
    matches['opponent'] = matches['opponent'].map(mapping)
    matches = matches.sort_values(by='date')
    df_1 = matches[(matches['venue'] == 'Home')]
    df_2 = matches[(matches['venue'] == 'Away')]
    df_1_select = df_1[['date', 'venue', 'gf', 'xg', 'xga', 'sh', 'sot', 'dist', 'sca', 'gca', 'team', 'year', 'opponent']]
    df_2_select = df_2[['date', 'venue', 'gf', 'xg', 'xga', 'sh', 'sot', 'dist', 'sca', 'gca', 'team', 'year', 'opponent']]
    # Specify the columns for merging
    merge_columns = ['date', 'year']

    # Merge the DataFrames on the specified columns and team-opponent condition
    merged_df = pd.merge(
        df_1_select,
        df_2_select,
        how='inner',
        left_on=merge_columns + ['team'],
        right_on=merge_columns + ['opponent'],
        suffixes=('_df1', '_df2')
    )

    merged_df = merged_df[['date', 'gf_df1', 'xg_df1', 'xga_df1',
                       'sh_df1', 'sot_df1', 'dist_df1', 'sca_df1', 'gca_df1', 'team_df1', 'opponent_df1',
                       'gf_df2', 'xg_df2', 'xga_df2', 'sh_df2', 'sot_df2', 'dist_df2', 'gca_df2', 'sca_df2', 'year'
                      ]]
    # Rename columns with '_df1' to '_home' and '_df2' to '_away'
    merged_df = merged_df.rename(columns=lambda col: col.replace('_df1', '_home').replace('_df2', '_away'))
    merged_df = merged_df.rename(columns={
    'team_home': 'home',
    'opponent_home': 'away'
    })

    # Create the 'result' column
    merged_df['result'] = np.where(merged_df['gf_home'] == merged_df['gf_away'], 'Draw', np.where(merged_df['gf_home'] > merged_df['gf_away'], 'Home', 'Away'))

    # Sort merged_df by the 'date' column
    merged_df = merged_df.sort_values(by='date')

    # Concatenate fixtures below merged_df
    merged_df['result_numeric'] = merged_df['result'].map({'Draw': 0, 'Home': 1, 'Away': -1})

    merged_df['home_points'] = 0  # Initialize the column with default value

    # Set home_points based on the conditions
    merged_df.loc[merged_df['gf_home'] > merged_df['gf_away'], 'home_points'] = 3
    merged_df.loc[merged_df['gf_home'] == merged_df['gf_away'], 'home_points'] = 1


    merged_df['away_points'] = 0  # Initialize the column with default value

    # Set home_points based on the conditions
    merged_df.loc[merged_df['gf_away'] > merged_df['gf_home'], 'away_points'] = 3
    merged_df.loc[merged_df['gf_away'] == merged_df['gf_home'], 'away_points'] = 1



    merged_df['home_win'] = (merged_df['gf_home'] > merged_df['gf_away']).astype(int)
    merged_df['away_win'] = (merged_df['gf_away'] > merged_df['gf_home']).astype(int)


    merged_df['home_lost'] = (merged_df['gf_home'] < merged_df['gf_away']).astype(int)
    merged_df['away_lost'] = (merged_df['gf_away'] < merged_df['gf_home']).astype(int)

    merged_df['total_goals'] = (merged_df['gf_home'] + merged_df['gf_away'])
    merged_df['over_one_and_half_goals'] = np.where(merged_df['total_goals'] > 1, 1, 0)
    merged_df['over_two_and_half_goals'] = np.where(merged_df['total_goals'] > 2, 1, 0)

    merged_df['less_than_four_goals'] = np.where(merged_df['total_goals'] < 4, 1, 0)
    merged_df['less_than_five_goals'] = np.where(merged_df['total_goals'] < 5, 1, 0)

    merged_df['home_clean_sheet'] = np.where(merged_df['gf_away'] == 0, 1, 0)
    merged_df['away_clean_sheet'] = np.where(merged_df['gf_home'] == 0, 1, 0)

    # 1. home_scored_two_goals_or_more
    merged_df['home_scored_two_goals_or_more'] = np.where(merged_df['gf_home'] > 1, 1, 0)

    # 2. away_scored_two_goals_or_more
    merged_df['away_scored_two_goals_or_more'] = np.where(merged_df['gf_away'] > 1, 1, 0)

    # 3. home_conceded_two_goals_or_more
    merged_df['home_conceded_two_goals_or_more'] = np.where(merged_df['gf_away'] > 1, 1, 0)

    # 4. away_conceded_two_goals_or_more
    merged_df['away_conceded_two_goals_or_more'] = np.where(merged_df['gf_home'] > 1, 1, 0)

    # Add 'draw_or_gg' column based on conditions
    merged_df['draw_or_gg'] = ((merged_df['result_numeric'] == 0) | ((merged_df['gf_home'] > 0) & (merged_df['gf_away'] > 0))).astype(int)
    merged_df['gg'] = (((merged_df['gf_home'] > 0) & (merged_df['gf_away'] > 0))).astype(int)


    # Add a 'draw' column to merged_df
    merged_df['draw'] = np.where(merged_df['result_numeric'] == 0, 1, 0)

    # Add 'draw_or_gg' column based on conditions
    merged_df['draw_or_gg'] = ((merged_df['result_numeric'] == 0) | ((merged_df['gf_home'] > 0) & (merged_df['gf_away'] > 0))).astype(int)
    merged_df['gg'] = (((merged_df['gf_home'] > 0) & (merged_df['gf_away'] > 0))).astype(int)

    fixtures = get_fixtures()

    data = pd.concat([merged_df, fixtures], ignore_index=True, sort=False)

    # Apply the function to calculate rolling averages for all teams
    data = calculate_rolling_averages_all_teams(data)

    return data


def calculate_rolling_averages_all_teams(data, rolling_window=7):
    # List of columns for which you want to calculate rolling averages
    columns_to_average_home = ['gf_home', 'sh_home', 'sot_home', 'dist_home', 'xg_home', 'xga_home', 'sca_home', 
                               'gca_home', 'home_points', 'home_win', 'home_lost', 'home_clean_sheet',
                               'home_scored_two_goals_or_more', 'home_conceded_two_goals_or_more']

    columns_to_average_away = ['gf_away', 'sh_away', 'sot_away', 'dist_away', 'xg_away', 'xga_away', 'sca_away', 
                               'gca_away', 'away_points', 'away_win', 'away_lost', 'away_clean_sheet',
                               'away_scored_two_goals_or_more', 'away_conceded_two_goals_or_more']

    for column in columns_to_average_home:
        # Calculate rolling averages for home team
        data[f'{column}_rolling_avg'] = (
            data.groupby('home')[column]
                .shift(1)
                .rolling(window=rolling_window, min_periods=rolling_window)
                .mean()
        )
    

    for column in columns_to_average_away:
        # Calculate rolling averages for home team
        data[f'{column}_rolling_avg'] = (
            data.groupby('away')[column]
                .shift(1)
                .rolling(window=rolling_window, min_periods=rolling_window)
                .mean()
        )

    # Calculate conversion rates
    data['home_conversion_rate'] = data['gf_home_rolling_avg'] / data['sh_home_rolling_avg']
    data['away_conversion_rate'] = data['gf_away_rolling_avg'] / data['sh_away_rolling_avg']

    data['sot_percent_home'] = (data['sot_home_rolling_avg'] / data['sh_home_rolling_avg']) * 100
    data['sot_percent_away'] = (data['sot_away_rolling_avg'] / data['sh_away_rolling_avg']) * 100

    data['g/sot_home'] = (data['sh_home_rolling_avg'] / data['sot_home_rolling_avg']) 
    data['g/sot_away'] = (data['sh_away_rolling_avg'] / data['sot_away_rolling_avg']) 


    return data



def get_fixtures(): 
    standing_url = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
    data = requests.get(standing_url)
    soup = BeautifulSoup(data.text)
    df = pd.read_html(data.text, match="Scores & Fixtures")[0]
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    # Calculate today's date and the date three days from now
    today = pd.to_datetime('today').normalize()  # normalize() sets the time to 00:00:00 for comparison
    two_days_later = today + timedelta(days=2)

    # Select rows where the date is between today and three days later
    fixtures = df[(df['date'] >= today) & (df['date'] <= two_days_later)]
    fixtures = fixtures[['date', 'home', 'away']]

    mapping = MissingDict(**map_values)
    fixtures['away'] = fixtures['away'].map(mapping)
    fixtures['home'] = fixtures['home'].map(mapping)

    return fixtures


def upcoming_fixtures():
    data = get_matches()
    # Calculate today's date and the date three days from now
    today = pd.to_datetime('today').normalize()  # normalize() sets the time to 00:00:00 for comparison
    three_days_later = today + timedelta(days=3)

    upcoming_fixtures = data[(data['date'] >= today) & (data['date'] <= three_days_later)]

    return upcoming_fixtures




def process_data():
    data = upcoming_fixtures()
    data = data[[ 'home', 'away',
    'gf_home_rolling_avg', 
    'sh_home_rolling_avg', 'sot_home_rolling_avg',
    'gf_away_rolling_avg', 'sh_away_rolling_avg', 
    'sot_away_rolling_avg', 
    'home_conversion_rate', 'away_conversion_rate', 
    'sot_percent_home','sot_percent_away', 'g/sot_home', 'g/sot_away',
    'home_scored_two_goals_or_more_rolling_avg',
    'home_conceded_two_goals_or_more_rolling_avg',
    'away_scored_two_goals_or_more_rolling_avg',
    'away_conceded_two_goals_or_more_rolling_avg',]]
    
    return data


if __name__ == "__main__":
    app.run(debug=True)



