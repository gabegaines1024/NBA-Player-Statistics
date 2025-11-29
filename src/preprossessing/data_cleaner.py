import pandas as pd

#remove duplicates from the data
def remove_duplicates(data):
    return data.drop_duplicates()

#handle missing values
def handle_missing_values(data):
    return data.fillna(data.mean())

#remove games with low minutes played
def filter_minimum_minutes(data, min_minutes: int):
    return data[data['MP'] >= min_minutes]

#ensure consistent column names
def standardize_column_names(data):
    return data.rename(columns={
        'GAME_ID': 'game_id',
        'GAME_DATE': 'game_date',
        'MATCHUP': 'matchup',
        'PTS': 'points',
        'MIN': 'minutes',
        'MP': 'minutes_played',
        'FG': 'field_goals',
        'FGA': 'field_goals_attempted',
        'FG%': 'field_goal_percentage',
    })

#remove outliers from the data
def remove_outliers(data):
    return data[data['points'] < 100]

#validate data types
def validate_data_types(data):
    return data.dtypes == 'int64' or data.dtypes == 'float64'