import pandas as pd
import numpy as np
from typing import Optional, List


def create_rolling_features(df: pd.DataFrame, 
                           columns: List[str], 
                           windows: List[int] = [3, 5, 10],
                           player_col: str = 'PLAYER_NAME') -> pd.DataFrame:
    """
    Create rolling average features for specified columns.
    
    Args:
        df: DataFrame with game logs
        columns: List of column names to create rolling features for
        windows: List of window sizes for rolling averages
        player_col: Column name for player identifier
        
    Returns:
        DataFrame with rolling features added
    """
    df = df.copy()
    
    # Ensure data is sorted by date for rolling calculations
    if 'GAME_DATE' in df.columns:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values(['GAME_DATE', player_col])
    
    # Group by player to calculate rolling stats per player
    for col in columns:
        if col in df.columns:
            for window in windows:
                df[f'{col}_rolling_{window}'] = df.groupby(player_col)[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                )
    
    return df


def create_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create efficiency metrics like true shooting percentage, usage rate, etc.
    
    Args:
        df: DataFrame with game logs
        
    Returns:
        DataFrame with efficiency features added
    """
    df = df.copy()
    
    # True Shooting Percentage (TS%)
    if all(col in df.columns for col in ['PTS', 'FGA', 'FTA']):
        df['TS_PCT'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
        df['TS_PCT'] = df['TS_PCT'].replace([np.inf, -np.inf], np.nan)
    
    # Effective Field Goal Percentage (eFG%)
    if all(col in df.columns for col in ['FG', 'FG3M', 'FGA']):
        df['EFG_PCT'] = (df['FG'] + 0.5 * df['FG3M']) / df['FGA']
        df['EFG_PCT'] = df['EFG_PCT'].replace([np.inf, -np.inf], np.nan)
    
    # Points per shot attempt
    if all(col in df.columns for col in ['PTS', 'FGA']):
        df['PTS_PER_FGA'] = df['PTS'] / df['FGA']
        df['PTS_PER_FGA'] = df['PTS_PER_FGA'].replace([np.inf, -np.inf], np.nan)
    
    # Assist to turnover ratio
    if all(col in df.columns for col in ['AST', 'TOV']):
        df['AST_TO_TOV'] = df['AST'] / (df['TOV'] + 1)  # +1 to avoid division by zero
        df['AST_TO_TOV'] = df['AST_TO_TOV'].replace([np.inf, -np.inf], np.nan)
    
    # Rebound rate (total rebounds per game)
    if all(col in df.columns for col in ['OREB', 'DREB']):
        df['TREB'] = df['OREB'] + df['DREB']
    
    # Double-double indicator
    if all(col in df.columns for col in ['PTS', 'TREB', 'AST']):
        df['DOUBLE_DOUBLE'] = (
            ((df['PTS'] >= 10) & (df['TREB'] >= 10)) |
            ((df['PTS'] >= 10) & (df['AST'] >= 10)) |
            ((df['TREB'] >= 10) & (df['AST'] >= 10))
        ).astype(int)
    
    # Triple-double indicator
    if all(col in df.columns for col in ['PTS', 'TREB', 'AST']):
        df['TRIPLE_DOUBLE'] = (
            (df['PTS'] >= 10) & (df['TREB'] >= 10) & (df['AST'] >= 10)
        ).astype(int)
    
    return df


def create_game_streak_features(df: pd.DataFrame, 
                                player_col: str = 'PLAYER_NAME') -> pd.DataFrame:
    """
    Create streak features (consecutive games, win/loss streaks, etc.).
    
    Args:
        df: DataFrame with game logs
        player_col: Column name for player identifier
        
    Returns:
        DataFrame with streak features added
    """
    df = df.copy()
    
    # Ensure data is sorted by date
    if 'GAME_DATE' in df.columns:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values(['GAME_DATE', player_col])
    
    # Games played streak (consecutive games)
    df['GAMES_PLAYED_STREAK'] = df.groupby(player_col).cumcount() + 1
    
    # Win streak (if WL column exists)
    if 'WL' in df.columns:
        df['WIN_STREAK'] = df.groupby(player_col)['WL'].apply(
            lambda x: (x == 'W').groupby((x != x.shift()).cumsum()).cumsum()
        ).reset_index(0, drop=True)
    
    return df


def create_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features (day of week, month, days rest, etc.).
    
    Args:
        df: DataFrame with game logs
        
    Returns:
        DataFrame with time-based features added
    """
    df = df.copy()
    
    if 'GAME_DATE' in df.columns:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        
        # Day of week
        df['DAY_OF_WEEK'] = df['GAME_DATE'].dt.dayofweek
        
        # Month
        df['MONTH'] = df['GAME_DATE'].dt.month
        
        # Days since last game (rest days)
        df['DAYS_REST'] = df.groupby('PLAYER_NAME')['GAME_DATE'].diff().dt.days.fillna(0)
        
        # Back-to-back indicator
        df['BACK_TO_BACK'] = (df['DAYS_REST'] == 1).astype(int)
    
    return df


def create_matchup_features(df: pd.DataFrame, player_col: str = 'PLAYER_NAME') -> pd.DataFrame:
    """
    Create features related to matchup including opponent-specific performance history.
    
    This function creates:
    - Home/Away indicators
    - Opponent team extraction
    - Player's historical stats vs specific opponents (rolling averages)
    - Recent performance vs opponent (last 3, 5 games against them)
    
    Args:
        df: DataFrame with game logs
        player_col: Column name for player identifier
        
    Returns:
        DataFrame with matchup features added
    """
    df = df.copy()
    
    # Home/Away indicator
    if 'MATCHUP' in df.columns:
        df['IS_HOME'] = df['MATCHUP'].str.contains('vs.').astype(int)
        df['IS_AWAY'] = df['MATCHUP'].str.contains('@').astype(int)
        
        # Extract opponent team abbreviation from MATCHUP
        # Format is either "LAL vs. BOS" or "LAL @ BOS"
        df['OPPONENT'] = df['MATCHUP'].str.extract(r'(?:vs\.|@)\s*([A-Z]{2,3})')
    
    # Create opponent-specific rolling statistics
    if 'OPPONENT' in df.columns:
        # Sort by player and date to ensure chronological order
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values(['GAME_DATE', player_col])
        
        # Define key stats to track vs opponents
        stat_columns = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'FGM', 'FGA', 'FG3M', 'FG3A']
        available_stats = [col for col in stat_columns if col in df.columns]
        
        # For each stat, create rolling averages vs specific opponent
        for stat in available_stats:
            # Average vs this opponent (all previous games)
            df[f'{stat}_vs_OPP_avg'] = df.groupby([player_col, 'OPPONENT'])[stat].transform(
                lambda x: x.shift(1).expanding().mean()
            )
            
            # Last 3 games vs this opponent
            df[f'{stat}_vs_OPP_L3'] = df.groupby([player_col, 'OPPONENT'])[stat].transform(
                lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
            )
            
            # Last 5 games vs this opponent
            df[f'{stat}_vs_OPP_L5'] = df.groupby([player_col, 'OPPONENT'])[stat].transform(
                lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
            )
        
        # Count of games played vs this opponent (experience factor)
        df['GAMES_vs_OPP'] = df.groupby([player_col, 'OPPONENT']).cumcount()
        
        # Days since last played this opponent
        df['DAYS_SINCE_vs_OPP'] = df.groupby([player_col, 'OPPONENT'])['GAME_DATE'].transform(
            lambda x: (x - x.shift(1)).dt.days if 'GAME_DATE' in df.columns else np.nan
        )
    
    return df


def engineer_features(df: pd.DataFrame, 
                     target_column: str = 'PTS',
                     player_col: str = 'PLAYER_NAME') -> pd.DataFrame:
    """
    Main function to engineer all features for machine learning.
    
    Args:
        df: DataFrame with game logs
        target_column: Column to predict (e.g., 'PTS', 'AST', 'REB')
        player_col: Column name for player identifier
        
    Returns:
        DataFrame with all engineered features
    """
    df = df.copy()
    
    # Standard statistical columns to create rolling features for
    stat_columns = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 'FG', 'FGA', 'FG3M', 'FG3A', 'FT', 'FTA']
    available_stat_columns = [col for col in stat_columns if col in df.columns]
    
    # Create rolling features
    if available_stat_columns:
        df = create_rolling_features(df, available_stat_columns, windows=[3, 5, 10], player_col=player_col)
    
    # Create efficiency features
    df = create_efficiency_features(df)
    
    # Create streak features
    df = create_game_streak_features(df, player_col=player_col)
    
    # Create time-based features
    df = create_time_based_features(df)
    
    # Create matchup features (including opponent-specific stats)
    df = create_matchup_features(df, player_col=player_col)
    
    # Create target variable shift (for time series prediction)
    if target_column in df.columns:
        df[f'{target_column}_next'] = df.groupby(player_col)[target_column].shift(-1)
    
    return df

