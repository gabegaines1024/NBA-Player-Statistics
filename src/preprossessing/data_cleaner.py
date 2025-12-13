import numpy as np  
import pandas as pd
import scipy.stats as stats

#remove duplicates from the data
def remove_duplicates(data):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        data: pandas DataFrame
        
    Returns:
        DataFrame with duplicates removed
    """
    if isinstance(data, pd.DataFrame):
        return data.drop_duplicates()
    else:
        # If it's a numpy array, convert to DataFrame first
        data_df = pd.DataFrame(data)
        return data_df.drop_duplicates()

#handle missing values
def handle_missing_values(data):
    """
    Handle missing values appropriately based on data type.
    - Numeric columns: fill with median or mean
    - Categorical columns: fill with mode or 'Unknown'
    
    Args:
        data: pandas DataFrame
        
    Returns:
        DataFrame with missing values handled
    """
    data_df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data.copy()
    
    # Handle numeric columns
    numeric_cols = data_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data_df[col].isna().any():
            # Fill with median (more robust to outliers than mean)
            data_df[col] = data_df[col].fillna(data_df[col].median())
    
    # Handle categorical/object columns
    categorical_cols = data_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if data_df[col].isna().any():
            # Fill with mode (most common value) or 'Unknown'
            mode_value = data_df[col].mode()
            if len(mode_value) > 0:
                data_df[col] = data_df[col].fillna(mode_value[0])
            else:
                data_df[col] = data_df[col].fillna('Unknown')
    
    return data_df

#filter minimum minutes played
def filter_minimum_minutes(data, min_minutes: int):
    """
    Filter rows where minutes_played is at least min_minutes.
    
    Args:
        data: pandas DataFrame
        min_minutes: minimum minutes threshold
        
    Returns:
        Filtered DataFrame
    """
    data_df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    
    # Check for minutes column (could be 'minutes', 'MIN', or 'minutes_played')
    if 'minutes' in data_df.columns:
        return data_df[data_df['minutes'] >= min_minutes]
    elif 'MIN' in data_df.columns:
        return data_df[data_df['MIN'] >= min_minutes]
    elif 'minutes_played' in data_df.columns:
        return data_df[data_df['minutes_played'] >= min_minutes]
    else:
        print("Warning: No minutes column found (minutes, MIN, or minutes_played)")
        return data_df
    
#standardize column names
def standardize_column_names(data):
    """
    Standardize column names to a consistent format.
    Note: We keep uppercase names for stat columns to maintain compatibility
    with NBA API and feature engineering.
    
    Args:
        data: pandas DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    data_df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    
    # Create a mapping for standardization
    rename_map = {
        'GAME_ID': 'game_id',
        'GAME_DATE': 'game_date', 
        'MATCHUP': 'matchup',
        'MP': 'minutes',  # Some APIs use MP
        'MIN': 'minutes',  # Most use MIN
        'FG': 'FGM',  # Standardize to FGM (field goals made)
        'FG%': 'FG_PCT',  # Standardize percentage notation
    }
    
    # Apply the renames, but keep most stat columns uppercase (PTS, AST, REB, etc.)
    return data_df.rename(columns=rename_map)

#detect and remove outliers
def detect_and_remove_outliers(data, threshold=3, columns=None):
    """
    Detect and remove outliers using Z-score method on numerical columns only.
    
    Args:
        data: pandas DataFrame
        threshold: Z-score threshold (default 3)
        columns: specific columns to check (None = all numerical columns)
        
    Returns:
        DataFrame with outliers removed
    """
    data_df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data.copy()

    # Only check numerical columns
    if columns is None:
        numerical_cols = data_df.select_dtypes(include=[np.number]).columns
    else:
        numerical_cols = [col for col in columns if col in data_df.columns and 
                         data_df[col].dtype in [np.int64, np.float64]]
    
    if len(numerical_cols) == 0:
        print("Warning: No numerical columns found for outlier detection")
        return data_df
    
    # Calculate Z-scores only for numerical columns
    z_scores = np.abs(stats.zscore(data_df[numerical_cols], nan_policy='omit'))
    outliers = (z_scores > threshold).any(axis=1)
    
    return data_df[~outliers]

#validate data types
def validate_data_types(data, expected_numeric_cols=None):
    """
    Validate that specified columns have correct data types.
    Only validates numerical columns - categorical columns are expected to be strings/objects.
    
    Args:
        data: pandas DataFrame
        expected_numeric_cols: list of column names that should be numeric (None = auto-detect)
        
    Returns:
        dict: validation results with column names and their types
    """
    data_df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    
    validation_results = {}
    
    if expected_numeric_cols is None:
        # Auto-detect: check that statistical columns are numeric
        numeric_cols = data_df.select_dtypes(include=[np.number]).columns
        categorical_cols = data_df.select_dtypes(include=['object', 'datetime64']).columns
        
        validation_results = {
            'numeric_columns': list(numeric_cols),
            'categorical_columns': list(categorical_cols),
            'all_valid': True
        }
    else:
        # Validate specific columns
        for col in expected_numeric_cols:
            if col in data_df.columns:
                is_numeric = pd.api.types.is_numeric_dtype(data_df[col])
                validation_results[col] = {
                    'is_numeric': is_numeric,
                    'dtype': str(data_df[col].dtype)
                }
    
    return validation_results