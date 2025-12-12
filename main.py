"""
Main entry point for NBA Player Statistics ML Pipeline
"""
import pandas as pd
import json
from pathlib import Path
from typing import Optional

from src.data_collection.nba_api_collector import get_player_game_logs
from src.preprossessing.data_cleaner import (
    remove_duplicates,
    handle_missing_values,
    filter_minimum_minutes,
    standardize_column_names,
    detect_and_remove_outliers
)
from src.preprossessing.feature_engineer import engineer_features
from src.models.player_predictor import PlayerPerformancePredictor, train_multiple_models


# Define paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_FILE = DATA_DIR / "raw.json"
PROCESSED_DATA_FILE = DATA_DIR / "processed.json"
FEATURE_DATA_FILE = DATA_DIR / "feature.json"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_or_collect_data(player_name: str = "LeBron James", 
                        season_type: str = "Regular Season",
                        season_year: int = 2024,
                        use_cached: bool = True) -> pd.DataFrame:
    """
    Load data from file or collect from API.
    
    Args:
        player_name: Name of player to get data for
        season_type: Type of season
        season_year: Season year
        use_cached: Whether to use cached data if available
        
    Returns:
        DataFrame with player game logs
    """
    # Try to load from file first
    if use_cached and RAW_DATA_FILE.exists():
        try:
            with open(RAW_DATA_FILE, 'r') as f:
                data = json.load(f)
            
            if data:
                df = pd.DataFrame(data)
                print(f"Loaded {len(df)} records from {RAW_DATA_FILE}")
                return df
        except Exception as e:
            print(f"Error loading cached data: {e}")
    
    # Collect from API
    print(f"Collecting data for {player_name}...")
    game_logs = get_player_game_logs(player_name, season_type, season_year)
    
    if game_logs is None:
        raise ValueError(f"Failed to collect data for {player_name}")
    
    # Ensure we have a DataFrame
    if isinstance(game_logs, pd.DataFrame):
        df = game_logs.copy()
    else:
        # If it's not a DataFrame, try to convert
        df = pd.DataFrame(game_logs)
    
    # Save raw data
    try:
        records = df.to_dict('records')
        with open(RAW_DATA_FILE, 'w') as f:
            json.dump(records, f, indent=2, default=str)
        print(f"Saved {len(df)} records to {RAW_DATA_FILE}")
    except Exception as e:
        print(f"Warning: Could not save raw data: {e}")
    
    return df


def preprocess_data(df: pd.DataFrame, min_minutes: int = 10) -> pd.DataFrame:
    """
    Preprocess the data: clean, remove duplicates, handle missing values, etc.
    
    Args:
        df: Raw DataFrame
        min_minutes: Minimum minutes played to include
        
    Returns:
        Cleaned DataFrame
    """
    print("\nPreprocessing data...")
    
    # Standardize column names
    df = standardize_column_names(df)
    
    # Remove duplicates
    df = remove_duplicates(df)
    print(f"  After removing duplicates: {len(df)} records")
    
    # Filter minimum minutes
    df = filter_minimum_minutes(df, min_minutes)
    print(f"  After filtering minimum minutes ({min_minutes}): {len(df)} records")
    
    # Handle missing values
    df = handle_missing_values(df)
    print(f"  After handling missing values: {len(df)} records")
    
    # Remove outliers
    df = detect_and_remove_outliers(df, threshold=3)
    print(f"  After removing outliers: {len(df)} records")
    
    # Save processed data
    try:
        records = df.to_dict('records')
        with open(PROCESSED_DATA_FILE, 'w') as f:
            json.dump(records, f, indent=2, default=str)
        print(f"  Saved processed data to {PROCESSED_DATA_FILE}")
    except Exception as e:
        print(f"  Warning: Could not save processed data: {e}")
    
    return df


def create_features(df: pd.DataFrame, target_column: str = 'PTS') -> pd.DataFrame:
    """
    Engineer features for machine learning.
    
    Args:
        df: Processed DataFrame
        target_column: Column to predict
        
    Returns:
        DataFrame with engineered features
    """
    print(f"\nEngineering features for target: {target_column}...")
    
    df = engineer_features(df, target_column=target_column)
    
    print(f"  Created features. Total columns: {len(df.columns)}")
    
    # Save feature data
    try:
        records = df.to_dict('records')
        with open(FEATURE_DATA_FILE, 'w') as f:
            json.dump(records, f, indent=2, default=str)
        print(f"  Saved feature data to {FEATURE_DATA_FILE}")
    except Exception as e:
        print(f"  Warning: Could not save feature data: {e}")
    
    return df


def train_model(df: pd.DataFrame, 
                target_column: str = 'PTS',
                model_type: str = 'random_forest',
                compare_models: bool = False) -> PlayerPerformancePredictor:
    """
    Train the machine learning model.
    
    Args:
        df: DataFrame with features
        target_column: Column to predict
        model_type: Type of model to train
        compare_models: Whether to compare multiple models
        
    Returns:
        Trained predictor
    """
    print(f"\nTraining model for {target_column}...")
    
    if compare_models:
        # Train and compare multiple models
        results = train_multiple_models(df, target_column=target_column)
        
        if 'best_model' in results:
            best_type = results['best_model']
            predictor = results[best_type]['predictor']
            
            # Save best model
            model_path = MODEL_DIR / f"best_model_{target_column}.pkl"
            predictor.save_model(str(model_path))
            
            return predictor
        else:
            raise ValueError("No models were successfully trained")
    else:
        # Train single model
        predictor = PlayerPerformancePredictor(
            model_type=model_type,
            target_column=target_column
        )
        
        metrics = predictor.train(df)
        
        # Save model
        model_path = MODEL_DIR / f"{model_type}_{target_column}.pkl"
        predictor.save_model(str(model_path))
        
        # Print feature importance
        try:
            importance = predictor.get_feature_importance(top_n=10)
            print("\nTop 10 Most Important Features:")
            print(importance.to_string(index=False))
        except Exception as e:
            print(f"  Could not get feature importance: {e}")
        
        return predictor


def make_predictions(predictor: PlayerPerformancePredictor, 
                    df: pd.DataFrame,
                    n_predictions: int = 5) -> pd.DataFrame:
    """
    Make predictions on the data.
    
    Args:
        predictor: Trained predictor
        df: DataFrame with features
        n_predictions: Number of predictions to show
        
    Returns:
        DataFrame with predictions
    """
    print(f"\nMaking predictions...")
    
    predictions = predictor.predict(df)
    
    # Create results DataFrame with available columns
    result_cols = []
    if 'GAME_DATE' in df.columns:
        result_cols.append('GAME_DATE')
    if 'MATCHUP' in df.columns:
        result_cols.append('MATCHUP')
    if predictor.target_column in df.columns:
        result_cols.append(predictor.target_column)
    
    if result_cols:
        results_df = df[result_cols].copy()
    else:
        results_df = pd.DataFrame(index=df.index)
    
    results_df['predicted'] = predictions
    
    # Calculate error if we have the target column
    if predictor.target_column in df.columns:
        results_df['error'] = results_df[predictor.target_column] - results_df['predicted']
        results_df['abs_error'] = results_df['error'].abs()
    
    # Show sample predictions
    print(f"\nSample Predictions (last {n_predictions} games):")
    print(results_df.tail(n_predictions).to_string(index=False))
    
    return results_df


def main():
    """
    Main pipeline execution.
    """
    print("="*60)
    print("NBA Player Statistics ML Pipeline")
    print("="*60)
    
    # Configuration
    player_name = "LeBron James"  # Change this to any player
    target_column = "PTS"  # Points, AST, REB, etc.
    model_type = "random_forest"  # or 'gradient_boosting', 'linear', 'ridge'
    compare_models = False  # Set to True to compare all models
    
    try:
        # Step 1: Load or collect data
        df = load_or_collect_data(
            player_name=player_name,
            season_type="Regular Season",
            season_year=2024,
            use_cached=True
        )
        
        if df.empty:
            print("No data available. Exiting.")
            return
        
        # Step 2: Preprocess data
        df = preprocess_data(df, min_minutes=10)
        
        if df.empty:
            print("No data after preprocessing. Exiting.")
            return
        
        # Step 3: Engineer features
        df = create_features(df, target_column=target_column)
        
        # Step 4: Train model
        predictor = train_model(
            df, 
            target_column=target_column,
            model_type=model_type,
            compare_models=compare_models
        )
        
        # Step 5: Make predictions
        results = make_predictions(predictor, df, n_predictions=10)
        
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError in pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

