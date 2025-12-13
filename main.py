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
from src.schemas.validators import (
    validate_game_logs,
    validate_processed_data,
    validate_feature_data,
    validate_training_request
)


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
                        use_cached: bool = False) -> pd.DataFrame:  # Changed default to False
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
    # Always collect fresh data for new player
    print(f"\nCollecting data for {player_name}...")
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
        print(f"Warning: Could not save data: {e}")
    
    return df


def preprocess_data(df: pd.DataFrame, min_minutes: int = 10, validate: bool = False) -> pd.DataFrame:
    """
    Preprocess the raw game log data.
    
    Args:
        df: Raw DataFrame
        min_minutes: Minimum minutes played to keep game
        validate: Whether to validate data with pydantic schemas
        
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


def create_features(df: pd.DataFrame, target_column: str = 'PTS', validate: bool = False) -> pd.DataFrame:
    """
    Engineer features for machine learning.
    
    Args:
        df: Processed DataFrame
        target_column: Column to predict
        validate: Whether to validate feature data with pydantic schemas
        
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
    Train a machine learning model.
    
    Args:
        df: Feature-engineered DataFrame
        target_column: Column to predict
        model_type: Type of model
        compare_models: Whether to compare multiple model types
        
    Returns:
        Trained predictor instance
    """
    print(f"\nTraining model for {target_column}...")
    
    if compare_models:
        # Train multiple models and select the best
        results = train_multiple_models(df, target_column=target_column)
        best_model_type = results['best_model']
        predictor = results[best_model_type]['predictor']
        metrics = results[best_model_type]['metrics']
        print(f"Best model: {best_model_type}")
        model_filename = f"best_model_{target_column}.pkl"
    else:
        # Train single model
        predictor = PlayerPerformancePredictor(
            model_type=model_type,
            target_column=target_column
        )
        metrics = predictor.train(df, test_size=0.2)
        model_filename = f"{model_type}_{target_column}.pkl"
    
    # Print results
    print("\nTraining Results:")
    print(f"  Train MAE: {metrics['train']['mae']:.2f}, RMSE: {metrics['train']['rmse']:.2f}, R²: {metrics['train']['r2']:.3f}")
    print(f"  Test MAE: {metrics['test']['mae']:.2f}, RMSE: {metrics['test']['rmse']:.2f}, R²: {metrics['test']['r2']:.3f}")
    if 'cross_validation' in metrics:
        print(f"  CV MAE: {metrics['cross_validation']['mae_mean']:.2f} ± {metrics['cross_validation']['mae_std']:.2f}")
    
    # Save model
    model_path = MODEL_DIR / model_filename
    predictor.save_model(str(model_path))
    print(f"Model saved to {model_path}")
    
    # Show feature importance
    try:
        importance_df = predictor.get_feature_importance(top_n=10)
        print("\nTop 10 Most Important Features:")
        print(importance_df.to_string(index=False))
    except:
        pass
    
    return predictor


def make_predictions(predictor: PlayerPerformancePredictor, 
                    df: pd.DataFrame,
                    n_predictions: int = 10) -> pd.DataFrame:
    """
    Make predictions using trained model.
    
    Args:
        predictor: Trained predictor
        df: Feature-engineered DataFrame
        n_predictions: Number of recent predictions to show
        
    Returns:
        DataFrame with predictions and actual values
    """
    print("\nMaking predictions...")
    
    predictions = predictor.predict(df)
    
    # Create results DataFrame
    results_df = df[[predictor.target_column]].copy()
    results_df['predicted'] = predictions
    results_df['error'] = results_df[predictor.target_column] - predictions
    results_df['abs_error'] = results_df['error'].abs()
    
    # Rename target column for clarity
    results_df = results_df.rename(columns={predictor.target_column: f'{predictor.target_column}_actual'})
    
    # Show sample predictions
    print(f"\nSample Predictions (last {n_predictions} games):")
    print(results_df.tail(n_predictions).to_string(index=False))
    
    return results_df


def get_user_input():
    """Get user input for player and model configuration."""
    print("\n" + "=" * 60)
    print("NBA Player Statistics ML Pipeline")
    print("=" * 60)
    print("\nThis tool predicts NBA player statistics using machine learning.")
    print("It analyzes historical game data and creates predictions.\n")
    
    # Get player name
    while True:
        player_name = input("Enter player name (e.g., 'LeBron James', 'Stephen Curry'): ").strip()
        if player_name:
            break
        print("❌ Player name cannot be empty. Please try again.")
    
    # Get target column
    print("\nWhat stat would you like to predict?")
    print("  1. PTS  - Points")
    print("  2. AST  - Assists")
    print("  3. REB  - Rebounds")
    print("  4. STL  - Steals")
    print("  5. BLK  - Blocks")
    
    stat_map = {"1": "PTS", "2": "AST", "3": "REB", "4": "STL", "5": "BLK"}
    while True:
        choice = input("Choose stat (1-5) [default: 1 for PTS]: ").strip() or "1"
        if choice in stat_map:
            target_column = stat_map[choice]
            break
        print("❌ Invalid choice. Please enter 1-5.")
    
    # Get model type
    print("\nChoose model type:")
    print("  1. Random Forest      - Best all-around (recommended)")
    print("  2. Gradient Boosting  - Highest accuracy, slower")
    print("  3. Linear Regression  - Fastest, simple")
    print("  4. Ridge Regression   - Linear with regularization")
    
    model_map = {
        "1": "random_forest",
        "2": "gradient_boosting", 
        "3": "linear",
        "4": "ridge"
    }
    while True:
        choice = input("Choose model (1-4) [default: 1 for Random Forest]: ").strip() or "1"
        if choice in model_map:
            model_type = model_map[choice]
            break
        print("❌ Invalid choice. Please enter 1-4.")
    
    # Get season year
    while True:
        year_input = input("\nEnter season year [default: 2024]: ").strip() or "2024"
        try:
            season_year = int(year_input)
            if 2000 <= season_year <= 2025:
                break
            print("❌ Please enter a year between 2000 and 2025.")
        except ValueError:
            print("❌ Invalid year. Please enter a number.")
    
    # Confirm settings
    print("\n" + "=" * 60)
    print("Configuration Summary:")
    print("=" * 60)
    print(f"Player:       {player_name}")
    print(f"Stat:         {target_column}")
    print(f"Model:        {model_type}")
    print(f"Season:       {season_year}")
    print("=" * 60)
    
    confirm = input("\nProceed with training? (y/n) [default: y]: ").strip().lower() or "y"
    if confirm != "y":
        print("❌ Training cancelled.")
        return None
    
    return {
        "player_name": player_name,
        "target_column": target_column,
        "model_type": model_type,
        "season_year": season_year,
        "season_type": "Regular Season",
        "min_minutes": 10
    }


def main():
    """
    Main pipeline execution.
    """
    # Get user configuration
    config = get_user_input()
    if config is None:
        return
    
    player_name = config["player_name"]
    target_column = config["target_column"]
    model_type = config["model_type"]
    season_type = config["season_type"]
    season_year = config["season_year"]
    min_minutes = config["min_minutes"]
    
    try:
        # Step 1: Load or collect data
        df = load_or_collect_data(
            player_name=player_name,
            season_type=season_type,
            season_year=season_year,
            use_cached=False  # Always get fresh data for user-specified player
        )
        
        if df.empty:
            print(f"\n❌ No data available for {player_name}. Please check the player name and try again.")
            return
        
        # Step 2: Preprocess data
        df = preprocess_data(df, min_minutes=min_minutes, validate=False)
        
        if df.empty:
            print("\n❌ No data after preprocessing. The player may not have enough games meeting the criteria.")
            return
        
        # Step 3: Engineer features
        df = create_features(df, target_column=target_column, validate=False)
        
        # Step 4: Train model
        predictor = train_model(
            df, 
            target_column=target_column,
            model_type=model_type,
            compare_models=False
        )
        
        # Step 5: Make predictions
        results = make_predictions(predictor, df, n_predictions=10)
        
        print("\n" + "="*60)
        print("✅ Pipeline completed successfully!")
        print("="*60)
        print(f"\nModel saved and ready to use!")
        print(f"\nTo make predictions via API:")
        print(f"  1. Start the API: python3 api.py")
        print(f"  2. Use your trained model for predictions")
        
        # Ask if user wants to run another player
        print("\n" + "="*60)
        another = input("\nWould you like to train another model? (y/n): ").strip().lower()
        if another == "y":
            print("\n" * 2)
            main()  # Recursively call main for another run
        
    except Exception as e:
        print(f"\n❌ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Training interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
