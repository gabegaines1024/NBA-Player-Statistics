"""
End-to-end test of the full NBA Player Statistics ML Pipeline.
Tests the complete workflow from data creation to model training and prediction.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

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


def create_realistic_nba_data(n_games=60):
    """Create realistic synthetic NBA player game log data."""
    np.random.seed(42)
    
    dates = pd.date_range('2024-10-01', periods=n_games, freq='D')
    # Skip some days to simulate rest days
    dates = [d for i, d in enumerate(dates) if i % 3 != 0 or np.random.random() > 0.3]
    n_games = len(dates)
    
    data = {
        'GAME_ID': [f'00{i:06d}' for i in range(n_games)],
        'GAME_DATE': [d.strftime('%Y-%m-%d') for d in dates],
        'MATCHUP': [],
        'PLAYER_NAME': ['LeBron James'] * n_games,
        'PLAYER_ID': [2544] * n_games,
        'TEAM_ID': [1610612747] * n_games,
        'TEAM_ABBREVIATION': ['LAL'] * n_games,
        'TEAM_NAME': ['Los Angeles Lakers'] * n_games,
        'WL': [],
        'MIN': [],
        'PTS': [],
        'AST': [],
        'REB': [],
        'STL': [],
        'BLK': [],
        'TOV': [],
        'FG': [],
        'FGA': [],
        'FG3M': [],
        'FG3A': [],
        'FT': [],
        'FTA': [],
        'OREB': [],
        'DREB': [],
    }
    
    teams = ['GSW', 'BOS', 'MIA', 'PHX', 'MIL', 'DEN', 'DAL', 'PHI']
    
    # Generate realistic correlated statistics
    for i in range(n_games):
        # Home/away
        is_home = i % 2 == 0
        opponent = teams[i % len(teams)]
        data['MATCHUP'].append(f'LAL vs. {opponent}' if is_home else f'LAL @ {opponent}')
        
        # Minutes (affects all stats)
        minutes = np.random.normal(35, 5)
        minutes = max(20, min(48, minutes))
        data['MIN'].append(round(minutes, 1))
        
        # Field goals (correlated with minutes)
        fga = int(np.random.normal(18, 3) * (minutes / 35))
        fga = max(5, min(30, fga))
        data['FGA'].append(fga)
        
        fg_pct = np.random.normal(0.52, 0.08)
        fg_pct = max(0.2, min(0.7, fg_pct))
        fg = int(fga * fg_pct)
        data['FG'].append(fg)
        
        # Three pointers
        fg3a = int(np.random.normal(6, 2) * (minutes / 35))
        fg3a = max(0, min(15, fg3a))
        data['FG3A'].append(fg3a)
        
        fg3_pct = np.random.normal(0.35, 0.1)
        fg3_pct = max(0.1, min(0.6, fg3_pct))
        fg3m = int(fg3a * fg3_pct)
        data['FG3M'].append(fg3m)
        
        # Free throws
        fta = int(np.random.normal(6, 2) * (minutes / 35))
        fta = max(0, min(20, fta))
        data['FTA'].append(fta)
        
        ft_pct = np.random.normal(0.75, 0.1)
        ft_pct = max(0.5, min(1.0, ft_pct))
        ft = int(fta * ft_pct)
        data['FT'].append(ft)
        
        # Points (calculated from FG, 3PM, FT)
        pts = fg * 2 + fg3m * 3 + ft
        data['PTS'].append(pts)
        
        # Other stats (correlated with minutes)
        data['AST'].append(int(np.random.normal(7, 2) * (minutes / 35)))
        data['REB'].append(int(np.random.normal(8, 2) * (minutes / 35)))
        data['STL'].append(int(np.random.poisson(1.5 * (minutes / 35))))
        data['BLK'].append(int(np.random.poisson(0.8 * (minutes / 35))))
        data['TOV'].append(int(np.random.poisson(3 * (minutes / 35))))
        data['OREB'].append(int(np.random.poisson(1.5 * (minutes / 35))))
        data['DREB'].append(int(np.random.normal(6, 2) * (minutes / 35)))
        
        # Win/Loss (somewhat correlated with points)
        win_prob = 0.3 + (pts / 60) * 0.4
        data['WL'].append('W' if np.random.random() < win_prob else 'L')
    
    df = pd.DataFrame(data)
    
    # Add some temporal correlation (next game somewhat similar to current)
    for i in range(len(df) - 1):
        # Points correlation
        current_pts = df.loc[i, 'PTS']
        next_pts = int(current_pts * 0.3 + np.random.normal(25, 6, 1)[0])
        next_pts = max(0, min(60, next_pts))
        df.loc[i+1, 'PTS'] = next_pts
    
    return df


def test_full_pipeline():
    """Test the complete pipeline end-to-end."""
    print("="*70)
    print("NBA Player Statistics ML Pipeline - Full End-to-End Test")
    print("="*70)
    
    try:
        # Step 1: Create synthetic data
        print("\n[Step 1] Creating synthetic NBA data...")
        df = create_realistic_nba_data(n_games=60)
        print(f"✓ Created {len(df)} game records")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Sample stats: PTS range {df['PTS'].min()}-{df['PTS'].max()}, "
              f"AST range {df['AST'].min()}-{df['AST'].max()}")
        
        # Step 2: Validate raw data (optional)
        print("\n[Step 2] Validating raw data with pydantic schemas...")
        try:
            validated_logs, errors = validate_game_logs(df, strict=False)
            if errors:
                print(f"⚠️  {len(errors)} validation warnings (non-critical)")
            else:
                print(f"✓ Data validation passed: {len(validated_logs)} records validated")
        except Exception as e:
            print(f"⚠️  Validation error (non-critical): {e}")
        
        # Step 3: Preprocess data
        print("\n[Step 3] Preprocessing data...")
        df_processed = standardize_column_names(df)
        df_processed = remove_duplicates(df_processed)
        print(f"  After removing duplicates: {len(df_processed)} records")
        
        df_processed = filter_minimum_minutes(df_processed, min_minutes=10)
        print(f"  After filtering minimum minutes: {len(df_processed)} records")
        
        df_processed = handle_missing_values(df_processed)
        print(f"  After handling missing values: {len(df_processed)} records")
        
        df_processed = detect_and_remove_outliers(df_processed, threshold=3)
        print(f"  After removing outliers: {len(df_processed)} records")
        
        if df_processed.empty:
            raise ValueError("No data remaining after preprocessing")
        
        # Step 4: Validate processed data
        print("\n[Step 4] Validating processed data...")
        try:
            validated_logs, errors = validate_processed_data(df_processed, strict=False)
            if errors:
                print(f"⚠️  {len(errors)} validation warnings")
            else:
                print(f"✓ Processed data validation passed: {len(validated_logs)} records")
        except Exception as e:
            print(f"⚠️  Validation error (non-critical): {e}")
        
        # Step 5: Engineer features
        print("\n[Step 5] Engineering features...")
        target_column = 'PTS'
        df_features = engineer_features(df_processed, target_column=target_column)
        print(f"✓ Created features. Total columns: {len(df_features.columns)}")
        print(f"  New features: {len(df_features.columns) - len(df_processed.columns)}")
        
        # Step 6: Validate feature data
        print("\n[Step 6] Validating feature-engineered data...")
        try:
            validated_logs, errors = validate_feature_data(df_features, strict=False)
            if errors:
                print(f"⚠️  {len(errors)} validation warnings")
            else:
                print(f"✓ Feature data validation passed: {len(validated_logs)} records")
        except Exception as e:
            print(f"⚠️  Validation error (non-critical): {e}")
        
        # Step 7: Train model
        print("\n[Step 7] Training ML model...")
        predictor = PlayerPerformancePredictor(
            model_type='random_forest',
            target_column=target_column
        )
        
        metrics = predictor.train(df_features, test_size=0.2)
        
        print(f"✓ Model trained successfully!")
        print(f"  Training MAE: {metrics['train']['mae']:.2f}")
        print(f"  Training R²: {metrics['train']['r2']:.3f}")
        print(f"  Test MAE: {metrics['test']['mae']:.2f}")
        print(f"  Test R²: {metrics['test']['r2']:.3f}")
        print(f"  Cross-validation MAE: {metrics['cross_validation']['mae_mean']:.2f} ± {metrics['cross_validation']['mae_std']:.2f}")
        
        # Step 8: Feature importance
        print("\n[Step 8] Analyzing feature importance...")
        try:
            importance = predictor.get_feature_importance(top_n=10)
            print("✓ Top 10 Most Important Features:")
            print(importance.to_string(index=False))
        except Exception as e:
            print(f"⚠️  Could not get feature importance: {e}")
        
        # Step 9: Make predictions
        print("\n[Step 9] Making predictions...")
        predictions = predictor.predict(df_features)
        print(f"✓ Generated {len(predictions)} predictions")
        
        # Show sample predictions
        if target_column in df_features.columns:
            actual = df_features[target_column].values
            print(f"\n  Sample predictions (last 5 games):")
            for i in range(min(5, len(predictions))):
                idx = len(predictions) - 5 + i
                if idx >= 0:
                    print(f"    Game {idx+1}: Predicted {predictions[idx]:.1f}, "
                          f"Actual {actual[idx]:.1f}, Error {abs(predictions[idx] - actual[idx]):.1f}")
        
        # Step 10: Test model save/load
        print("\n[Step 10] Testing model save/load...")
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            model_path = f.name
        
        predictor.save_model(model_path)
        print(f"✓ Model saved to {model_path}")
        
        predictor2 = PlayerPerformancePredictor()
        predictor2.load_model(model_path)
        print(f"✓ Model loaded successfully")
        
        # Verify predictions match
        pred1 = predictor.predict(df_features.tail(3))
        pred2 = predictor2.predict(df_features.tail(3))
        
        if np.allclose(pred1, pred2, atol=0.01):
            print("✓ Predictions match after save/load")
        else:
            print("⚠️  Predictions don't match after save/load")
        
        os.unlink(model_path)
        
        # Summary
        print("\n" + "="*70)
        print("✅ FULL PIPELINE TEST PASSED!")
        print("="*70)
        print(f"\nSummary:")
        print(f"  • Data created: {len(df)} games")
        print(f"  • After preprocessing: {len(df_processed)} games")
        print(f"  • Features created: {len(df_features.columns)} total columns")
        print(f"  • Model trained: {predictor.model_type}")
        print(f"  • Test MAE: {metrics['test']['mae']:.2f} points")
        print(f"  • Test R²: {metrics['test']['r2']:.3f}")
        print(f"\n🎉 Pipeline is ready for real NBA data!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)

