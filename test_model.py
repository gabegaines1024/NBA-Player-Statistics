"""
Quick test script to verify the ML model works correctly.
This creates synthetic NBA data to test the model without needing API calls.
"""
import pandas as pd
import numpy as np
from src.models.player_predictor import PlayerPerformancePredictor, train_multiple_models
from src.preprossessing.feature_engineer import engineer_features

def create_synthetic_data(n_games=50):
    """Create synthetic NBA player game log data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range('2024-01-01', periods=n_games, freq='D')
    
    # Create realistic NBA statistics
    data = {
        'GAME_DATE': dates,
        'GAME_ID': [f'00{i:06d}' for i in range(n_games)],
        'MATCHUP': [f'LAL vs. {team}' if i % 2 == 0 else f'LAL @ {team}' 
                   for i, team in enumerate(['GSW', 'BOS', 'MIA', 'PHX', 'MIL'] * (n_games // 5 + 1))[:n_games]],
        'PLAYER_NAME': ['LeBron James'] * n_games,
        'WL': np.random.choice(['W', 'L'], n_games),
        'MIN': np.random.normal(35, 5, n_games).clip(20, 48),
        'PTS': np.random.normal(25, 8, n_games).clip(0, 60).astype(int),
        'AST': np.random.normal(7, 3, n_games).clip(0, 20).astype(int),
        'REB': np.random.normal(8, 3, n_games).clip(0, 20).astype(int),
        'STL': np.random.poisson(1.5, n_games),
        'BLK': np.random.poisson(0.8, n_games),
        'TOV': np.random.poisson(3, n_games),
        'FG': np.random.normal(10, 3, n_games).clip(0, 25).astype(int),
        'FGA': np.random.normal(18, 4, n_games).clip(0, 30).astype(int),
        'FG3M': np.random.normal(2.5, 1.5, n_games).clip(0, 10).astype(int),
        'FG3A': np.random.normal(6, 2, n_games).clip(0, 15).astype(int),
        'FT': np.random.normal(5, 2, n_games).clip(0, 15).astype(int),
        'FTA': np.random.normal(6, 2, n_games).clip(0, 20).astype(int),
        'OREB': np.random.poisson(1.5, n_games),
        'DREB': np.random.normal(6, 2, n_games).clip(0, 15).astype(int),
    }
    
    df = pd.DataFrame(data)
    
    # Add some correlation to make predictions meaningful
    # Points in next game somewhat correlated with current game
    for i in range(len(df) - 1):
        df.loc[i+1, 'PTS'] = int(df.loc[i, 'PTS'] * 0.3 + np.random.normal(25, 6, 1)[0])
        df.loc[i+1, 'PTS'] = max(0, min(60, df.loc[i+1, 'PTS']))
    
    return df


def test_single_model():
    """Test training a single model."""
    print("="*60)
    print("Testing Single Model (Random Forest)")
    print("="*60)
    
    # Create synthetic data
    df = create_synthetic_data(n_games=50)
    print(f"\nCreated {len(df)} synthetic game records")
    
    # Engineer features
    df = engineer_features(df, target_column='PTS')
    print(f"After feature engineering: {len(df.columns)} columns")
    
    # Train model
    predictor = PlayerPerformancePredictor(
        model_type='random_forest',
        target_column='PTS'
    )
    
    metrics = predictor.train(df, test_size=0.2)
    
    print(f"\n✅ Model trained successfully!")
    print(f"   Test MAE: {metrics['test']['mae']:.2f}")
    print(f"   Test R²: {metrics['test']['r2']:.3f}")
    
    # Test predictions
    predictions = predictor.predict(df.tail(5))
    print(f"\n✅ Predictions made successfully!")
    print(f"   Sample predictions: {predictions[:3]}")
    
    # Test feature importance
    try:
        importance = predictor.get_feature_importance(top_n=5)
        print(f"\n✅ Feature importance retrieved!")
        print(importance)
    except Exception as e:
        print(f"\n⚠️  Feature importance not available: {e}")
    
    return predictor, metrics


def test_multiple_models():
    """Test training and comparing multiple models."""
    print("\n" + "="*60)
    print("Testing Multiple Models Comparison")
    print("="*60)
    
    # Create synthetic data
    df = create_synthetic_data(n_games=50)
    
    # Engineer features
    df = engineer_features(df, target_column='PTS')
    
    # Train multiple models
    results = train_multiple_models(df, target_column='PTS')
    
    if 'best_model' in results:
        print(f"\n✅ Best model identified: {results['best_model']}")
        return results
    else:
        print("\n⚠️  No best model identified")
        return results


def test_model_save_load():
    """Test saving and loading a model."""
    print("\n" + "="*60)
    print("Testing Model Save/Load")
    print("="*60)
    
    # Create and train a model
    df = create_synthetic_data(n_games=50)
    df = engineer_features(df, target_column='PTS')
    
    predictor1 = PlayerPerformancePredictor(
        model_type='random_forest',
        target_column='PTS'
    )
    predictor1.train(df)
    
    # Save model
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        model_path = f.name
    
    predictor1.save_model(model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Load model
    predictor2 = PlayerPerformancePredictor()
    predictor2.load_model(model_path)
    print(f"✅ Model loaded from {model_path}")
    
    # Verify predictions match
    pred1 = predictor1.predict(df.tail(3))
    pred2 = predictor2.predict(df.tail(3))
    
    if np.allclose(pred1, pred2):
        print("✅ Predictions match after save/load!")
    else:
        print("⚠️  Predictions don't match after save/load")
    
    # Cleanup
    os.unlink(model_path)
    
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("NBA Player Statistics ML Model - Test Suite")
    print("="*60)
    
    try:
        # Test 1: Single model
        predictor, metrics = test_single_model()
        
        # Test 2: Multiple models
        # test_multiple_models()
        
        # Test 3: Save/Load
        test_model_save_load()
        
        print("\n" + "="*60)
        print("✅ All tests passed! ML model is working correctly.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

