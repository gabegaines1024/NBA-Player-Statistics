"""
Machine Learning Model for Predicting NBA Player Performance
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import json


class PlayerPerformancePredictor:
    """
    Machine learning model for predicting NBA player performance metrics.
    """
    
    def __init__(self, model_type: str = 'random_forest', target_column: str = 'PTS'):
        """
        Initialize the predictor.
        
        Args:
            model_type: Type of model to use ('random_forest', 'gradient_boosting', 'linear', 'ridge')
            target_column: Column to predict (e.g., 'PTS', 'AST', 'REB')
        """
        self.model_type = model_type
        self.target_column = target_column
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on model_type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def _prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare features and target from DataFrame.
        
        Args:
            df: Input DataFrame
            is_training: Whether this is for training (includes target) or prediction
            
        Returns:
            Tuple of (features_df, target_series or None)
        """
        df = df.copy()
        
        # Remove non-feature columns
        exclude_cols = [
            'GAME_ID', 'GAME_DATE', 'MATCHUP', 'PLAYER_NAME', 'PLAYER_ID',
            'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'WL', 'MIN'
        ]
        
        # If training, we need the target column
        if is_training:
            # For time-series prediction, prefer the _next column if it exists
            # (this is created by feature engineering for next-game prediction)
            target_col_next = f'{self.target_column}_next'
            if target_col_next in df.columns:
                # Use the next-game target for time-series prediction
                self.target_column = target_col_next
            elif self.target_column not in df.columns:
                raise ValueError(f"Target column '{self.target_column}' or '{target_col_next}' not found in DataFrame")
            
            # Remove rows where target is NaN
            df = df.dropna(subset=[self.target_column])
            
            if df.empty:
                raise ValueError("No valid data after removing NaN target values")
            
            target = df[self.target_column].copy()
        else:
            target = None
        
        # Get feature columns (exclude target and metadata)
        exclude_cols.append(self.target_column)
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Select only numeric columns for features
        feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
        
        # Store feature columns for later use
        if is_training:
            self.feature_columns = feature_cols
        
        # Select features
        features = df[feature_cols].copy()
        
        # Handle any remaining NaN values
        features = features.fillna(features.median())
        
        return features, target
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2, 
              use_scaling: bool = True, cv_folds: int = 5) -> Dict:
        """
        Train the model on the provided data.
        
        Args:
            df: Training DataFrame with features and target
            test_size: Proportion of data to use for testing
            use_scaling: Whether to scale features
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training metrics
        """
        # Prepare features and target
        X, y = self._prepare_features(df, is_training=True)
        
        if X.empty:
            raise ValueError("No features available for training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features if requested
        if use_scaling:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Train model
        print(f"Training {self.model_type} model on {len(X_train)} samples...")
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=cv_folds, scoring='neg_mean_absolute_error'
        )
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        metrics = {
            'train': {
                'mae': float(train_mae),
                'rmse': float(train_rmse),
                'r2': float(train_r2)
            },
            'test': {
                'mae': float(test_mae),
                'rmse': float(test_rmse),
                'r2': float(test_r2)
            },
            'cross_validation': {
                'mae_mean': float(cv_mae),
                'mae_std': float(cv_std)
            },
            'n_features': len(self.feature_columns),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }
        
        print(f"\nTraining Results:")
        print(f"  Train MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}, R²: {train_r2:.3f}")
        print(f"  Test MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}, R²: {test_r2:.3f}")
        print(f"  CV MAE: {cv_mae:.2f} ± {cv_std:.2f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame, use_scaling: bool = True) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            df: DataFrame with features
            use_scaling: Whether to scale features
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X, _ = self._prepare_features(df, is_training=False)
        
        # Ensure same columns as training
        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(X.columns)
            if missing_cols:
                for col in missing_cols:
                    X[col] = 0
            
            X = X[self.feature_columns]
        
        # Scale if requested
        if use_scaling:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance scores (for tree-based models).
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError(f"Model type {self.model_type} does not support feature importance")
        
        if self.feature_columns is None:
            raise ValueError("Feature columns not available")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model_type': self.model_type,
            'target_column': self.target_column,
            'feature_columns': self.feature_columns,
            'model': self.model,
            'scaler': self.scaler
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)
        
        self.model_type = model_data['model_type']
        self.target_column = model_data['target_column']
        self.feature_columns = model_data['feature_columns']
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")


def train_multiple_models(df: pd.DataFrame, target_column: str = 'PTS') -> Dict:
    """
    Train multiple models and compare their performance.
    
    Args:
        df: Training DataFrame
        target_column: Column to predict
        
    Returns:
        Dictionary with results for each model
    """
    model_types = ['random_forest', 'gradient_boosting', 'linear', 'ridge']
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type} model")
        print(f"{'='*60}")
        
        try:
            predictor = PlayerPerformancePredictor(
                model_type=model_type,
                target_column=target_column
            )
            metrics = predictor.train(df)
            results[model_type] = {
                'metrics': metrics,
                'predictor': predictor
            }
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            results[model_type] = {'error': str(e)}
    
    # Find best model
    best_model = None
    best_score = float('inf')
    
    for model_type, result in results.items():
        if 'metrics' in result:
            test_mae = result['metrics']['test']['mae']
            if test_mae < best_score:
                best_score = test_mae
                best_model = model_type
    
    if best_model:
        print(f"\n{'='*60}")
        print(f"Best Model: {best_model} (Test MAE: {best_score:.2f})")
        print(f"{'='*60}")
        results['best_model'] = best_model
    
    return results

