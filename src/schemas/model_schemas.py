"""
Pydantic schemas for ML model training and prediction validation.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any


class ModelMetrics(BaseModel):
    """
    Schema for model training metrics.
    """
    train_mae: float = Field(..., ge=0, description="Training Mean Absolute Error")
    train_rmse: float = Field(..., ge=0, description="Training Root Mean Squared Error")
    train_r2: float = Field(..., description="Training R-squared score")
    test_mae: float = Field(..., ge=0, description="Test Mean Absolute Error")
    test_rmse: float = Field(..., ge=0, description="Test Root Mean Squared Error")
    test_r2: float = Field(..., description="Test R-squared score")
    cv_mae_mean: Optional[float] = Field(None, ge=0, description="Cross-validation MAE mean")
    cv_mae_std: Optional[float] = Field(None, ge=0, description="Cross-validation MAE standard deviation")
    n_features: int = Field(..., ge=1, description="Number of features used")
    n_train_samples: int = Field(..., ge=1, description="Number of training samples")
    n_test_samples: int = Field(..., ge=1, description="Number of test samples")


class ModelTrainingRequest(BaseModel):
    """
    Schema for model training request parameters.
    """
    player_name: str = Field(..., min_length=1, description="Player name")
    target_column: str = Field(default="PTS", description="Target column to predict")
    model_type: str = Field(
        default="random_forest",
        description="Model type: random_forest, gradient_boosting, linear, or ridge"
    )
    season_type: str = Field(default="Regular Season", description="Season type")
    season_year: int = Field(default=2024, ge=2000, le=2100, description="Season year")
    min_minutes: int = Field(default=10, ge=0, le=48, description="Minimum minutes played filter")
    test_size: float = Field(default=0.2, gt=0, lt=1, description="Test set size proportion")
    compare_models: bool = Field(default=False, description="Whether to compare multiple models")
    
    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        """Validate model type."""
        valid_types = ['random_forest', 'gradient_boosting', 'linear', 'ridge']
        if v not in valid_types:
            raise ValueError(f"Model type must be one of {valid_types}, got {v}")
        return v
    
    @field_validator('target_column')
    @classmethod
    def validate_target_column(cls, v: str) -> str:
        """Validate target column."""
        valid_targets = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 'FG', 'FGA', 'FG3M', 'FG3A']
        if v.upper() not in valid_targets:
            # Allow custom targets but warn
            return v.upper()
        return v.upper()


class ModelTrainingResponse(BaseModel):
    """
    Schema for model training response.
    """
    success: bool = Field(..., description="Whether training was successful")
    model_type: str = Field(..., description="Model type used")
    target_column: str = Field(..., description="Target column predicted")
    metrics: ModelMetrics = Field(..., description="Training metrics")
    model_path: Optional[str] = Field(None, description="Path to saved model file")
    feature_importance: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Top feature importance scores"
    )
    error: Optional[str] = Field(None, description="Error message if training failed")


class PredictionRequest(BaseModel):
    """
    Schema for prediction request.
    """
    model_path: Optional[str] = Field(None, description="Path to saved model file")
    player_name: Optional[str] = Field(None, description="Player name (if loading from cache)")
    target_column: str = Field(default="PTS", description="Target column to predict")
    opponent_team: Optional[str] = Field(
        None, 
        description="Opponent team abbreviation (e.g., 'BOS', 'LAL') for matchup-specific prediction"
    )
    game_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Game data for prediction (if not using cached data)"
    )
    n_predictions: int = Field(default=5, ge=1, le=100, description="Number of predictions to return")
    
    @field_validator('target_column')
    @classmethod
    def validate_target_column(cls, v: str) -> str:
        """Validate target column."""
        return v.upper()
    
    @field_validator('opponent_team')
    @classmethod
    def validate_opponent_team(cls, v: Optional[str]) -> Optional[str]:
        """Validate and uppercase opponent team abbreviation."""
        if v is not None:
            return v.upper().strip()
        return v


class PredictionResponse(BaseModel):
    """
    Schema for prediction response.
    """
    success: bool = Field(..., description="Whether prediction was successful")
    predictions: List[float] = Field(..., description="Predicted values")
    actual_values: Optional[List[float]] = Field(None, description="Actual values (if available)")
    errors: Optional[List[float]] = Field(None, description="Prediction errors (if actual values available)")
    game_dates: Optional[List[str]] = Field(None, description="Game dates for predictions")
    matchups: Optional[List[str]] = Field(None, description="Matchups for predictions")
    error: Optional[str] = Field(None, description="Error message if prediction failed")

