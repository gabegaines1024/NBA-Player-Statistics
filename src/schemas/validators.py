"""
Validation utilities using pydantic schemas.
"""
import pandas as pd
from typing import List, Optional, Dict, Any
from .game_log_schemas import PlayerGameLog, ProcessedGameLog, FeatureEngineeredGameLog
from .model_schemas import ModelTrainingRequest, PredictionRequest


def validate_game_logs(df: pd.DataFrame, strict: bool = False) -> tuple[List[PlayerGameLog], List[str]]:
    """
    Validate a DataFrame of game logs using pydantic schemas.
    
    Args:
        df: DataFrame with game log data
        strict: If True, raise errors on validation failure. If False, collect errors.
        
    Returns:
        Tuple of (validated_game_logs, errors)
    """
    validated_logs = []
    errors = []
    
    for idx, row in df.iterrows():
        try:
            # Convert row to dict and validate
            log_dict = row.to_dict()
            validated_log = PlayerGameLog(**log_dict)
            validated_logs.append(validated_log)
        except Exception as e:
            error_msg = f"Row {idx}: {str(e)}"
            errors.append(error_msg)
            if strict:
                raise ValueError(error_msg)
    
    return validated_logs, errors


def validate_processed_data(df: pd.DataFrame, strict: bool = False) -> tuple[List[ProcessedGameLog], List[str]]:
    """
    Validate processed game log data.
    
    Args:
        df: DataFrame with processed game log data
        strict: If True, raise errors on validation failure
        
    Returns:
        Tuple of (validated_logs, errors)
    """
    validated_logs = []
    errors = []
    
    for idx, row in df.iterrows():
        try:
            log_dict = row.to_dict()
            validated_log = ProcessedGameLog(**log_dict)
            validated_logs.append(validated_log)
        except Exception as e:
            error_msg = f"Row {idx}: {str(e)}"
            errors.append(error_msg)
            if strict:
                raise ValueError(error_msg)
    
    return validated_logs, errors


def validate_feature_data(df: pd.DataFrame, strict: bool = False) -> tuple[List[FeatureEngineeredGameLog], List[str]]:
    """
    Validate feature-engineered game log data.
    
    Args:
        df: DataFrame with feature-engineered data
        strict: If True, raise errors on validation failure
        
    Returns:
        Tuple of (validated_logs, errors)
    """
    validated_logs = []
    errors = []
    
    for idx, row in df.iterrows():
        try:
            log_dict = row.to_dict()
            validated_log = FeatureEngineeredGameLog(**log_dict)
            validated_logs.append(validated_log)
        except Exception as e:
            error_msg = f"Row {idx}: {str(e)}"
            errors.append(error_msg)
            if strict:
                raise ValueError(error_msg)
    
    return validated_logs, errors


def validate_training_request(request_dict: Dict[str, Any]) -> ModelTrainingRequest:
    """
    Validate a model training request.
    
    Args:
        request_dict: Dictionary with training request parameters
        
    Returns:
        Validated ModelTrainingRequest
    """
    return ModelTrainingRequest(**request_dict)


def validate_prediction_request(request_dict: Dict[str, Any]) -> PredictionRequest:
    """
    Validate a prediction request.
    
    Args:
        request_dict: Dictionary with prediction request parameters
        
    Returns:
        Validated PredictionRequest
    """
    return PredictionRequest(**request_dict)

