"""
FastAPI application for NBA Player Statistics ML Pipeline.
Provides REST API endpoints for model training and predictions.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime
import traceback

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
from src.schemas.model_schemas import (
    ModelTrainingRequest,
    ModelTrainingResponse,
    PredictionRequest,
    PredictionResponse,
    ModelMetrics
)
from src.schemas.validators import validate_training_request, validate_prediction_request

# Initialize FastAPI app
app = FastAPI(
    title="NBA Player Statistics ML API",
    description="API for training ML models and making predictions on NBA player performance",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# In-memory storage for training jobs (in production, use a database)
training_jobs: Dict[str, Dict[str, Any]] = {}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "NBA Player Statistics ML API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "train": "/train",
            "train_with_tuning": "/train/tuning",
            "predict": "/predict",
            "models": "/models",
            "training_status": "/training/{job_id}"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/train", response_model=ModelTrainingResponse)
async def train_model(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Train a model on NBA player data.
    
    Args:
        request: Training request with player name, model type, etc.
        
    Returns:
        Training response with metrics and model path
    """
    try:
        # Validate request
        validated_request = validate_training_request(request.model_dump())
        
        print(f"Training model for {validated_request.player_name}...")
        
        # Step 1: Collect data
        df = get_player_game_logs(
            player_name=validated_request.player_name,
            season_type=validated_request.season_type,
            season_year=validated_request.season_year
        )
        
        if df is None or df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for player {validated_request.player_name}"
            )
        
        # Step 2: Preprocess
        df = standardize_column_names(df)
        df = remove_duplicates(df)
        df = filter_minimum_minutes(df, validated_request.min_minutes)
        df = handle_missing_values(df)
        df = detect_and_remove_outliers(df, threshold=3)
        
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="No data remaining after preprocessing"
            )
        
        # Step 3: Engineer features
        df = engineer_features(df, target_column=validated_request.target_column)
        
        # Step 4: Train model
        if validated_request.compare_models:
            # Train multiple models and select best
            results = train_multiple_models(df, target_column=validated_request.target_column)
            
            if 'best_model' not in results:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to train any models"
                )
            
            best_type = results['best_model']
            predictor = results[best_type]['predictor']
            metrics_dict = results[best_type]['metrics']
        else:
            # Train single model
            predictor = PlayerPerformancePredictor(
                model_type=validated_request.model_type,
                target_column=validated_request.target_column
            )
            metrics_dict = predictor.train(df, test_size=validated_request.test_size)
        
        # Save model
        model_filename = f"{validated_request.model_type}_{validated_request.target_column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model_path = MODEL_DIR / model_filename
        predictor.save_model(str(model_path))
        
        # Get feature importance
        feature_importance = None
        try:
            importance_df = predictor.get_feature_importance(top_n=10)
            feature_importance = importance_df.to_dict('records')
        except:
            pass
        
        # Create response
        metrics = ModelMetrics(
            train_mae=metrics_dict['train']['mae'],
            train_rmse=metrics_dict['train']['rmse'],
            train_r2=metrics_dict['train']['r2'],
            test_mae=metrics_dict['test']['mae'],
            test_rmse=metrics_dict['test']['rmse'],
            test_r2=metrics_dict['test']['r2'],
            cv_mae_mean=metrics_dict.get('cross_validation', {}).get('mae_mean'),
            cv_mae_std=metrics_dict.get('cross_validation', {}).get('mae_std'),
            n_features=metrics_dict['n_features'],
            n_train_samples=metrics_dict['n_train_samples'],
            n_test_samples=metrics_dict['n_test_samples']
        )
        
        response = ModelTrainingResponse(
            success=True,
            model_type=validated_request.model_type,
            target_column=validated_request.target_column,
            metrics=metrics,
            model_path=str(model_path),
            feature_importance=feature_importance
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error training model: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error training model: {str(e)}"
        )


@app.post("/train/tuning", response_model=ModelTrainingResponse)
async def train_model_with_tuning(
    request: ModelTrainingRequest,
    n_iter: Optional[int] = None
):
    """
    Train a model with hyperparameter tuning.
    
    Args:
        request: Training request with player name, model type, etc.
        n_iter: Number of iterations for randomized search (None for grid search)
        
    Returns:
        Training response with metrics and model path
    """
    try:
        # Validate request
        validated_request = validate_training_request(request.model_dump())
        
        print(f"Training model with hyperparameter tuning for {validated_request.player_name}...")
        
        # Step 1: Collect data
        df = get_player_game_logs(
            player_name=validated_request.player_name,
            season_type=validated_request.season_type,
            season_year=validated_request.season_year
        )
        
        if df is None or df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for player {validated_request.player_name}"
            )
        
        # Step 2: Preprocess
        df = standardize_column_names(df)
        df = remove_duplicates(df)
        df = filter_minimum_minutes(df, validated_request.min_minutes)
        df = handle_missing_values(df)
        df = detect_and_remove_outliers(df, threshold=3)
        
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="No data remaining after preprocessing"
            )
        
        # Step 3: Engineer features
        df = engineer_features(df, target_column=validated_request.target_column)
        
        # Step 4: Train model with tuning
        predictor = PlayerPerformancePredictor(
            model_type=validated_request.model_type,
            target_column=validated_request.target_column
        )
        metrics_dict = predictor.train_with_tuning(
            df,
            test_size=validated_request.test_size,
            n_iter=n_iter
        )
        
        # Save model
        model_filename = f"{validated_request.model_type}_{validated_request.target_column}_tuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model_path = MODEL_DIR / model_filename
        predictor.save_model(str(model_path))
        
        # Get feature importance
        feature_importance = None
        try:
            importance_df = predictor.get_feature_importance(top_n=10)
            feature_importance = importance_df.to_dict('records')
        except:
            pass
        
        # Create response
        metrics = ModelMetrics(
            train_mae=metrics_dict['train']['mae'],
            train_rmse=metrics_dict['train']['rmse'],
            train_r2=metrics_dict['train']['r2'],
            test_mae=metrics_dict['test']['mae'],
            test_rmse=metrics_dict['test']['rmse'],
            test_r2=metrics_dict['test']['r2'],
            cv_mae_mean=None,
            cv_mae_std=None,
            n_features=metrics_dict['n_features'],
            n_train_samples=metrics_dict['n_train_samples'],
            n_test_samples=metrics_dict['n_test_samples']
        )
        
        response = ModelTrainingResponse(
            success=True,
            model_type=validated_request.model_type,
            target_column=validated_request.target_column,
            metrics=metrics,
            model_path=str(model_path),
            feature_importance=feature_importance
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error training model with tuning: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error training model: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse)
async def make_predictions(request: PredictionRequest):
    """
    Make predictions using a trained model.
    
    Args:
        request: Prediction request with model path or player name
        
    Returns:
        Prediction response with predicted values
    """
    try:
        # Validate request
        validated_request = validate_prediction_request(request.model_dump())
        
        # Load model
        if not validated_request.model_path:
            raise HTTPException(
                status_code=400,
                detail="model_path is required"
            )
        
        model_path = Path(validated_request.model_path)
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model not found at {validated_request.model_path}"
            )
        
        predictor = PlayerPerformancePredictor()
        predictor.load_model(str(model_path))
        
        # Get data for prediction
        if validated_request.player_name:
            # Collect fresh data
            df = get_player_game_logs(
                player_name=validated_request.player_name,
                season_type="Regular Season",
                season_year=2024
            )
            
            if df is None or df.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"No data found for player {validated_request.player_name}"
                )
            
            # Preprocess
            df = standardize_column_names(df)
            df = remove_duplicates(df)
            df = handle_missing_values(df)
            
            # Engineer features
            df = engineer_features(df, target_column=validated_request.target_column)
            
        elif validated_request.game_data:
            # Use provided game data
            df = pd.DataFrame([validated_request.game_data])
            df = engineer_features(df, target_column=validated_request.target_column)
        else:
            raise HTTPException(
                status_code=400,
                detail="Either player_name or game_data must be provided"
            )
        
        # Make predictions
        predictions = predictor.predict(df)
        
        # Get last n predictions
        n = min(validated_request.n_predictions, len(predictions))
        pred_values = predictions[-n:].tolist()
        
        # Get actual values if available
        actual_values = None
        errors = None
        if predictor.target_column in df.columns:
            actual_values = df[predictor.target_column].values[-n:].tolist()
            errors = [abs(p - a) for p, a in zip(pred_values, actual_values)]
        
        # Get game dates and matchups if available
        game_dates = None
        matchups = None
        if 'GAME_DATE' in df.columns:
            game_dates = df['GAME_DATE'].values[-n:].astype(str).tolist()
        if 'MATCHUP' in df.columns:
            matchups = df['MATCHUP'].values[-n:].tolist()
        
        response = PredictionResponse(
            success=True,
            predictions=pred_values,
            actual_values=actual_values,
            errors=errors,
            game_dates=game_dates,
            matchups=matchups
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error making predictions: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error making predictions: {str(e)}"
        )


@app.get("/models")
async def list_models():
    """
    List all available trained models.
    
    Returns:
        List of model files with metadata
    """
    try:
        models = []
        for model_file in MODEL_DIR.glob("*.pkl"):
            stat = model_file.stat()
            models.append({
                "filename": model_file.name,
                "path": str(model_file),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        return {
            "count": len(models),
            "models": sorted(models, key=lambda x: x['modified'], reverse=True)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing models: {str(e)}"
        )


@app.delete("/models/{model_filename}")
async def delete_model(model_filename: str):
    """
    Delete a trained model.
    
    Args:
        model_filename: Name of the model file to delete
        
    Returns:
        Success message
    """
    try:
        model_path = MODEL_DIR / model_filename
        
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_filename} not found"
            )
        
        model_path.unlink()
        
        return {
            "success": True,
            "message": f"Model {model_filename} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting model: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

