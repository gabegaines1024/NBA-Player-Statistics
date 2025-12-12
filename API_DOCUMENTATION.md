# NBA Player Statistics ML API Documentation

## Overview

FastAPI-based REST API for training machine learning models and making predictions on NBA player performance.

## Features

- ✅ Train ML models on NBA player data
- ✅ Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
- ✅ Make predictions using trained models
- ✅ List and manage saved models
- ✅ Pydantic schema validation
- ✅ CORS support for frontend integration

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the API server:
```bash
python3 api.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Root Endpoint
**GET** `/`

Get API information and available endpoints.

**Response:**
```json
{
  "message": "NBA Player Statistics ML API",
  "version": "1.0.0",
  "endpoints": { ... }
}
```

### 2. Health Check
**GET** `/health`

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-12-12T10:30:00"
}
```

### 3. Train Model
**POST** `/train`

Train a machine learning model on NBA player data.

**Request Body:**
```json
{
  "player_name": "LeBron James",
  "target_column": "PTS",
  "model_type": "random_forest",
  "season_type": "Regular Season",
  "season_year": 2024,
  "min_minutes": 10,
  "test_size": 0.2,
  "compare_models": false
}
```

**Parameters:**
- `player_name` (string, required): NBA player name
- `target_column` (string, default: "PTS"): Stat to predict (PTS, AST, REB, etc.)
- `model_type` (string, default: "random_forest"): Model type
  - Options: `random_forest`, `gradient_boosting`, `linear`, `ridge`
- `season_type` (string, default: "Regular Season"): Season type
- `season_year` (int, default: 2024): Season year
- `min_minutes` (int, default: 10): Minimum minutes filter
- `test_size` (float, default: 0.2): Test set proportion
- `compare_models` (bool, default: false): Compare multiple models

**Response:**
```json
{
  "success": true,
  "model_type": "random_forest",
  "target_column": "PTS",
  "metrics": {
    "train_mae": 4.23,
    "train_rmse": 5.67,
    "train_r2": 0.856,
    "test_mae": 5.12,
    "test_rmse": 6.89,
    "test_r2": 0.823,
    "cv_mae_mean": 5.45,
    "cv_mae_std": 0.67,
    "n_features": 45,
    "n_train_samples": 48,
    "n_test_samples": 12
  },
  "model_path": "/path/to/model.pkl",
  "feature_importance": [
    {"feature": "PTS_rolling_5", "importance": 0.234},
    {"feature": "MIN", "importance": 0.156}
  ]
}
```

### 4. Train Model with Hyperparameter Tuning
**POST** `/train/tuning?n_iter=10`

Train a model with automatic hyperparameter tuning.

**Query Parameters:**
- `n_iter` (int, optional): Number of iterations for RandomizedSearchCV. If not provided, uses GridSearchCV (slower but exhaustive).

**Request Body:** Same as `/train`

**Response:** Same as `/train`, but with optimized hyperparameters

**Note:** This endpoint can take significantly longer (5-30 minutes depending on data size and n_iter).

### 5. Make Predictions
**POST** `/predict`

Make predictions using a trained model.

**Request Body:**
```json
{
  "model_path": "/path/to/model.pkl",
  "player_name": "LeBron James",
  "target_column": "PTS",
  "n_predictions": 5
}
```

**Parameters:**
- `model_path` (string, required): Path to trained model file
- `player_name` (string, optional): Player name to get fresh data
- `game_data` (object, optional): Custom game data for prediction
- `target_column` (string, default: "PTS"): Target column
- `n_predictions` (int, default: 5): Number of predictions to return

**Response:**
```json
{
  "success": true,
  "predictions": [25.3, 28.1, 23.7, 26.4, 27.8],
  "actual_values": [24.0, 29.0, 22.0, 27.0, 28.0],
  "errors": [1.3, 0.9, 1.7, 0.6, 0.2],
  "game_dates": ["2024-01-15", "2024-01-17", ...],
  "matchups": ["LAL vs. GSW", "LAL @ BOS", ...]
}
```

### 6. List Models
**GET** `/models`

List all trained models.

**Response:**
```json
{
  "count": 3,
  "models": [
    {
      "filename": "random_forest_PTS_20241212_103000.pkl",
      "path": "/full/path/to/model.pkl",
      "size_mb": 2.45,
      "created": "2024-12-12T10:30:00",
      "modified": "2024-12-12T10:30:00"
    }
  ]
}
```

### 7. Delete Model
**DELETE** `/models/{model_filename}`

Delete a trained model.

**Response:**
```json
{
  "success": true,
  "message": "Model deleted successfully"
}
```

## Usage Examples

### Python (requests)

```python
import requests

# Train a model
response = requests.post(
    "http://localhost:8000/train",
    json={
        "player_name": "Stephen Curry",
        "target_column": "PTS",
        "model_type": "random_forest"
    }
)
result = response.json()
model_path = result["model_path"]

# Make predictions
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "model_path": model_path,
        "player_name": "Stephen Curry",
        "n_predictions": 5
    }
)
predictions = response.json()
```

### cURL

```bash
# Train a model
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "player_name": "LeBron James",
    "target_column": "PTS",
    "model_type": "random_forest"
  }'

# List models
curl "http://localhost:8000/models"

# Make predictions
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/path/to/model.pkl",
    "player_name": "LeBron James",
    "n_predictions": 5
  }'
```

### JavaScript (fetch)

```javascript
// Train a model
const response = await fetch('http://localhost:8000/train', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    player_name: 'Kevin Durant',
    target_column: 'PTS',
    model_type: 'gradient_boosting'
  })
});
const result = await response.json();

// Make predictions
const predResponse = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model_path: result.model_path,
    player_name: 'Kevin Durant',
    n_predictions: 5
  })
});
const predictions = await predResponse.json();
```

## Testing

Run the test suite:

```bash
# Start the API server in one terminal
python3 api.py

# Run tests in another terminal
python3 test_api.py
```

## Interactive Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These interfaces allow you to:
- View all endpoints
- See request/response schemas
- Test endpoints directly in the browser

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid parameters)
- `404`: Resource not found (player, model, etc.)
- `500`: Internal server error

Error responses include a `detail` field with error information:

```json
{
  "detail": "No data found for player John Doe"
}
```

## Performance Notes

- **Training**: 30 seconds - 5 minutes depending on data size
- **Training with tuning**: 5-30 minutes depending on n_iter and data size
- **Predictions**: < 1 second
- **Data collection**: 1-5 seconds (NBA API rate limits apply)

## Production Deployment

For production use:

1. Update CORS origins in `api.py`:
```python
allow_origins=["https://yourdomain.com"]
```

2. Use a production ASGI server:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

3. Add authentication/authorization
4. Use a proper database for model metadata
5. Implement request rate limiting
6. Add monitoring and logging

## Model Types

### Random Forest
- Best for: General purpose, feature importance
- Pros: Robust, handles non-linear relationships
- Cons: Can be slow with large datasets

### Gradient Boosting
- Best for: High accuracy predictions
- Pros: Often highest accuracy
- Cons: Slower training, prone to overfitting

### Linear Regression
- Best for: Simple, interpretable models
- Pros: Fast, interpretable
- Cons: Assumes linear relationships

### Ridge Regression
- Best for: Linear models with regularization
- Pros: Prevents overfitting, fast
- Cons: Assumes linear relationships

## Hyperparameter Tuning

The `/train/tuning` endpoint automatically tunes:

**Random Forest:**
- n_estimators: [50, 100, 200]
- max_depth: [5, 10, 15, 20, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]
- max_features: ['sqrt', 'log2', None]

**Gradient Boosting:**
- n_estimators: [50, 100, 200]
- max_depth: [3, 5, 7]
- learning_rate: [0.01, 0.05, 0.1, 0.2]
- subsample: [0.8, 0.9, 1.0]
- min_samples_split: [2, 5, 10]

**Ridge:**
- alpha: [0.01, 0.1, 1.0, 10.0, 100.0]
- solver: ['auto', 'svd', 'cholesky', 'lsqr']

Use `n_iter` parameter to limit search iterations for faster results.

