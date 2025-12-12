# Quick Start Guide - FastAPI

Get started with the NBA Player Statistics ML API in 5 minutes.

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Start the API Server

```bash
python3 api.py
```

The server will start at `http://localhost:8000`

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## 3. View Interactive Documentation

Open your browser and go to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive API documentation where you can test endpoints directly.

## 4. Test the API

### Option A: Use the Test Script

In a new terminal:
```bash
python3 test_api.py
```

### Option B: Use cURL

```bash
# Health check
curl http://localhost:8000/health

# Train a model
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "player_name": "Stephen Curry",
    "target_column": "PTS",
    "model_type": "random_forest"
  }'

# List models
curl http://localhost:8000/models
```

### Option C: Use Python

```python
import requests

# Train a model
response = requests.post(
    "http://localhost:8000/train",
    json={
        "player_name": "LeBron James",
        "target_column": "PTS",
        "model_type": "random_forest"
    }
)
result = response.json()
print(f"Model trained! Test MAE: {result['metrics']['test_mae']:.2f}")

# Make predictions
model_path = result["model_path"]
pred_response = requests.post(
    "http://localhost:8000/predict",
    json={
        "model_path": model_path,
        "player_name": "LeBron James",
        "n_predictions": 5
    }
)
predictions = pred_response.json()
print(f"Predictions: {predictions['predictions']}")
```

## 5. Common Use Cases

### Train with Hyperparameter Tuning

```bash
curl -X POST "http://localhost:8000/train/tuning?n_iter=10" \
  -H "Content-Type: application/json" \
  -d '{
    "player_name": "Kevin Durant",
    "target_column": "PTS",
    "model_type": "gradient_boosting"
  }'
```

**Note:** This can take 5-15 minutes. Use `n_iter=10` for faster results.

### Compare Multiple Models

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "player_name": "Giannis Antetokounmpo",
    "target_column": "PTS",
    "compare_models": true
  }'
```

### Predict Different Stats

```bash
# Predict assists
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "player_name": "Chris Paul",
    "target_column": "AST",
    "model_type": "random_forest"
  }'

# Predict rebounds
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "player_name": "Nikola Jokic",
    "target_column": "REB",
    "model_type": "gradient_boosting"
  }'
```

## 6. Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/train` | POST | Train model |
| `/train/tuning` | POST | Train with hyperparameter tuning |
| `/predict` | POST | Make predictions |
| `/models` | GET | List saved models |
| `/models/{filename}` | DELETE | Delete a model |

## 7. Model Types

Choose the best model for your use case:

- **`random_forest`** - Best all-around choice, good accuracy and speed
- **`gradient_boosting`** - Highest accuracy, slower training
- **`linear`** - Fastest, simple linear relationships
- **`ridge`** - Linear with regularization, prevents overfitting

## 8. Target Columns

Predict any NBA stat:
- `PTS` - Points
- `AST` - Assists
- `REB` - Rebounds
- `STL` - Steals
- `BLK` - Blocks
- `TOV` - Turnovers
- `FG` - Field goals made
- `FG3M` - Three-pointers made

## Troubleshooting

### API won't start
- Check if port 8000 is already in use
- Install dependencies: `pip install -r requirements.txt`

### Training fails
- Check player name spelling
- Verify season year is valid (2000-2024)
- Ensure player has games in that season

### Predictions fail
- Verify model_path exists
- Check that player_name or game_data is provided
- Ensure model was trained successfully

## Next Steps

- Read full [API Documentation](API_DOCUMENTATION.md)
- Explore interactive docs at http://localhost:8000/docs
- Build a frontend application using the API
- Deploy to production with proper authentication

## Performance Tips

1. **Use caching**: Train models once, reuse for predictions
2. **Batch predictions**: Predict multiple games at once
3. **Use n_iter**: For tuning, start with `n_iter=10` for faster results
4. **Choose right model**: Random Forest is usually the best balance

## Example: Full Workflow

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Train a model
print("Training model...")
train_response = requests.post(
    f"{BASE_URL}/train",
    json={
        "player_name": "Damian Lillard",
        "target_column": "PTS",
        "model_type": "random_forest"
    }
)
result = train_response.json()
print(f"✓ Model trained! Test MAE: {result['metrics']['test_mae']:.2f}")

# 2. View feature importance
if result.get('feature_importance'):
    print("\nTop features:")
    for feat in result['feature_importance'][:5]:
        print(f"  {feat['feature']}: {feat['importance']:.3f}")

# 3. Make predictions
print("\nMaking predictions...")
pred_response = requests.post(
    f"{BASE_URL}/predict",
    json={
        "model_path": result["model_path"],
        "player_name": "Damian Lillard",
        "n_predictions": 5
    }
)
predictions = pred_response.json()

print("\nLast 5 game predictions:")
for i, (pred, actual) in enumerate(zip(
    predictions['predictions'],
    predictions['actual_values']
), 1):
    error = abs(pred - actual)
    print(f"  Game {i}: Predicted {pred:.1f}, Actual {actual:.1f}, Error {error:.1f}")

print("\n✓ Complete!")
```

Happy predicting! 🏀

