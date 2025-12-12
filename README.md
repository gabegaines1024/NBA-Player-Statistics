# NBA Player Statistics ML Pipeline

Machine learning pipeline for predicting NBA player performance with FastAPI REST API.

## Features

- 🏀 **Data Collection**: Automatic NBA API data collection
- 🧹 **Data Preprocessing**: Cleaning, outlier detection, feature engineering
- 🤖 **Machine Learning**: Multiple model types (Random Forest, Gradient Boosting, Linear, Ridge)
- ⚙️ **Hyperparameter Tuning**: Automatic optimization with GridSearchCV/RandomizedSearchCV
- 🚀 **REST API**: FastAPI endpoints for training and predictions
- ✅ **Data Validation**: Pydantic schemas for type safety
- 📊 **Feature Engineering**: Rolling stats, efficiency metrics, time-based features

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Option A: Use the Command Line Pipeline
```bash
python3 main.py
```

### 2. Option B: Use the REST API
```bash
# Start the API server
python3 api.py

# In another terminal, test it
python3 test_api.py
```

## Usage

### Command Line

```bash
# Train and predict (edit main.py to configure)
python3 main.py

# Test the ML model
python3 test_model.py

# Test the full pipeline
python3 test_full_pipeline.py
```

### REST API

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

# Make predictions
predictions = requests.post(
    "http://localhost:8000/predict",
    json={
        "model_path": response.json()["model_path"],
        "player_name": "Stephen Curry",
        "n_predictions": 5
    }
)
```

See [QUICKSTART_API.md](QUICKSTART_API.md) for detailed API usage.

## Documentation

- [API Documentation](API_DOCUMENTATION.md) - Complete API reference
- [Quick Start Guide](QUICKSTART_API.md) - Get started in 5 minutes
- [Implementation Status](STATUS.md) - Current project status
- [Interactive API Docs](http://localhost:8000/docs) - Swagger UI (when server is running)

## Project Structure

```
NBA-Player-Statistics/
├── api.py                      # FastAPI application
├── main.py                     # Command-line pipeline
├── test_api.py                 # API test suite
├── test_model.py               # ML model tests
├── test_full_pipeline.py       # End-to-end tests
├── src/
│   ├── data_collection/        # NBA API data collection
│   ├── preprossessing/         # Data cleaning & feature engineering
│   ├── models/                 # ML models with hyperparameter tuning
│   └── schemas/                # Pydantic validation schemas
├── data/                       # Data storage
├── models/                     # Trained model storage
└── requirements.txt            # Python dependencies
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/train` | POST | Train a model |
| `/train/tuning` | POST | Train with hyperparameter tuning |
| `/predict` | POST | Make predictions |
| `/models` | GET | List saved models |
| `/health` | GET | Health check |

## Model Types

- **Random Forest**: Best all-around, good accuracy and speed
- **Gradient Boosting**: Highest accuracy, slower training
- **Linear Regression**: Fast, simple relationships
- **Ridge Regression**: Linear with regularization

## Hyperparameter Tuning

Automatically tunes model hyperparameters:

```bash
curl -X POST "http://localhost:8000/train/tuning?n_iter=10" \
  -H "Content-Type: application/json" \
  -d '{"player_name": "LeBron James", "model_type": "random_forest"}'
```

## Examples

### Predict Points
```python
requests.post("http://localhost:8000/train", json={
    "player_name": "Kevin Durant",
    "target_column": "PTS"
})
```

### Predict Assists
```python
requests.post("http://localhost:8000/train", json={
    "player_name": "Chris Paul",
    "target_column": "AST"
})
```

### Compare Models
```python
requests.post("http://localhost:8000/train", json={
    "player_name": "Giannis Antetokounmpo",
    "compare_models": true
})
```

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- nba-api
- fastapi, uvicorn
- pydantic

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
