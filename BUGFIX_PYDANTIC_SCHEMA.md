# 🐛 Bug Fix #2: Pydantic Schema Validation

## Problem
After fixing the NaN → None conversion, you got a new error:
```
2 validation errors for PredictionResponse
actual_values.4
  Input should be a valid number [type=float_type, input_value=None, input_type=NoneType]
errors.4
  Input should be a valid number [type=float_type, input_value=None, input_type=NoneType]
```

## Root Cause
The Pydantic schema `PredictionResponse` was defined as:
```python
actual_values: Optional[List[float]]  # List of floats (no None allowed inside)
errors: Optional[List[float]]          # List of floats (no None allowed inside)
```

But we're now returning lists that contain `None` values:
```python
actual_values = [27.0, 35.0, 32.0, 21.0, None]  # Last value is None
errors = [3.2, 2.1, 4.5, 1.2, None]              # Last error is None
```

## Solution
Updated the Pydantic schema in `src/schemas/model_schemas.py`:

```python
# Before:
actual_values: Optional[List[float]]
errors: Optional[List[float]]

# After:
actual_values: Optional[List[Optional[float]]]  # List can contain None
errors: Optional[List[Optional[float]]]          # List can contain None
```

This allows `None` values inside the lists, which is needed for:
- Future games that don't have actual values yet
- Prediction errors that can't be calculated without actual values

## Test
**Restart the API:**
1. In Terminal 3: `Ctrl+C` to stop
2. Run: `python3 api.py`

**Then make prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "models/random_forest_PTS_20251212_233603.pkl",
    "player_name": "LeBron James",
    "n_predictions": 5
  }'
```

Should work now! ✅

