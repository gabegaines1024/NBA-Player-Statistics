# NBA API Issue & Python Version - Complete Fix Guide

## Issue 1: NBA API Parameter Error
**Error:** `PlayerGameLogs.__init__() got an unexpected keyword argument 'player_id'`

**Fixed:** Updated `src/data_collection/nba_api_collector.py` to use the correct NBA API parameters.

## Issue 2: Dependencies Not Installed
**Error:** `ModuleNotFoundError: No module named 'pandas'`

**Solution:** Install dependencies in your virtual environment.

## Steps to Fix

### 1. Install Dependencies

You have a `.venv` activated, so install the packages:

```bash
# Option A: Try minimal requirements (recommended for Python 3.14)
pip install -r requirements-minimal.txt
```

Or if that file doesn't exist:

```bash
# Option B: Install core packages manually
pip install nba-api pandas numpy scikit-learn fastapi uvicorn pydantic joblib requests
```

Or:

```bash
# Option C: Try the main requirements (may have issues with Python 3.14)
pip install -r requirements.txt
```

### 2. Test the Fix

Once packages are installed:

```bash
python3 main.py
```

### 3. If You Still Have Issues with Python 3.14

Python 3.14.0 is very new. If you continue to have problems:

```bash
# Install Python 3.12 via Homebrew
brew install python@3.12

# Create new venv with Python 3.12
deactivate
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## What Was Fixed in the Code

**File:** `src/data_collection/nba_api_collector.py`

**Before (BROKEN):**
```python
game_logs = playergamelogs.PlayerGameLogs(
    player_id=player_id,  # ❌ This parameter doesn't exist
    season_type=season_type,
    season_year=season_year
).get_data_frames()[0]
```

**After (FIXED):**
```python
# Format season string (e.g., "2023-24")
season_str = f"{season_year-1}-{str(season_year)[-2:]}"

# Get all game logs for the season
game_logs = playergamelogs.PlayerGameLogs(
    season_nullable=season_str,  # ✅ Correct parameter
    season_type_nullable=season_type  # ✅ Correct parameter
).get_data_frames()[0]

# Filter for specific player
game_logs = game_logs[game_logs['PLAYER_ID'] == player_id]
```

## Quick Start

After installing dependencies:

```bash
# Test the ML model with synthetic data (no NBA API needed)
python3 test_model.py

# Run the full pipeline with real NBA data
python3 main.py

# Start the API server
python3 api.py
```

## Summary

✅ **Fixed:** NBA API parameter issue
⚠️ **Action needed:** Install dependencies (see step 1 above)
⚠️ **Recommended:** Use Python 3.11 or 3.12 instead of 3.14

