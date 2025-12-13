# NBA API Issue - Fixed

## Problem
The error `PlayerGameLogs.__init__() got an unexpected keyword argument 'player_id'` indicates that the NBA API has changed.

The `PlayerGameLogs` endpoint no longer accepts `player_id` directly. Instead, it retrieves all game logs for a season and you need to filter by player.

## Solution
Updated `get_player_game_logs()` in `src/data_collection/nba_api_collector.py`:

**Changes:**
1. Remove `player_id` parameter from API call
2. Use `season_nullable` with proper format (e.g., "2023-24")
3. Use `season_type_nullable` parameter
4. Filter results by `PLAYER_ID` after fetching

**New approach:**
```python
# Format season string
season_str = f"{season_year-1}-{str(season_year)[-2:]}"  # "2023-24"

# Get all game logs for the season
game_logs = playergamelogs.PlayerGameLogs(
    season_nullable=season_str,
    season_type_nullable=season_type
).get_data_frames()[0]

# Filter for specific player
game_logs = game_logs[game_logs['PLAYER_ID'] == player_id]
```

## Test it
```bash
python3 main.py
```

The pipeline should now work correctly with the NBA API.

