# NBA-API Module Imports Guide

## Understanding `nba_api.stats.static`

The `nba_api.stats.static` module contains **helper functions** and **data dictionaries**, NOT endpoint classes. Here's what's actually available:

### 1. `nba_api.stats.static.players` Module

This module provides functions to get player information:

```python
from nba_api.stats.static import players

# Get all players (returns a list of dictionaries)
all_players = players.get_players()

# Get a specific player by full name
player = players.find_players_by_full_name("LeBron James")
# Returns: [{'id': 2544, 'full_name': 'LeBron James', ...}]

# Get player by ID
player = players.find_player_by_id(2544)
# Returns: {'id': 2544, 'full_name': 'LeBron James', ...}

# Check if player is active
is_active = players.find_players_by_full_name("LeBron James")[0].get('is_active', False)
```

**Available Functions:**
- `get_players()` - Returns all players (list of dicts)
- `find_players_by_full_name(name)` - Find players by name (returns list)
- `find_players_by_first_name(name)` - Find by first name
- `find_players_by_last_name(name)` - Find by last name
- `find_player_by_id(player_id)` - Find single player by ID
- `get_active_players()` - Get only active players
- `get_inactive_players()` - Get inactive players

### 2. `nba_api.stats.static.teams` Module

This module provides functions to get team information:

```python
from nba_api.stats.static import teams

# Get all teams (returns a list of dictionaries)
all_teams = teams.get_teams()

# Find team by full name
team = teams.find_teams_by_full_name("Los Angeles Lakers")
# Returns: [{'id': 1610612747, 'full_name': 'Los Angeles Lakers', ...}]

# Find team by abbreviation
team = teams.find_teams_by_abbreviation("LAL")
# Returns: [{'id': 1610612747, 'abbreviation': 'LAL', ...}]

# Get team by ID
team = teams.find_team_by_id(1610612747)
# Returns: {'id': 1610612747, 'full_name': 'Los Angeles Lakers', ...}
```

**Available Functions:**
- `get_teams()` - Returns all teams (list of dicts)
- `find_teams_by_full_name(name)` - Find teams by full name
- `find_teams_by_city(city)` - Find teams by city
- `find_teams_by_state(state)` - Find teams by state
- `find_teams_by_abbreviation(abbrev)` - Find by abbreviation
- `find_team_by_id(team_id)` - Find single team by ID

### 3. `playerstats` - DOES NOT EXIST IN STATIC

**There is NO `playerstats` module in `nba_api.stats.static`**. 

Player statistics come from **endpoints**, not static data:

```python
from nba_api.stats.endpoints import playergamelogs, playerdashboardbygeneralsplits

# For game logs (what you're already doing)
from nba_api.stats.endpoints import playergamelogs
game_logs = playergamelogs.PlayerGameLogs(player_id=2544, season='2024-25')

# For detailed player stats
from nba_api.stats.endpoints import playergamestats
player_stats = playergamestats.PlayerGameStats(player_id=2544, season='2024-25')
```

## Common Endpoints for Player Stats

```python
from nba_api.stats.endpoints import (
    playergamelogs,           # Individual game logs
    playerdashboardbygeneralsplits,  # Player dashboard stats
    playergamestats,          # Detailed game stats
    playerestimatedmetrics,   # Estimated metrics
    playerdashboardbyclutch,  # Clutch performance
    # ... and many more
)
```

## How to Fix Your Code

Based on your current code, here are the corrections needed:

### ❌ INCORRECT (What you have now):
```python
from nba_api.stats.static import players, playerstats, teams

# This won't work - players is a module with functions, not a class
player_info = players.PlayerInfo(player_id=player_id).get_data_frames()[0]

# This won't work - playerstats doesn't exist in static
player_stats = playerstats.PlayerStats(player_id=player_id).get_data_frames()[0]
```

### ✅ CORRECT (What it should be):

```python
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelogs, playerdashboardbygeneralsplits

# To get player biographical info (static data):
player_info = players.find_player_by_id(player_id)
# Returns a dictionary, not a DataFrame

# To get player statistics (use endpoints):
player_stats = playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits(
    player_id=player_id,
    season='2024-25'
).get_data_frames()[0]
# Returns a DataFrame
```

## Summary

- **`nba_api.stats.static`** = Helper functions for looking up players/teams by ID, name, etc. (returns dicts/lists)
- **`nba_api.stats.endpoints`** = Actual statistics classes that return DataFrames
- **`playerstats`** doesn't exist in static - use endpoint classes instead

