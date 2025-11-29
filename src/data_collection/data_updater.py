from nba_api.stats.endpoints import teamgamelogs, playergamelogs
from nba_api.stats.static import players, teams
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
from .nba_api_collector import get_player_game_logs, _get_player_id

# Define paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_FILE = RAW_DIR / "update_metadata.json"
CURRENT_SEASON_YEAR = 2024
CURRENT_SEASON_TYPE = 'Regular Season'


def _load_metadata():
    """Load metadata about last updates."""
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_metadata(metadata):
    """Save metadata about last updates."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        return True
    except IOError as e:
        print(f"Error saving metadata: {e}")
        return False


def check_for_new_games():
    """
    Check if new games have been played since last data collection.
    
    Returns:
        bool: True if new games found, False otherwise
    """
    try:
        metadata = _load_metadata()
        last_update_str = metadata.get('last_update_date', None)
        
        if not last_update_str:
            # No previous update, assume there might be new games
            print("No previous update found. New games may be available.")
            return True
        
        # Parse last update date
        try:
            last_update = datetime.fromisoformat(last_update_str)
        except (ValueError, AttributeError):
            # If date format is different, assume new games available
            return True
        
        # Check if it's been more than a day since last update
        # (games typically happen daily during season)
        days_since_update = (datetime.now() - last_update).days
        
        # During NBA season, check if it's been at least 1 day
        # (adjust logic based on your needs)
        if days_since_update >= 1:
            print(f"Last update was {days_since_update} days ago. New games may be available.")
            return True
        else:
            print(f"Last update was {days_since_update} days ago. No new games likely.")
            return False
            
    except Exception as e:
        print(f"Error checking for new games: {e}")
        # On error, assume new games might be available
        return True


def update_player_data(player_id: int, season_type: str = CURRENT_SEASON_TYPE, 
                       season_year: int = CURRENT_SEASON_YEAR) -> bool:
    """
    Add new game data for a player.
    
    Args:
        player_id: Player ID from NBA API
        season_type: Type of season - 'Regular Season' or 'Playoffs'
        season_year: Season year (e.g., 2024 for 2024-2025 season)
    
    Returns:
        bool: True if update successful, False otherwise
    """
    try:
        # Get player name for file naming
        player_info = players.find_player_by_id(player_id)
        if not player_info:
            print(f"Player with ID {player_id} not found")
            return False
        
        player_name = player_info.get('full_name', f"player_{player_id}")
        safe_name = player_name.replace(' ', '_').replace('/', '_')
        
        # Get game logs
        game_logs = playergamelogs.PlayerGameLogs(
            player_id=player_id,
            season_type=season_type,
            season_year=season_year
        ).get_data_frames()[0]
        
        if game_logs is None or game_logs.empty:
            print(f"No game logs found for player {player_id}")
            return False
        
        # Create data directory if it doesn't exist
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        
        # File path for this player's data
        player_file = RAW_DIR / f"player_{player_id}_{safe_name}.json"
        
        # Load existing data if it exists
        existing_data = []
        if player_file.exists():
            try:
                with open(player_file, 'r') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing_data = []
        
        # Convert new game logs to records
        new_records = game_logs.to_dict('records')
        
        # Get existing game IDs to avoid duplicates
        existing_game_ids = set()
        if existing_data:
            for record in existing_data:
                game_id = record.get('GAME_ID')
                if game_id:
                    existing_game_ids.add(str(game_id))
        
        # Add only new games
        updated = False
        for record in new_records:
            game_id = str(record.get('GAME_ID', ''))
            if game_id and game_id not in existing_game_ids:
                existing_data.append(record)
                existing_game_ids.add(game_id)
                updated = True
        
        # Save updated data
        if updated or not existing_data:
            with open(player_file, 'w') as f:
                json.dump(existing_data, f, indent=2, default=str)
            print(f"Updated data for {player_name} (ID: {player_id})")
            return True
        else:
            print(f"No new games for {player_name} (ID: {player_id})")
            return False
            
    except Exception as e:
        print(f"Error updating player data for player_id {player_id}: {e}")
        return False


def update_all_players(season_type: str = CURRENT_SEASON_TYPE, 
                      season_year: int = CURRENT_SEASON_YEAR,
                      active_only: bool = True) -> dict:
    """
    Update the data for all players.
    
    Args:
        season_type: Type of season - 'Regular Season' or 'Playoffs'
        season_year: Season year (e.g., 2024 for 2024-2025 season)
        active_only: If True, only update active players
    
    Returns:
        dict: Summary of update results with counts of successful/failed updates
    """
    try:
        # Get all players
        if active_only:
            all_players = players.get_active_players()
        else:
            all_players = players.get_players()
        
        if not all_players:
            print("No players found")
            return {"success": 0, "failed": 0, "total": 0}
        
        total_players = len(all_players)
        success_count = 0
        failed_count = 0
        
        print(f"Updating data for {total_players} players...")
        
        for i, player in enumerate(all_players, 1):
            player_id = player.get('id')
            player_name = player.get('full_name', 'Unknown')
            
            if not player_id:
                failed_count += 1
                continue
            
            print(f"[{i}/{total_players}] Processing {player_name}...")
            
            if update_player_data(player_id, season_type, season_year):
                success_count += 1
            else:
                failed_count += 1
        
        # Update metadata with last update time
        metadata = _load_metadata()
        metadata['last_update_date'] = datetime.now().isoformat()
        metadata['last_update_summary'] = {
            'total_players': total_players,
            'success': success_count,
            'failed': failed_count
        }
        _save_metadata(metadata)
        
        result = {
            "success": success_count,
            "failed": failed_count,
            "total": total_players
        }
        
        print(f"\nUpdate complete: {success_count} successful, {failed_count} failed out of {total_players} total")
        return result
        
    except Exception as e:
        print(f"Error updating all players: {e}")
        return {"success": 0, "failed": 0, "total": 0, "error": str(e)}


def validate_data_completeness() -> dict:
    """
    Validate the completeness of the data.
    
    Returns:
        dict: Validation results with details about data completeness
    """
    try:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        
        validation_results = {
            "data_directory_exists": RAW_DIR.exists(),
            "metadata_file_exists": METADATA_FILE.exists(),
            "player_files_count": 0,
            "total_records": 0,
            "players_with_data": 0,
            "players_without_data": 0,
            "issues": []
        }
        
        if not RAW_DIR.exists():
            validation_results["issues"].append("Data directory does not exist")
            return validation_results
        
        # Check metadata
        if METADATA_FILE.exists():
            try:
                metadata = _load_metadata()
                last_update = metadata.get('last_update_date', 'Unknown')
                validation_results["last_update_date"] = last_update
            except Exception as e:
                validation_results["issues"].append(f"Error reading metadata: {e}")
        
        # Count player files
        player_files = list(RAW_DIR.glob("player_*.json"))
        validation_results["player_files_count"] = len(player_files)
        
        # Validate each player file
        for player_file in player_files:
            try:
                with open(player_file, 'r') as f:
                    data = json.load(f)
                    
                if isinstance(data, list) and len(data) > 0:
                    validation_results["players_with_data"] += 1
                    validation_results["total_records"] += len(data)
                    
                    # Check if records have expected fields
                    sample_record = data[0]
                    required_fields = ['GAME_ID', 'GAME_DATE', 'MATCHUP', 'PTS']
                    missing_fields = [field for field in required_fields if field not in sample_record]
                    
                    if missing_fields:
                        validation_results["issues"].append(
                            f"{player_file.name} missing fields: {', '.join(missing_fields)}"
                        )
                else:
                    validation_results["players_without_data"] += 1
                    validation_results["issues"].append(f"{player_file.name} is empty or invalid")
                    
            except (json.JSONDecodeError, IOError) as e:
                validation_results["players_without_data"] += 1
                validation_results["issues"].append(f"Error reading {player_file.name}: {e}")
        
        # Overall validation status
        validation_results["is_valid"] = (
            validation_results["data_directory_exists"] and
            validation_results["player_files_count"] > 0 and
            len(validation_results["issues"]) == 0
        )
        
        return validation_results
        
    except Exception as e:
        return {
            "is_valid": False,
            "error": str(e),
            "issues": [f"Validation error: {e}"]
        }