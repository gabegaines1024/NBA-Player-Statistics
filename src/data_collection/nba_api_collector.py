
from nba_api.stats.endpoints import teamgamelogs, playergamelogs
from nba_api.stats.static import players, teams
import pandas as pd
import numpy as np
import json
def _get_player_id(player_name: str) -> int:
    """
    Helper function to lookup player ID from player name using static players module.
    
    Args:
        player_name: Full name of the player (e.g., "LeBron James")
        
    Returns:
        int: Player ID if found, None otherwise
    """
    try:
        player_list = players.find_players_by_full_name(player_name)
        if player_list:
            return player_list[0]['id']
        else:
            print(f"Player '{player_name}' not found")
            return None
    except Exception as e:
        print(f"Error looking up player '{player_name}': {e}")
        return None


def _get_team_id(team_name_or_abbrev: str) -> int:
    """
    Helper function to lookup team ID from team name or abbreviation using static teams module.
    
    Args:
        team_name_or_abbrev: Team name (e.g., "Los Angeles Lakers") or abbreviation (e.g., "LAL")
        
    Returns:
        int: Team ID if found, None otherwise
    """
    try:
        # Try abbreviation first (shorter, more common)
        team_list = teams.find_teams_by_abbreviation(team_name_or_abbrev)
        if team_list:
            return team_list[0]['id']
        
        # If not found by abbreviation, try full name
        team_list = teams.find_teams_by_full_name(team_name_or_abbrev)
        if team_list:
            return team_list[0]['id']
        
        print(f"Team '{team_name_or_abbrev}' not found")
        return None
    except Exception as e:
        print(f"Error looking up team '{team_name_or_abbrev}': {e}")
        return None


def get_player_game_logs(player_name: str, season_type: str = 'Regular Season', season_year: int = 2024) -> pd.DataFrame:
    """
    Get game logs for a player in a given season.
    
    Args:
        player_name: Full name of the player (e.g., "LeBron James")
        season_type: Type of season - 'Regular Season' or 'Playoffs'
        season_year: Season year (e.g., 2024 for 2024-2025 season)
        
    Returns:
        pd.DataFrame: Game logs DataFrame if successful, None otherwise
    """
    try:
        # Lookup player ID from name
        player_id = _get_player_id(player_name)
        if player_id is None:
            return None
        
        # Make API call using endpoint
        game_logs = playergamelogs.PlayerGameLogs(
            player_id=player_id, 
            season_type=season_type, 
            season_year=season_year
        ).get_data_frames()[0]
        
        print(f"Game logs collected successfully for {player_name}")
        return game_logs
        
    except Exception as e:
        print(f"Error getting game logs for {player_name}: {e}")
        return None


def get_team_stats(team_name_or_abbrev: str, season_type: str = 'Regular Season', season_year: int = 2024) -> pd.DataFrame:
    """
    Get team stats for a team in a given season.
    
    Args:
        team_name_or_abbrev: Team name (e.g., "Los Angeles Lakers") or abbreviation (e.g., "LAL")
        season_type: Type of season - 'Regular Season' or 'Playoffs'
        season_year: Season year (e.g., 2024 for 2024-2025 season)
        
    Returns:
        pd.DataFrame: Team stats DataFrame if successful, None otherwise
    """
    try:
        # Lookup team ID from name/abbreviation
        team_id = _get_team_id(team_name_or_abbrev)
        if team_id is None:
            return None
        
        # Make API call using endpoint
        team_stats = teamgamelogs.TeamGameLogs(
            team_id=team_id, 
            season_type=season_type, 
            season_year=season_year
        ).get_data_frames()[0]
        
        print(f"Team stats collected successfully for {team_name_or_abbrev}")
        return team_stats
        
    except Exception as e:
        print(f"Error getting team stats for {team_name_or_abbrev}: {e}")
        return None


def get_player_info(player_name: str) -> dict:
    """
    Get player biographical information from static data.
    
    Args:
        player_name: Full name of the player (e.g., "LeBron James")
        
    Returns:
        dict: Player information dictionary if found, None otherwise
    """
    try:
        player_list = players.find_players_by_full_name(player_name)
        if player_list:
            player_info = player_list[0]
            print(f"Player biographical info collected successfully for {player_name}")
            return player_info
        else:
            print(f"Player '{player_name}' not found")
            return None
    except Exception as e:
        print(f"Error getting player info for {player_name}: {e}")
        return None


def get_team_info(team_name_or_abbrev: str) -> dict:
    """
    Get team information from static data.
    
    Args:
        team_name_or_abbrev: Team name (e.g., "Los Angeles Lakers") or abbreviation (e.g., "LAL")
        
    Returns:
        dict: Team information dictionary if found, None otherwise
    """
    try:
        # Try abbreviation first
        team_list = teams.find_teams_by_abbreviation(team_name_or_abbrev)
        if not team_list:
            # Try full name
            team_list = teams.find_teams_by_full_name(team_name_or_abbrev)
        
        if team_list:
            team_info = team_list[0]
            print(f"Team info collected successfully for {team_name_or_abbrev}")
            return team_info
        else:
            print(f"Team '{team_name_or_abbrev}' not found")
            return None
    except Exception as e:
        print(f"Error getting team info for {team_name_or_abbrev}: {e}")
        return None

def get_opponent_stats(team_id: int, date: str) -> dict:
    """
    Get opponent defensive rating for a given data.
    """
    try:
        opponent_stats = teamgamelogs.TeamGameLogs(team_id=team_id, date=date).get_data_frames()[0]['DEF_RATING']
        return opponent_stats
    except Exception as e:
        print(f"Error getting opponent stats for {team_id} on {date}: {e}")
        return None
    
def get_injury_report(date: str) -> dict:
    """
    Get injury report for a given date.
    """
    try:
        injury_report = teamgamelogs.InjuryReport(date=date).get_data_frames()[0]['INJURY_REPORT']
        return injury_report
    except Exception as e:
        print(f"Error getting injury report for {date}: {e}")
        return None

def save_raw_data(data: dict) -> str:
    """
    Save raw data to a JSON file.
    """
    file_name = f"/Users/gabegaines/Desktop/python projects/NBA-Player-Statistics/data/raw/raw.json"
    try:
        with open(file_name, 'a') as f:
            json.dump(data, f)
        return f"Data saved to {file_name}"
    except Exception as e:
        print(f"Error saving data to {file_name}: {e}")
        return None