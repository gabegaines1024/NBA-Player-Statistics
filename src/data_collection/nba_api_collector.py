
from nba_api.stats.endpoints import teamgamelogs, playergamelogs
from nba_api.stats.static import players, teams
import pandas as pd
import numpy as np
import json
import time

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
        # Get all teams
        all_teams = teams.get_teams()
        
        # Search by abbreviation or full name
        for team in all_teams:
            if (team['abbreviation'].upper() == team_name_or_abbrev.upper() or 
                team['full_name'].upper() == team_name_or_abbrev.upper() or
                team['nickname'].upper() == team_name_or_abbrev.upper()):
                return team['id']
        
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
        time.sleep(1)
        game_logs = playergamelogs.PlayerGameLogs(
            player_id=player_id, 
            season_type=season_type, 
            season_year=season_year
        ).get_data_frames()[0]

        if game_logs is not None and not game_logs.empty:
            return game_logs
        else:
            print(f"Game logs not found for {player_name}")
            return None

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
        time.sleep(1)
        team_stats = teamgamelogs.TeamGameLogs(
            team_id=team_id, 
            season_type=season_type, 
            season_year=season_year
        ).get_data_frames()[0]
        
        if team_stats is not None and not team_stats.empty:
            return team_stats
        else:
            print(f"Team stats not found for {team_name_or_abbrev}")
            return None
        
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
        player_info = players.find_players_by_full_name(player_name)
        if player_info:
            return player_info[0] if isinstance(player_info, list) else player_info
        else:
            print(f"Player info not found for {player_name}")
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
        # Get all teams
        all_teams = teams.get_teams()
        
        # Search by abbreviation or full name
        for team in all_teams:
            if (team['abbreviation'].upper() == team_name_or_abbrev.upper() or 
                team['full_name'].upper() == team_name_or_abbrev.upper() or
                team['nickname'].upper() == team_name_or_abbrev.upper()):
                print(f"Team info collected successfully for {team_name_or_abbrev}")
                return team
        
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
        time.sleep(1)
        opponent_stats = teamgamelogs.TeamGameLogs(team_id=team_id, date=date).get_data_frames()[0]['DEF_RATING']
        if opponent_stats is not None and not opponent_stats.empty:
            return opponent_stats
        else:
            print(f"Opponent stats not found for {team_id} on {date}")
            return None
    except Exception as e:
        print(f"Error getting opponent stats for {team_id} on {date}: {e}")
        return None
    
def get_injury_report(date: str) -> dict:
    """
    Get injury report for a given date.
    """
    try:
        time.sleep(1)
        injury_report = teamgamelogs.InjuryReport(date=date).get_data_frames()[0]['INJURY_REPORT']
        if injury_report is not None and not injury_report.empty:
            return injury_report
        else:
            print(f"Injury report not found for {date}")
            return None
    except Exception as e:
        print(f"Error getting injury report for {date}: {e}")
        return None

def save_raw_data(data: dict) -> str:
    """
    Save raw data to a JSON file.
    """
    time.sleep(1)
    file_name = f"/Users/gabegaines/Desktop/python projects/NBA-Player-Statistics/data/raw/raw.json"
    try:
        with open(file_name, 'a') as f:
            json.dump(data, f)
        return f"Data saved to {file_name}"
    except Exception as e:
        print(f"Error saving data to {file_name}: {e}")
        return None