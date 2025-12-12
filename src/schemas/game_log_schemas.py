"""
Pydantic schemas for NBA player game log data validation.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime


class PlayerGameLog(BaseModel):
    """
    Schema for raw player game log data from NBA API.
    """
    GAME_ID: str = Field(..., description="Unique game identifier")
    GAME_DATE: str = Field(..., description="Date of the game (YYYY-MM-DD)")
    MATCHUP: str = Field(..., description="Matchup string (e.g., 'LAL vs. GSW')")
    WL: Optional[str] = Field(None, description="Win/Loss indicator")
    MIN: Optional[float] = Field(None, ge=0, le=48, description="Minutes played")
    PTS: Optional[float] = Field(None, ge=0, description="Points scored")
    AST: Optional[float] = Field(None, ge=0, description="Assists")
    REB: Optional[float] = Field(None, ge=0, description="Rebounds")
    STL: Optional[float] = Field(None, ge=0, description="Steals")
    BLK: Optional[float] = Field(None, ge=0, description="Blocks")
    TOV: Optional[float] = Field(None, ge=0, description="Turnovers")
    FG: Optional[float] = Field(None, ge=0, description="Field goals made")
    FGA: Optional[float] = Field(None, ge=0, description="Field goals attempted")
    FG3M: Optional[float] = Field(None, ge=0, description="Three-pointers made")
    FG3A: Optional[float] = Field(None, ge=0, description="Three-pointers attempted")
    FT: Optional[float] = Field(None, ge=0, description="Free throws made")
    FTA: Optional[float] = Field(None, ge=0, description="Free throws attempted")
    OREB: Optional[float] = Field(None, ge=0, description="Offensive rebounds")
    DREB: Optional[float] = Field(None, ge=0, description="Defensive rebounds")
    PLAYER_NAME: Optional[str] = Field(None, description="Player name")
    PLAYER_ID: Optional[int] = Field(None, description="Player ID")
    TEAM_ID: Optional[int] = Field(None, description="Team ID")
    TEAM_ABBREVIATION: Optional[str] = Field(None, description="Team abbreviation")
    TEAM_NAME: Optional[str] = Field(None, description="Team name")
    
    @field_validator('GAME_DATE')
    @classmethod
    def validate_game_date(cls, v: str) -> str:
        """Validate game date format."""
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            try:
                # Try alternative format
                datetime.strptime(v, '%b %d, %Y')
                return v
            except ValueError:
                raise ValueError(f"Invalid date format: {v}. Expected YYYY-MM-DD or 'MMM DD, YYYY'")
    
    @field_validator('WL')
    @classmethod
    def validate_wl(cls, v: Optional[str]) -> Optional[str]:
        """Validate win/loss indicator."""
        if v is None:
            return v
        if v.upper() not in ['W', 'L']:
            raise ValueError(f"WL must be 'W' or 'L', got {v}")
        return v.upper()
    
    class Config:
        """Pydantic config."""
        extra = 'allow'  # Allow extra fields from API that we don't explicitly define


class PlayerGameLogResponse(BaseModel):
    """
    Schema for API response containing player game logs.
    """
    player_name: str = Field(..., description="Player name")
    season_type: str = Field(..., description="Season type (Regular Season/Playoffs)")
    season_year: int = Field(..., ge=2000, le=2100, description="Season year")
    game_logs: List[PlayerGameLog] = Field(..., description="List of game logs")
    total_games: int = Field(..., ge=0, description="Total number of games")
    
    @field_validator('season_type')
    @classmethod
    def validate_season_type(cls, v: str) -> str:
        """Validate season type."""
        valid_types = ['Regular Season', 'Playoffs', 'Pre Season']
        if v not in valid_types:
            raise ValueError(f"Season type must be one of {valid_types}, got {v}")
        return v


class ProcessedGameLog(PlayerGameLog):
    """
    Schema for processed/cleaned game log data.
    Inherits from PlayerGameLog but adds validation for processed data.
    """
    # Override with processed column names (standardized)
    game_id: Optional[str] = Field(None, alias='GAME_ID', description="Unique game identifier")
    game_date: Optional[str] = Field(None, alias='GAME_DATE', description="Date of the game")
    matchup: Optional[str] = Field(None, alias='MATCHUP', description="Matchup string")
    points: Optional[float] = Field(None, alias='PTS', ge=0, description="Points scored")
    minutes: Optional[float] = Field(None, alias='MIN', ge=0, le=48, description="Minutes played")
    
    class Config:
        """Pydantic config."""
        populate_by_name = True  # Allow both alias and field name
        extra = 'allow'


class FeatureEngineeredGameLog(ProcessedGameLog):
    """
    Schema for feature-engineered game log data with ML features.
    """
    # Rolling features (examples - actual features depend on feature engineering)
    PTS_rolling_3: Optional[float] = Field(None, description="3-game rolling average of points")
    PTS_rolling_5: Optional[float] = Field(None, description="5-game rolling average of points")
    PTS_rolling_10: Optional[float] = Field(None, description="10-game rolling average of points")
    
    # Efficiency features
    TS_PCT: Optional[float] = Field(None, ge=0, le=1, description="True shooting percentage")
    EFG_PCT: Optional[float] = Field(None, ge=0, le=1, description="Effective field goal percentage")
    
    # Time-based features
    DAY_OF_WEEK: Optional[int] = Field(None, ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    MONTH: Optional[int] = Field(None, ge=1, le=12, description="Month (1-12)")
    DAYS_REST: Optional[float] = Field(None, ge=0, description="Days of rest since last game")
    BACK_TO_BACK: Optional[int] = Field(None, ge=0, le=1, description="Back-to-back game indicator")
    
    # Matchup features
    IS_HOME: Optional[int] = Field(None, ge=0, le=1, description="Home game indicator")
    IS_AWAY: Optional[int] = Field(None, ge=0, le=1, description="Away game indicator")
    
    # Target variable for prediction
    PTS_next: Optional[float] = Field(None, ge=0, description="Next game points (target variable)")
    
    class Config:
        """Pydantic config."""
        populate_by_name = True
        extra = 'allow'

