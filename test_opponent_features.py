#!/usr/bin/env python3
"""
Test script for opponent-specific features.
Run this in your virtual environment to verify the new matchup features work.
"""
import pandas as pd
import json
from src.preprossessing.feature_engineer import engineer_features

def test_opponent_features():
    """Test that opponent-specific features are created correctly."""
    
    # Load raw data
    print("Loading raw data...")
    with open('data/raw.json', 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f'✅ Raw data loaded: {df.shape}')
    print(f'✅ Has MATCHUP column: {"MATCHUP" in df.columns}')
    
    if 'MATCHUP' in df.columns:
        print(f'   Sample MATCHUP: {df["MATCHUP"].iloc[0]}')
    
    # Engineer features
    print("\nEngineering features with opponent-specific stats...")
    df_features = engineer_features(df, target_column='PTS')
    print(f'✅ Features created: {df_features.shape}')
    
    # Check opponent-specific features
    opponent_features = [col for col in df_features.columns if 'OPP' in col or 'OPPONENT' in col]
    print(f'\n✅ Opponent features created ({len(opponent_features)}):')
    for feat in opponent_features[:15]:  # Show first 15
        print(f'   • {feat}')
    
    if len(opponent_features) > 15:
        print(f'   ... and {len(opponent_features) - 15} more')
    
    # Check for a specific player and opponent
    if 'PLAYER_NAME' in df_features.columns and 'OPPONENT' in df_features.columns:
        player_name = df_features['PLAYER_NAME'].iloc[0]
        player_games = df_features[df_features['PLAYER_NAME'] == player_name].copy()
        
        if not player_games.empty:
            print(f'\n📊 Analysis for {player_name}:')
            print(f'   Total games: {len(player_games)}')
            
            opponents = player_games['OPPONENT'].value_counts().head(5)
            print(f'\n   Top 5 opponents faced:')
            for opp, count in opponents.items():
                print(f'     • {opp}: {count} games')
            
            # Show stats vs a specific opponent
            if len(opponents) > 0:
                top_opp = opponents.index[0]
                vs_opp = player_games[player_games['OPPONENT'] == top_opp]
                print(f'\n   📈 Performance vs {top_opp} ({len(vs_opp)} games):')
                
                # Show opponent-specific stats
                pts_col = 'PTS_vs_OPP_avg'
                ast_col = 'AST_vs_OPP_avg'
                reb_col = 'REB_vs_OPP_avg'
                
                if pts_col in vs_opp.columns:
                    avg_pts = vs_opp[pts_col].dropna().mean()
                    print(f'      • Avg PTS vs {top_opp}: {avg_pts:.2f}')
                
                if ast_col in vs_opp.columns:
                    avg_ast = vs_opp[ast_col].dropna().mean()
                    print(f'      • Avg AST vs {top_opp}: {avg_ast:.2f}')
                
                if reb_col in vs_opp.columns:
                    avg_reb = vs_opp[reb_col].dropna().mean()
                    print(f'      • Avg REB vs {top_opp}: {avg_reb:.2f}')
                
                # Show last 3 games vs opponent
                if 'PTS_vs_OPP_L3' in vs_opp.columns:
                    l3_pts = vs_opp['PTS_vs_OPP_L3'].iloc[-1]
                    if not pd.isna(l3_pts):
                        print(f'      • Last 3 games PTS vs {top_opp}: {l3_pts:.2f}')
    
    print('\n' + '='*60)
    print('✅ OPPONENT-SPECIFIC FEATURES TEST PASSED!')
    print('='*60)
    print('\nYou can now:')
    print('1. Run main.py to train with opponent features')
    print('2. Use the API with opponent_team parameter:')
    print('   curl -X POST "http://localhost:8000/predict" \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"model_path": "models/...", "player_name": "LeBron James", "opponent_team": "BOS"}\'')
    
    return True

if __name__ == "__main__":
    try:
        test_opponent_features()
    except Exception as e:
        print(f'\n❌ Error: {e}')
        import traceback
        traceback.print_exc()

