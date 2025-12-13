#!/bin/bash
# Quick prediction commands with the correct model path

echo "======================================================"
echo "Making Predictions with Opponent-Specific Features"
echo "======================================================"
echo ""

# Get the most recent model
MODEL_PATH=$(ls -t models/*.pkl | head -1)
echo "Using model: $MODEL_PATH"
echo ""

# 1. General prediction (all games)
echo "1. General Prediction (all games):"
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{
    \"model_path\": \"$MODEL_PATH\",
    \"player_name\": \"LeBron James\",
    \"n_predictions\": 5
  }"

echo -e "\n\n"

# 2. Prediction vs specific opponent
echo "2. Matchup-Specific Prediction (vs specific opponent):"
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{
    \"model_path\": \"$MODEL_PATH\",
    \"player_name\": \"LeBron James\",
    \"opponent_team\": \"NOP\",
    \"n_predictions\": 3
  }"

echo -e "\n\n======================================================"
echo "Done!"
echo "======================================================"

