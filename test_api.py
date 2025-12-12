"""
Test script for the FastAPI endpoints.
Run this after starting the API server with: python3 api.py
"""
import requests
import json
from time import sleep

# API base URL
BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint."""
    print("\n" + "="*70)
    print("Testing Health Check Endpoint")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ Health check passed!")


def test_root_endpoint():
    """Test the root endpoint."""
    print("\n" + "="*70)
    print("Testing Root Endpoint")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    print("✅ Root endpoint passed!")


def test_train_model():
    """Test the model training endpoint."""
    print("\n" + "="*70)
    print("Testing Model Training Endpoint")
    print("="*70)
    
    # Training request
    request_data = {
        "player_name": "Stephen Curry",
        "target_column": "PTS",
        "model_type": "random_forest",
        "season_type": "Regular Season",
        "season_year": 2024,
        "min_minutes": 10,
        "test_size": 0.2,
        "compare_models": False
    }
    
    print(f"Request: {json.dumps(request_data, indent=2)}")
    print("\nSending training request... (this may take a minute)")
    
    response = requests.post(
        f"{BASE_URL}/train",
        json=request_data,
        timeout=300  # 5 minute timeout
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        print("\n✅ Model training passed!")
        return result.get("model_path")
    else:
        print(f"❌ Training failed: {response.text}")
        return None


def test_train_with_tuning():
    """Test the model training with hyperparameter tuning endpoint."""
    print("\n" + "="*70)
    print("Testing Model Training with Hyperparameter Tuning")
    print("="*70)
    
    # Training request
    request_data = {
        "player_name": "LeBron James",
        "target_column": "PTS",
        "model_type": "random_forest",
        "season_type": "Regular Season",
        "season_year": 2024,
        "min_minutes": 10,
        "test_size": 0.2,
        "compare_models": False
    }
    
    print(f"Request: {json.dumps(request_data, indent=2)}")
    print("\nSending training request with tuning... (this may take several minutes)")
    
    # Use randomized search with 10 iterations for faster testing
    response = requests.post(
        f"{BASE_URL}/train/tuning?n_iter=10",
        json=request_data,
        timeout=600  # 10 minute timeout
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        print("\n✅ Model training with tuning passed!")
        return result.get("model_path")
    else:
        print(f"❌ Training with tuning failed: {response.text}")
        return None


def test_list_models():
    """Test the list models endpoint."""
    print("\n" + "="*70)
    print("Testing List Models Endpoint")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Found {result['count']} models")
        if result['models']:
            print("\nModels:")
            for model in result['models'][:5]:  # Show first 5
                print(f"  - {model['filename']} ({model['size_mb']} MB)")
        print("✅ List models passed!")
    else:
        print(f"❌ List models failed: {response.text}")


def test_predictions(model_path: str = None):
    """Test the predictions endpoint."""
    print("\n" + "="*70)
    print("Testing Predictions Endpoint")
    print("="*70)
    
    if not model_path:
        # Get the latest model
        response = requests.get(f"{BASE_URL}/models")
        if response.status_code == 200:
            models = response.json()['models']
            if models:
                model_path = models[0]['path']
                print(f"Using latest model: {models[0]['filename']}")
            else:
                print("❌ No models available for prediction test")
                return
        else:
            print("❌ Could not fetch models")
            return
    
    # Prediction request
    request_data = {
        "model_path": model_path,
        "player_name": "Stephen Curry",
        "target_column": "PTS",
        "n_predictions": 5
    }
    
    print(f"Request: {json.dumps(request_data, indent=2)}")
    print("\nSending prediction request...")
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=request_data,
        timeout=120
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        print("\n✅ Predictions passed!")
    else:
        print(f"❌ Predictions failed: {response.text}")


def run_all_tests():
    """Run all API tests."""
    print("="*70)
    print("NBA Player Statistics ML API - Test Suite")
    print("="*70)
    print("\nMake sure the API server is running:")
    print("  python3 api.py")
    print("\nStarting tests in 3 seconds...")
    sleep(3)
    
    try:
        # Basic tests
        test_health_check()
        test_root_endpoint()
        test_list_models()
        
        # Training test (can take a while)
        print("\n⚠️  The following tests may take several minutes...")
        model_path = test_train_model()
        
        # Prediction test
        if model_path:
            test_predictions(model_path)
        
        # Optional: Test hyperparameter tuning (takes longer)
        # Uncomment to test:
        # test_train_with_tuning()
        
        print("\n" + "="*70)
        print("✅ All API tests completed!")
        print("="*70)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to API server.")
        print("Make sure the server is running: python3 api.py")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()

