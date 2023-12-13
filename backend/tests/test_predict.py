import os
import sys

from fastapi.testclient import TestClient


# Add the root directory of the project to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app

client = TestClient(app)


def test_predict_endpoint():
    # Test with a valid request
    response = client.post("/predict", json={"word": "english"})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "all_matches" in data
    assert isinstance(data["prediction"], str)
    assert isinstance(data["all_matches"], dict)

    # Test with an invalid request (missing 'word')
    response = client.post("/predict", json={})
    assert response.status_code == 422  # 422 Unprocessable Entity

    # Add more tests as needed, e.g., with different words, with edge cases, etc
    # @TODO: Add more tests
