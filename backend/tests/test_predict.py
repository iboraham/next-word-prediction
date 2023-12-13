def test_predict_endpoint(client):
    # Test with a valid request
    response = client.post("/predict", json={"word": "english", "data_source": "csv"})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "top_n_predictions" in data
    assert isinstance(data["prediction"], str)
    assert isinstance(data["top_n_predictions"], list)

    # Test with an invalid request (missing 'word')
    response = client.post("/predict", json={})
    assert response.status_code == 422  # 422 Unprocessable Entity

    # Add more tests as needed, e.g., with different words, with edge cases, etc
    # @TODO: Add more tests
