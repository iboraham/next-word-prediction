### API Specification for Text Prediction Service

#### Overview

This document outlines the API specifications for the Text Prediction Service, designed to process text data and predict the next word based on n-grams analysis. The service can handle data from two sources: a CSV file and a database. It returns the most probable next word along with the top n predictions and their probabilities.

#### Base URL

The base URL for accessing the API will depend on the deployment but typically follows the format: `http://<host>:<port>`. For local testing, it's usually `http://localhost:8000`.

#### Endpoints

1. **Predict Next Word**

   - **URL**: `/predict`
   - **Method**: `POST`
   - **Description**: Receives a word and a data source, processes the text data, and returns the most likely next word along with the top n predictions and their probabilities.
   - **Request Body**:
     - `word` (string): The word for which the next word prediction is needed.
     - `data_source` (string): Specifies the data source; it can be either `"csv"` for CSV file or `"db"` for database.
   - **Response**:
     - `prediction` (string): The most probable next word.
     - `top_3_predictions` (array of objects): Each object contains a `word` (string) and its `probability` (float), representing the top n predicted next words and their respective probabilities.
   - **Sample Request**:
     ```json
     {
       "word": "example",
       "data_source": "csv"
     }
     ```
   - **Sample Response**:
     ```json
     {
       "prediction": "word",
       "top_3_predictions": [
         { "word": "word", "probability": 0.5 },
         { "word": "test", "probability": 0.3 },
         { "word": "example", "probability": 0.2 }
       ]
     }
     ```

#### Error Handling

Errors are returned as standard HTTP response codes. Common errors include:

- `400 Bad Request`: The request was unacceptable, often due to missing a required parameter.
- `500 Internal Server Error`: Something went wrong on the server side.

Each error response will contain a message explaining the nature of the error.

#### Security and Authentication

Currently, the API does not require authentication. However, depending on the deployment environment and usage, implementing an authentication mechanism (like API keys or OAuth tokens) is recommended for production use.

#### Rate Limiting

There is no rate limiting in place for this API. However, it is advisable to implement rate limiting to prevent abuse and ensure service availability.

#### Data Privacy

Ensure that any sensitive data processed through this API is handled in compliance with data protection regulations applicable in your region (like GDPR in Europe). It is recommended to anonymize data where possible.

#### Best Practices for Using the API

- Validate and sanitize input data on the client side before making API requests.
- Handle all possible HTTP response codes gracefully in your frontend application.
- Monitor the performance and adjust the deployment setup as needed to handle the expected load.

---

Created by [Onur Serbetci](iboraham.github.io)
