# Next Word Prediction App

## Overview

This repository contains the code for a Next Word Prediction application. The application predicts the next word in a given text sequence using n-grams analysis. The frontend is a simple web interface built with HTML and Tailwind CSS, and the backend is a FastAPI server that handles text prediction requests.

## Features

- Predicts the next word based on the input text.
- Offers a simple and interactive web interface.
- Utilizes FastAPI for efficient backend processing.
- Supports different data sources for predictions.

## Installation

### Prerequisites

- Python 3.8+
- Pipenv
- Node.js (optional, for frontend modifications)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/iboraham/next-word-prediction.git
   ```

2) Navigate to the backend directory and install dependencies:
   ```bash
   cd backend
   pipenv install
   ```
3) Start the FastAPI server:
   ```bash
   pipenv run uvicorn app:app --reload
   ```

### Running the Application

Open the `index.html` file in a web browser to access the Next Word Prediction application. Ensure the backend server is running for the prediction functionality to work.

Optionally, you can run the frontend in development mode to make changes to the UI. To do so:

```bash
npx serve .
```

## API Documentation

Refer to `./backend/README.md` for detailed API documentation.

## Testing

To run the tests, navigate to the backend directory and execute:

```bash
pipenv run pytest
```

## Contributing

Contributions are welcome! If you have suggestions or want to contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push to your fork.
4. Create a pull request against the main repository.

---

Created by [Onur Serbetci](iboraham.github.io)
