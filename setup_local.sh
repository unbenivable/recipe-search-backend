#!/bin/bash
# Setup script for local testing of recipe-search-backend

echo "Recipe Search Backend - Local Setup"
echo "=================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "Error: pip is required but not installed."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Ask for credentials file
echo
echo "Google Cloud credentials setup:"
echo "------------------------------"
echo "Do you have a Google Cloud service account JSON file? (y/n)"
read -r has_credentials

if [ "$has_credentials" = "y" ]; then
    echo "Enter the path to your service account JSON file:"
    read -r credentials_path
    
    if [ -f "$credentials_path" ]; then
        # Create .env file
        echo "Creating .env file with credentials..."
        echo "GOOGLE_APPLICATION_CREDENTIALS_JSON='$(cat "$credentials_path" | tr -d '\n')'" > .env
        echo "Environment file created successfully."
    else
        echo "Error: File not found at $credentials_path"
        echo "Creating .env file with placeholder..."
        echo "GOOGLE_APPLICATION_CREDENTIALS_JSON='YOUR_CREDENTIALS_HERE'" > .env
    fi
else
    echo "No credentials provided. The app will use mock data for testing."
    echo "Creating .env file with placeholder..."
    echo "GOOGLE_APPLICATION_CREDENTIALS_JSON='YOUR_CREDENTIALS_HERE'" > .env
fi

echo
echo "Setup completed!"
echo
echo "To start the server, run:"
echo "------------------------"
echo "set -a; source .env; set +a"
echo "uvicorn app:app --reload --port 8000"
echo
echo "The API will be available at: http://localhost:8000"
echo "API documentation at: http://localhost:8000/docs" 