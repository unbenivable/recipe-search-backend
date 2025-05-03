# Recipe Search Backend

This is a FastAPI backend for recipe search and ingredient extraction from images, using Google Vertex AI (Gemini) for image analysis and BigQuery for recipe data.

## Features
- Extract ingredients from food images using Google Vertex AI Gemini
- Search recipes by ingredients (BigQuery)
- Ready for deployment on Railway

## Requirements
- Python 3.9+
- Google Cloud project with Vertex AI and BigQuery enabled
- Service account with appropriate permissions

## Setup

1. **Clone the repository**
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Set environment variables:**
   - `GOOGLE_APPLICATION_CREDENTIALS_JSON`: Paste the full contents of your GCP service account JSON (as a single line) into this environment variable.
   - `PROJECT_ID`: (Optional) Set in code or as an env var if you want to override the default.

   Example (local) - make sure to use a secure method:
   ```sh
   # NEVER store credentials in files that might be committed to git
   # NEVER hardcode credentials in your code
   # Use environment variables or secure credential management services
   export GOOGLE_APPLICATION_CREDENTIALS_JSON="$(cat /path/to/gcp-credentials.json)"
   ```

4. **Run the server:**
   ```sh
   uvicorn app:app --reload --port 8000
   ```

## API Endpoints

### `POST /analyze-image`
Upload a food image and get a list of detected ingredients.

**Request:**
- `file`: image file (form-data)

**Response:**
```json
{
  "ingredients": ["ingredient1", "ingredient2", ...]
}
```

## Deployment (Railway)
- Add `GOOGLE_APPLICATION_CREDENTIALS_JSON` as a Railway environment variable using their secure environment variable storage.
- Deploy as a Python service.

## Security Notes
- All credentials are handled via environment variables.
- No API keys or service account credentials are stored in the codebase.
- This repository is configured to ignore credential files (.env, *.json, etc.)
- Use BFG Repo-Cleaner if you need to remove sensitive information from git history.
- Use Railway's secure environment variables for deployment.

## License
MIT 