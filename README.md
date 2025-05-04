# Recipe Search Backend

This is a FastAPI backend for recipe search and ingredient extraction from images, using Google Vertex AI (Gemini) for image analysis and BigQuery for recipe data.

## Repository
- **GitHub**: https://github.com/unbenivable/recipe-search-backend

## Features
- Extract ingredients from food images using Google Vertex AI Gemini
- Search recipes by ingredients (BigQuery)
- Visual recipe search using Vertex AI multimodal embeddings
- Comprehensive recipe recommendations combining ingredient and visual similarity
- Ready for deployment on Railway
- Optimized for high-volume search with batch processing and intelligent caching

## Deployment URLs
- **Backend (Railway)**: https://recipe-search-backend.railway.app
- **Frontend (Vercel)**: https://recipe-search-frontend.vercel.app

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

## Local Testing
See [README.local.md](README.local.md) for detailed instructions on local testing, including:
- Setting up credentials
- Running the server locally
- Testing with the Swagger UI
- Connecting to the Vercel frontend

## API Endpoints

### `/search` - Search for recipes by ingredients
- **Method**: POST
- **Parameters**: 
  - `ingredients`: List of ingredient strings
  - `page`: Page number (default: 1)
  - `page_size`: Results per page (default: 20)
  - `min_matches`: Minimum number of matching ingredients (default: 1)
  - `max_results`: Maximum total results to return (default: 100)

### `/batch-search` - Process multiple searches in one request
- **Method**: POST
- **Parameters**:
  - `ingredientSets`: List of ingredient lists (maximum 5 sets)
  - `page`: Page number (default: 1)
  - `page_size`: Results per page (default: 20)
  - `min_matches`: Minimum number of matching ingredients (default: 1)
  - `max_results`: Maximum total results to return (default: 100)
- **Description**: Optimized endpoint for reducing frontend request volume by processing multiple searches in a single API call

### `/analyze-image` - Extract ingredients from a food image
- **Method**: POST
- **Parameters**: Image file (multipart/form-data)

### `/visual-search` - Find visually similar recipes
- **Method**: POST
- **Parameters**: Image file (multipart/form-data)

### `/health` - Healthcheck endpoint
- **Method**: GET

### Rate Limiting

The API includes rate limiting to prevent excessive usage:
- Maximum 10 requests per minute per IP address
- Maximum 500 requests per 10-minute window globally
- Maximum 1,000 results per 10-minute window globally
- Duplicate searches within 5 seconds are throttled
- Batch searches limited to 5 ingredient sets per request

If limits are exceeded, the API will return a 429 status code with a "Too many requests" message.

## Performance Optimization

This API is designed for high-performance with multiple optimization techniques:
- Intelligent two-level caching (in-memory TTL cache + query-level caching)
- Rate limiting to prevent abuse and ensure fair usage
- Batch processing to reduce request volume
- Minimum length requirements for search terms
- Duplicate search detection to prevent hammering

## Frontend Integration Best Practices

To achieve optimal performance when integrating with the frontend:
1. **Implement debouncing** - Wait until user stops typing before sending requests (min 500ms delay)
2. **Add minimum character limits** - Only search after 2+ characters are entered
3. **Use batch searching** - Combine multiple searches into fewer requests
4. **Implement client-side caching** - Cache results to reduce duplicate queries
5. **Optimize React component lifecycle** - Prevent unnecessary re-renders

See the `frontend-optimizations.js` file for code examples of these techniques.

## Deployment (Railway)
- Add `GOOGLE_APPLICATION_CREDENTIALS_JSON` as a Railway environment variable using their secure environment variable storage.
- Deploy as a Python service.
- The backend is deployed at: https://recipe-search-backend.railway.app

## Frontend Integration
The frontend code should point to the appropriate backend URL:
- For production: https://recipe-search-backend.railway.app
- For local testing: http://localhost:8000 or your ngrok public URL

## Security Notes
- All credentials are handled via environment variables.
- No API keys or service account credentials are stored in the codebase.
- This repository is configured to ignore credential files (.env, *.json, etc.)
- Use BFG Repo-Cleaner if you need to remove sensitive information from git history.
- Use Railway's secure environment variables for deployment.

## License
MIT 