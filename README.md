# Recipe Search Backend

This is a FastAPI backend for recipe search and ingredient extraction from images, using Google Vertex AI (Gemini) for image analysis and BigQuery for recipe data.

## Features
- Extract ingredients from food images using Google Vertex AI Gemini
- Search recipes by ingredients (BigQuery)
- Visual recipe search using Vertex AI multimodal embeddings
- Comprehensive recipe recommendations combining ingredient and visual similarity
- Ready for deployment on Railway

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

### `POST /similar-recipe-images`
Upload a food image and get visually similar recipes.

**Request:**
- `file`: image file (form-data)
- `limit`: (optional) maximum number of results to return

**Response:**
```json
{
  "similar_recipes": [
    {
      "title": "Recipe Title",
      "similarity_score": 0.92,
      "image_url": "https://example.com/recipe.jpg",
      "recipe_id": "recipe123"
    },
    ...
  ]
}
```

### `POST /image-to-recipe`
A comprehensive endpoint that combines ingredient analysis and visual similarity.

**Request:**
- `file`: image file (form-data)

**Response:**
```json
{
  "detected_ingredients": ["ingredient1", "ingredient2", ...],
  "recipes_by_ingredients": [
    {
      "title": "Recipe Title",
      "ingredients": ["ingredient1", "ingredient2", ...],
      "directions": ["Step 1", "Step 2", ...]
    },
    ...
  ],
  "similar_looking_recipes": [
    {
      "title": "Recipe Title",
      "similarity_score": 0.92,
      "image_url": "https://example.com/recipe.jpg",
      "recipe_id": "recipe123"
    },
    ...
  ]
}
```

## Pagination Support

The API now includes enhanced pagination features to simplify frontend implementation.

### Using Pagination

All search endpoints support the following pagination parameters:
- `page` - The page number to return (default: 1)
- `page_size` - Number of results per page (default: 20)
- `max_results` - Maximum total results to return (default: 200)

The response includes a `pagination` object with the following properties:

```json
"pagination": {
  "total": 120,         // Total number of matching results
  "page": 2,            // Current page
  "page_size": 20,      // Results per page
  "pages": 6,           // Total number of pages
  "has_next": true,     // Whether there is a next page
  "has_prev": true,     // Whether there is a previous page
  "next_page": 3,       // Next page number
  "prev_page": 1,       // Previous page number
  "first_page": 1,      // First page number
  "last_page": 6,       // Last page number
  "page_numbers": [1,2,3,4] // Suggested page numbers for navigation
}
```

### Pagination Helper Endpoint

For advanced pagination UI needs, use the `/api/pagination-helper` endpoint.

Example request:
```
GET /api/pagination-helper?current_page=3&total_pages=10&items_per_page=20&total_items=200&max_page_buttons=5
```

This returns an optimized pagination structure for building UI components, including:
- Visible page numbers for displaying page buttons
- Previous/next navigation
- First/last page links
- Information about what items are being displayed

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