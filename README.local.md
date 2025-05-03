# Local Testing Guide for Recipe Search Backend

## Prerequisites
- Python 3.9+
- Google Cloud account with Vertex AI and BigQuery enabled
- Service account with access to these services

## Setup Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Google Cloud Credentials

#### Option 1: Using Environment Variables Directly
```bash
# Export your credentials JSON as an environment variable
export GOOGLE_APPLICATION_CREDENTIALS_JSON="$(cat /path/to/your-service-account.json)"
```

#### Option 2: Create a .env File
Create a file named `.env` with the following content:
```
GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type":"service_account","project_id":"recipe-data-pipeline",...}'
```

Then load it before running the app:
```bash
set -a; source .env; set +a
```

### 3. Running the Application
```bash
# Make sure you're in the recipe-search-backend directory
cd /path/to/recipe-search-backend

# Start the server
uvicorn app:app --reload --port 8000
```

The API will be available at: http://localhost:8000

### 4. Testing the API

#### Using the Swagger UI
Open your browser and navigate to: http://localhost:8000/docs

#### Using cURL

**Test the image analysis endpoint:**
```bash
curl -X POST http://localhost:8000/analyze-image \
  -F "file=@/path/to/your/food-image.jpg"
```

**Test the image search endpoint:**
```bash
curl -X POST http://localhost:8000/similar-recipe-images \
  -F "file=@/path/to/your/food-image.jpg"
```

**Test the comprehensive endpoint:**
```bash
curl -X POST http://localhost:8000/image-to-recipe \
  -F "file=@/path/to/your/food-image.jpg"
```

### 5. Connecting with Frontend

#### Frontend on Vercel
The frontend is deployed on Vercel at: https://recipe-search-frontend.vercel.app

When testing locally, you need to update the frontend to point to your local backend:

1. **For local frontend development**:
   ```javascript
   // Update API URL in your frontend code (typically in a config.js or similar file)
   const API_URL = 'http://localhost:8000';
   ```

2. **For testing with deployed Vercel frontend**:
   - Run your local backend with a tool like ngrok to expose it publicly:
     ```bash
     ngrok http 8000
     ```
   - Update the API URL in your deployed Vercel frontend settings:
     - Go to the Vercel Dashboard → Your Project → Settings → Environment Variables
     - Set API_URL to your ngrok URL (e.g., `https://your-ngrok-subdomain.ngrok.io`)
     - Redeploy the frontend

3. **For production**:
   - The frontend should use the Railway-deployed backend URL:
     ```javascript
     const API_URL = 'https://recipe-search-backend.railway.app';
     ```

When testing with a frontend deployed on Vercel:
1. Make sure CORS is properly configured (already handled in the backend code)
2. If testing from a different device, ensure you're using the correct network IP (not localhost)

### Troubleshooting

**Mock Data for Testing:**
If you don't have Google Cloud credentials set up, the app will use mock data for testing.

**Common Issues:**
- "Cannot load ASGI app": Make sure you're in the correct directory when running uvicorn
- Vertex AI errors: Verify your service account has the necessary permissions
- Empty response: Check that your image format is supported (JPEG, PNG, etc.)
- CORS errors: Make sure your backend's CORS configuration allows requests from your frontend origin

### Production vs Local
The main difference between local testing and the Railway deployment is:
1. In Railway, environment variables are set in the platform's dashboard
2. In production, the app will be exposed on a custom domain (https://recipe-search-backend.railway.app) 