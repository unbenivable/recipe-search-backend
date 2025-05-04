# Recipe Search Backend API - With Google Vertex AI
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import base64
import sys
import uvicorn
from google.cloud import bigquery
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.vision_models import Image
from vertexai.vision_models import MultiModalEmbeddingModel
from io import BytesIO
import time
from functools import lru_cache
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Simple in-memory cache with TTL
class TTLCache:
    def __init__(self, max_size=100, ttl=3600):  # 1 hour default TTL
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        
    def get(self, key):
        if key not in self.cache:
            return None
        
        value, timestamp = self.cache[key]
        if time.time() - timestamp > self.ttl:
            # Expired
            del self.cache[key]
            return None
            
        return value
        
    def set(self, key, value):
        # Prune cache if it's too large
        if len(self.cache) >= self.max_size:
            # Remove oldest items
            oldest_key = min(self.cache.items(), key=lambda x: x[1][1])[0]
            del self.cache[oldest_key]
            
        self.cache[key] = (value, time.time())
        
    def clear(self):
        self.cache.clear()

# Initialize cache
recipe_cache = TTLCache(max_size=200, ttl=1800)  # 30 minute cache

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Google Cloud clients
client = None
try:
    # Use environment variable for credentials
    print("Using credentials from environment variable")
    creds_str = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON", "{}")
    
    # Clean up the credentials string
    # Remove any single quotes at beginning and end (common in .env files)
    if creds_str.startswith("'") and creds_str.endswith("'"):
        creds_str = creds_str[1:-1]
    elif creds_str.startswith('"') and creds_str.endswith('"'):
        creds_str = creds_str[1:-1]
        
    # Remove any line breaks that might have been introduced
    creds_str = creds_str.replace('\n', '\\n')
    
    # Print first few characters to debug (but don't expose full credentials)
    if len(creds_str) > 0:
        print(f"Credentials string length: {len(creds_str)}")
        if len(creds_str) < 10:
            print("WARNING: Credentials string is too short")
        
    try:
        credentials_info = json.loads(creds_str)
        print("Successfully parsed credentials JSON")
    except json.JSONDecodeError as je:
        print(f"JSON parse error at position {je.pos}, line {je.lineno}, column {je.colno}")
        raise
    
    # Initialize BigQuery client
    credentials = service_account.Credentials.from_service_account_info(credentials_info)
    client = bigquery.Client(credentials=credentials, project="recipe-data-pipeline")
    
    # Initialize Vertex AI
    vertexai.init(
        project="recipe-data-pipeline",
        location="us-central1",
        credentials=credentials
    )
    
    print(f"Successfully initialized Google Cloud clients for project: recipe-data-pipeline")
except json.JSONDecodeError as je:
    print(f"Warning: Invalid JSON in credentials: {je}")
    print("Check your GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable format")
    print("Using mock data for local testing")
except Exception as e:
    print(f"Warning: Unable to initialize Google Cloud clients: {e}")
    print("Using mock data for local testing")

# Load the multimodal embedding model for image search
embedding_model = None
try:
    embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    print("Successfully loaded MultiModal Embedding model")
except Exception as e:
    print(f"Warning: Unable to load MultiModal Embedding model: {e}")

class IngredientRequest(BaseModel):
    ingredients: List[str]
    
class SearchRequest(BaseModel):
    ingredients: List[str]
    page: int = 1
    page_size: int = 20
    min_matches: int = 1
    max_results: int = 100  # Reduced from 200 to 100 maximum results

@app.get("/")
async def root():
    """
    Root endpoint that provides basic application info.
    """
    return {
        "app": "Recipe Search API",
        "status": "ok",
        "docs": "/docs"
    }

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint for Railway deployment.
    Returns status OK to indicate the service is up and running.
    """
    return {"status": "ok"}
    
# Add explicit endpoint at root level (no /api prefix) as fallback
@app.get("/health")
async def health_check_root():
    """
    Alternative health check endpoint without /api prefix.
    """
    return {"status": "ok"}

@app.post("/search")
async def search_recipes(request: SearchRequest):
    # Apply absolute hard limits regardless of what's requested
    ABSOLUTE_MAX_RESULTS = 100  # Never return more than 100 results total
    
    user_ingredients = [ing.lower().strip() for ing in request.ingredients]
    page = max(1, min(100, request.page))  # Limit page between 1-100
    page_size = max(1, min(20, request.page_size))  # Limit page_size between 1-20
    min_matches = request.min_matches
    max_results = min(ABSOLUTE_MAX_RESULTS, request.max_results)
    
    print(f"SEARCH REQUEST: ingredients={user_ingredients}, page={page}, page_size={page_size}, max_results={max_results}")

    if not user_ingredients:
        return {"recipes": [], "total": 0, "page": page, "page_size": page_size, "pages": 0}
    
    # Create a cache key from the search parameters
    cache_params = {
        "ingredients": sorted(user_ingredients),
        "page": page, 
        "page_size": page_size,
        "min_matches": min_matches
    }
    cache_key = hashlib.md5(json.dumps(cache_params).encode()).hexdigest()
    
    # Check cache first
    cached_result = recipe_cache.get(cache_key)
    if cached_result:
        print(f"Cache hit for ingredients: {', '.join(user_ingredients)}, page {page}")
        return cached_result
        
    # For local testing or when BigQuery is not available
    if client is None:
        print("Using mock data for recipe search")
        # Return mock data with the searched ingredients
        result = {
            "recipes": [
                {
                    "title": f"Mock Recipe with {', '.join(user_ingredients[:2])}",
                    "ingredients": user_ingredients + ["salt", "pepper", "olive oil"],
                    "directions": ["Mix ingredients", "Cook for 20 minutes", "Serve hot"]
                },
                {
                    "title": f"Another Recipe with {user_ingredients[0]}",
                    "ingredients": [user_ingredients[0], "garlic", "onion", "butter"],
                    "directions": ["Prepare ingredients", "Cook slowly", "Garnish and serve"]
                }
            ],
            "total": 2,
            "page": page,
            "page_size": page_size,
            "pages": 1
        }
        # Cache the result
        recipe_cache.set(cache_key, result)
        return result
    
    try:
        start_time = time.time()
        
        # Optimize query: Use array contains for better performance
        # Build optimized clauses for each ingredient
        match_clauses = []
        for ing in user_ingredients:
            # For exact matches (faster)
            exact_match = f"EXISTS(SELECT 1 FROM UNNEST(ingredients) AS i WHERE LOWER(i) = '{ing}')"
            # For partial matches
            partial_match = f"EXISTS(SELECT 1 FROM UNNEST(ingredients) AS i WHERE LOWER(i) LIKE '%{ing}%')"
            match_clauses.append(f"({exact_match} OR {partial_match})")

        score_expression = " + ".join([f"CASE WHEN {clause} THEN 1 ELSE 0 END" for clause in match_clauses])
        
        # First, get the total count for pagination
        count_query = f"""
            SELECT COUNT(*) as total
            FROM (
                SELECT title
                FROM `recipe-data-pipeline.recipes.structured_recipes`
                WHERE ({score_expression}) >= {min_matches}
                LIMIT {ABSOLUTE_MAX_RESULTS}  -- Absolute hard limit
            )
        """
        
        count_result = client.query(count_query).result()
        total_count = min(next(iter(count_result)).total, ABSOLUTE_MAX_RESULTS)  # Apply max result limit
        total_pages = (total_count + page_size - 1) // page_size  # Ceiling division
        
        # Calculate offset for pagination, ensuring we don't exceed the max_results
        offset = min((page - 1) * page_size, total_count - 1)
        
        # Adjust page_size to not exceed maximum results
        effective_page_size = min(page_size, ABSOLUTE_MAX_RESULTS - offset)
        
        # Add debug logging
        print(f"SQL QUERY: ingredients={len(user_ingredients)}, page={page}, limit={effective_page_size}, offset={offset}, max_results={max_results}")
        
        # Main query with pagination
        query = f"""
            SELECT title, ingredients, directions,
                   ({score_expression}) AS match_score
            FROM `recipe-data-pipeline.recipes.structured_recipes`
            WHERE ({score_expression}) >= {min_matches}
            ORDER BY match_score DESC
            LIMIT {effective_page_size} OFFSET {offset}
        """

        results = client.query(query).result()

        recipes = []
        for row in results:
            # Only add up to our absolute max
            if len(recipes) >= ABSOLUTE_MAX_RESULTS:
                print(f"WARNING: Truncating results at {ABSOLUTE_MAX_RESULTS}")
                break
                
            recipes.append({
                "title": row.title,
                "ingredients": row.ingredients,
                "directions": row.directions,
                "match_score": row.match_score
            })

        query_time = time.time() - start_time
        print(f"Query executed in {query_time:.2f} seconds")
        
        # Enhanced pagination info for frontend
        pagination = {
            "total": total_count,
            "page": page,
            "page_size": page_size,
            "pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
            "next_page": min(page + 1, total_pages) if page < total_pages else None,
            "prev_page": page - 1 if page > 1 else None,
            "first_page": 1,
            "last_page": total_pages,
            # Generate an array of page numbers for pagination controls
            "page_numbers": list(range(
                max(1, page - 2),  # 2 pages before current
                min(total_pages + 1, page + 3)  # 2 pages after current
            ))
        }
        
        result = {
            "recipes": recipes[:ABSOLUTE_MAX_RESULTS],  # Final safety check
            "query_time_seconds": query_time,
            "pagination": pagination
        }
        
        # Cache the result
        recipe_cache.set(cache_key, result)
        
        return result
    except Exception as e:
        print(f"Error in recipe search: {e}")
        # Return mock data if BigQuery query fails
        error_result = {
            "recipes": [
                {
                    "title": f"Mock Recipe with {', '.join(user_ingredients[:2])} (Error fallback)",
                    "ingredients": user_ingredients + ["salt", "pepper", "olive oil"],
                    "directions": ["Mix ingredients", "Cook for 20 minutes", "Serve hot"]
                }
            ],
            "error": str(e),
            "total": 1,
            "page": page,
            "page_size": page_size,
            "pages": 1
        }
        return error_result

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze a food image to extract ingredients using Google Vertex AI Gemini.
    
    Returns a list of identified ingredients that can be used for recipe search.
    """
    # Read the image file
    contents = await file.read()
    
    try:
        # Initialize Gemini model
        model = GenerativeModel("gemini-pro-vision")
        
        # Create the prompt and image part
        prompt = "Output ONLY food items and ingredients visible in the image. No descriptions, no explanations, just the food items themselves. Format as a simple comma-separated list."
        image_part = Part.from_data(mime_type=file.content_type, data=contents)
        
        # Generate content
        response = model.generate_content([prompt, image_part])
        
        # Extract ingredients from the response
        ingredients_text = response.text
        
        # Parse the comma-separated list into individual ingredients
        ingredients = [item.strip() for item in ingredients_text.split(',')]
        
        # Remove any empty strings or bullet points
        ingredients = [item.replace('-', '').strip() for item in ingredients if item.strip()]
        
        return {"ingredients": ingredients}
    except Exception as e:
        print(f"Error analyzing image: {e}")
        # Return mock data for testing if analysis fails
        return {"ingredients": ["tomato", "onion", "garlic"], "error": str(e)}

@app.post("/search-by-image")
async def search_by_image(
    file: UploadFile = File(...), 
    max_results: int = 200,
    page: int = 1,
    page_size: int = 20
):
    """
    Analyze a food image and search for recipes based on the detected ingredients.
    
    This combines the image analysis and recipe search in one convenient endpoint.
    """
    # Apply absolute hard limits
    ABSOLUTE_MAX_RESULTS = 100
    page = max(1, min(100, page))
    page_size = max(1, min(20, page_size))
    max_results = min(ABSOLUTE_MAX_RESULTS, max_results)
    
    print(f"IMAGE SEARCH: page={page}, page_size={page_size}, max_results={max_results}")
    
    # First, analyze the image to get ingredients
    analysis_result = await analyze_image(file)
    ingredients = analysis_result.get("ingredients", [])
    
    if not ingredients:
        return {
            "recipes": [], 
            "detected_ingredients": [],
            "pagination": {
                "total": 0,
                "page": page,
                "page_size": page_size,
                "pages": 0,
                "has_next": False,
                "has_prev": False,
                "next_page": None,
                "prev_page": None,
                "first_page": 1,
                "last_page": 0,
                "page_numbers": []
            }
        }
    
    # Create a SearchRequest to use our optimized search endpoint
    search_request = SearchRequest(
        ingredients=ingredients,
        page=page,
        page_size=page_size,
        min_matches=1,
        max_results=max_results
    )
    
    # Use the existing search function
    search_result = await search_recipes(search_request)
    
    return {
        "recipes": search_result.get("recipes", []),
        "detected_ingredients": ingredients,
        "pagination": search_result.get("pagination", {})
    }

@app.post("/similar-recipe-images")
async def search_similar_recipe_images(
    file: UploadFile = File(...),
    limit: Optional[int] = 10
):
    """
    Find recipes with similar looking dishes based on the uploaded image.
    Uses Vertex AI's multimodal embedding model to find visually similar recipes.
    """
    if embedding_model is None:
        return {"error": "Image search capability is not available"}
    
    try:
        # Read the image file
        contents = await file.read()
        
        # Create a Vertex AI image from the uploaded bytes
        vertex_image = Image(contents)
        
        # Generate embedding for the query image
        image_embedding = embedding_model.get_embeddings(
            image=vertex_image,
            contextual_text="food dish"
        )
        
        # In a production environment, you would:
        # 1. Have a database of recipe images with pre-computed embeddings
        # 2. Perform a vector search to find the closest matches
        # 3. Return the associated recipes
        
        # For now, we'll use a mock response:
        return {
            "similar_recipes": [
                {
                    "title": "Similar Recipe 1",
                    "similarity_score": 0.92,
                    "image_url": "https://example.com/recipe1.jpg",
                    "recipe_id": "recipe123"
                },
                {
                    "title": "Similar Recipe 2", 
                    "similarity_score": 0.87,
                    "image_url": "https://example.com/recipe2.jpg",
                    "recipe_id": "recipe456"
                }
            ],
            "message": "In production, this would return actual similar recipes based on vector search"
        }
    except Exception as e:
        print(f"Error in image similarity search: {e}")
        return {"error": str(e)}

@app.post("/image-to-recipe")
async def image_to_recipe(
    file: UploadFile = File(...), 
    max_results: int = 200,
    page: int = 1,
    page_size: int = 10
):
    """
    A comprehensive endpoint that:
    1. Analyzes the food image to identify ingredients
    2. Searches for recipes based on the ingredients
    3. Finds visually similar recipes based on the image appearance
    
    Returns all three results combined.
    """
    # Apply absolute hard limits
    ABSOLUTE_MAX_RESULTS = 100
    page = max(1, min(100, page))
    page_size = max(1, min(20, page_size))
    max_results = min(ABSOLUTE_MAX_RESULTS, max_results)
    
    print(f"IMAGE-TO-RECIPE: page={page}, page_size={page_size}, max_results={max_results}")
    
    start_time = time.time()
    
    # First, analyze the image to get ingredients
    contents = await file.read()
    
    # Create a cache key for this image
    image_hash = hashlib.md5(contents).hexdigest()
    cache_key = f"img2recipe_{image_hash}_{max_results}_{page}_{page_size}"  # Include pagination in cache key
    
    # Check cache first
    cached_result = recipe_cache.get(cache_key)
    if cached_result:
        print(f"Cache hit for image: {file.filename}")
        return cached_result
    
    # Set up async tasks to run in parallel
    async def analyze_image_task():
        # Reset file position
        await file.seek(0)
        return await analyze_image(file)
    
    async def get_similar_recipes_task():
        # Reset file position
        await file.seek(0)
        return await search_similar_recipe_images(file, limit=min(10, max_results))
    
    # Run both tasks in parallel
    analysis_task = asyncio.create_task(analyze_image_task())
    similar_task = asyncio.create_task(get_similar_recipes_task())
    
    # Wait for analysis to complete first (we need ingredients for recipe search)
    analysis_result = await analysis_task
    ingredients = analysis_result.get("ingredients", [])
    
    # For recipe search, we'll create a search request
    search_request = SearchRequest(
        ingredients=ingredients,
        page=page,
        page_size=page_size,
        min_matches=1,
        max_results=max_results
    )
    
    # Start recipe search immediately
    recipes_task = asyncio.create_task(search_recipes(search_request))
    
    # Wait for both remaining tasks to complete
    similar_result, recipes_result = await asyncio.gather(similar_task, recipes_task)
    
    similar_recipes = similar_result.get("similar_recipes", [])
    recipes_by_ingredients = recipes_result.get("recipes", [])
    
    # Prepare the final response
    result = {
        "detected_ingredients": ingredients,
        "recipes_by_ingredients": recipes_by_ingredients,
        "similar_looking_recipes": similar_recipes,
        "processing_time_seconds": time.time() - start_time,
        "pagination": recipes_result.get("pagination", {})
    }
    
    # Cache the result
    recipe_cache.set(cache_key, result)
    
    return result

@app.get("/api/pagination-helper")
async def pagination_helper(
    current_page: int = 1,
    total_pages: int = 1,
    items_per_page: int = 20,
    total_items: int = 0,
    max_page_buttons: int = 5
):
    """
    Helper endpoint to generate pagination navigation for the frontend.
    
    Returns a structure with information for building a pagination UI component.
    """
    # Calculate visible page range
    half_max = max_page_buttons // 2
    start_page = max(1, current_page - half_max)
    end_page = min(total_pages, start_page + max_page_buttons - 1)
    
    # Adjust start if we're near the end
    if end_page == total_pages:
        start_page = max(1, end_page - max_page_buttons + 1)
    
    # Generate page numbers
    page_numbers = list(range(start_page, end_page + 1))
    
    # Generate navigation links
    pagination = {
        "current_page": current_page,
        "total_pages": total_pages,
        "items_per_page": items_per_page,
        "total_items": total_items,
        "visible_pages": page_numbers,
        "has_previous": current_page > 1,
        "has_next": current_page < total_pages,
        "previous_page": current_page - 1 if current_page > 1 else None,
        "next_page": current_page + 1 if current_page < total_pages else None,
        "first_page": 1,
        "last_page": total_pages,
        "showing_items": {
            "from": ((current_page - 1) * items_per_page) + 1 if total_items > 0 else 0,
            "to": min(current_page * items_per_page, total_items)
        }
    }
    
    return pagination

# Start the server for local testing
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
