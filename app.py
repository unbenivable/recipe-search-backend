# Recipe Search Backend API - With Google Vertex AI
from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException, Depends
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
import datetime
from fastapi.responses import JSONResponse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Initialize cache with longer TTL and larger size
recipe_cache = TTLCache(max_size=500, ttl=3600)  # 1 hour cache, 500 items (increased from 30 min, 200 items)

# Global request limiter
class RequestThrottler:
    def __init__(self):
        self.request_count = 0
        self.result_count = 0
        self.last_reset = datetime.datetime.now()
        self.reset_interval = datetime.timedelta(minutes=10)
        self.ip_timestamps = {}  # Track IPs and their request timestamps
        self.search_term_timestamps = {}  # Track recent search terms to prevent duplicates
        
    def should_throttle(self, client_ip, search_term=None):
        """Check if we should throttle requests"""
        now = datetime.datetime.now()
        
        # Reset counters periodically
        if now - self.last_reset > self.reset_interval:
            self.request_count = 0
            self.result_count = 0
            self.last_reset = now
            self.ip_timestamps = {}
            self.search_term_timestamps = {}
            logger.info(f"REQUEST COUNTERS RESET. New interval started at {now}")
        
        # Limit requests per IP (max 20 requests per minute per IP)
        ip_history = self.ip_timestamps.get(client_ip, [])
        ip_history = [ts for ts in ip_history if now - ts < datetime.timedelta(minutes=1)]
        
        self.ip_timestamps[client_ip] = ip_history + [now]
        
        # Check IP rate limit - stricter limit (10 per minute instead of 20)
        if len(ip_history) >= 10:
            logger.info(f"THROTTLING {client_ip}: Made {len(ip_history)} requests in the last minute")
            return True
            
        # If search term is provided, check for duplicate searches
        if search_term:
            # Create a composite key of IP + search term
            key = f"{client_ip}:{search_term}"
            last_search_time = self.search_term_timestamps.get(key)
            
            # If the same search was made in the last 5 seconds, throttle
            if last_search_time and (now - last_search_time < datetime.timedelta(seconds=5)):
                logger.info(f"THROTTLING {client_ip}: Duplicate search for '{search_term}' within 5 seconds")
                return True
                
            # Update the timestamp for this search term
            self.search_term_timestamps[key] = now
            
        # Global limits: max 500 requests or 1,000 results per 10-minute window
        if self.request_count >= 500:
            logger.info(f"GLOBAL THROTTLING: {self.request_count} requests in the current window")
            return True
            
        if self.result_count >= 1000:
            logger.info(f"GLOBAL THROTTLING: {self.result_count} results in the current window")
            return True
            
        # Increment request counter
        self.request_count += 1
        return False
        
    def add_results(self, count):
        """Add to the result counter"""
        self.result_count += count
        logger.info(f"COUNTER UPDATE: {self.request_count} requests, {self.result_count} results in current window")

# Initialize throttler
throttler = RequestThrottler()

app = FastAPI()

allowed_origins = [
    "https://recipe-search-frontend.vercel.app",
    "https://ingreddit.com",
    "https://www.ingreddit.com"
]
if os.environ.get("ENV") == "development":
    allowed_origins.append("http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Google Cloud clients
client = None
try:
    # Use environment variable for credentials
    logger.info("Using credentials from environment variable")
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
        logger.info(f"Credentials string length: {len(creds_str)}")
        if len(creds_str) < 10:
            logger.warning("WARNING: Credentials string is too short")
        
    try:
        credentials_info = json.loads(creds_str)
        logger.info("Successfully parsed credentials JSON")
    except json.JSONDecodeError as je:
        logger.error(f"JSON parse error at position {je.pos}, line {je.lineno}, column {je.colno}")
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
    
    logger.info(f"Successfully initialized Google Cloud clients for project: recipe-data-pipeline")
except json.JSONDecodeError as je:
    logger.warning(f"Warning: Invalid JSON in credentials: {je}")
    logger.warning("Check your GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable format")
    logger.warning("Using mock data for local testing")
except Exception as e:
    logger.warning(f"Warning: Unable to initialize Google Cloud clients: {e}")
    logger.warning("Using mock data for local testing")

# Load the multimodal embedding model for image search
embedding_model = None
try:
    embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    logger.info("Successfully loaded MultiModal Embedding model")
except Exception as e:
    logger.warning(f"Warning: Unable to load MultiModal Embedding model: {e}")

class IngredientRequest(BaseModel):
    ingredients: List[str]
    
class SearchRequest(BaseModel):
    ingredients: List[str]
    page: int = 1
    page_size: int = 20
    min_matches: int = 1
    max_results: int = 100  # Reduced from 200 to 100 maximum results
    matchAll: Optional[bool] = None  # Added for frontend compatibility
    
    def __init__(self, **data):
        super().__init__(**data)
        # If matchAll is provided and True, set min_matches to match all ingredients
        if self.matchAll is not None:
            if self.matchAll and len(self.ingredients) > 0:
                self.min_matches = len(self.ingredients)
            elif not self.matchAll:
                self.min_matches = 1

# Use environment variable for ABSOLUTE_MAX_RESULTS
ABSOLUTE_MAX_RESULTS = int(os.environ.get("MAX_RESULTS", 50))

# Batch search request model
class BatchSearchRequest(BaseModel):
    ingredientSets: List[List[str]]
    page: int = 1
    page_size: int = 20
    min_matches: int = 1
    max_results: int = 100

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
    logger.info("Health check called")
    return {"status": "ok"}

@app.post("/search")
async def search_recipes(request: SearchRequest, req: Request):
    # Apply absolute hard limits regardless of what's requested
    page = max(1, min(100, request.page))  # Limit page between 1-100
    page_size = max(1, min(20, request.page_size))  # Limit page_size between 1-20
    min_matches = request.min_matches
    max_results = min(ABSOLUTE_MAX_RESULTS, request.max_results)
    
    client_ip = req.state.client_ip
    logger.info(f"SEARCH REQUEST from {client_ip}: ingredients={request.ingredients}, page={page}, page_size={page_size}, max_results={max_results}")

    # Require at least 2 characters in any ingredient for search
    if not request.ingredients or any(len(ing.strip()) < 2 for ing in request.ingredients):
        return {
            "recipes": [], 
            "total": 0, 
            "page": page, 
            "page_size": page_size, 
            "pages": 0,
            "error": "Please enter ingredients with at least 2 characters each"
        }
    
    # Create a search term string for throttling (combine all ingredients)
    search_term = ",".join(sorted([ing.lower().strip() for ing in request.ingredients]))
    
    # Check if we should throttle this request including the search term
    if throttler.should_throttle(client_ip, search_term):
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many similar requests. Please try again later."}
        )
    
    # Create a cache key from the search parameters
    cache_params = {
        "ingredients": sorted([ing.lower().strip() for ing in request.ingredients]),
        "page": page, 
        "page_size": page_size,
        "min_matches": min_matches
    }
    cache_key = hashlib.md5(json.dumps(cache_params).encode()).hexdigest()
    
    # Check cache first
    cached_result = recipe_cache.get(cache_key)
    if cached_result:
        logger.info(f"Cache hit for ingredients: {', '.join(request.ingredients)}, page {page}")
        return cached_result
        
    # For local testing or when BigQuery is not available
    if client is None and os.environ.get("ENV") != "production":
        logger.info("Using mock data for recipe search (development mode)")
        # Return mock data with the searched ingredients
        result = {
            "recipes": [
                {
                    "title": f"Mock Recipe with {', '.join(request.ingredients[:2])}",
                    "ingredients": request.ingredients + ["salt", "pepper", "olive oil"],
                    "directions": ["Mix ingredients", "Cook for 20 minutes", "Serve hot"]
                },
                {
                    "title": f"Another Recipe with {request.ingredients[0]}",
                    "ingredients": [request.ingredients[0], "garlic", "onion", "butter"],
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
        for ing in request.ingredients:
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
        logger.info(f"SQL QUERY: ingredients={len(request.ingredients)}, page={page}, limit={effective_page_size}, offset={offset}, max_results={max_results}")
        
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
                logger.warning(f"WARNING: Truncating results at {ABSOLUTE_MAX_RESULTS}")
                break
                
            recipes.append({
                "title": row.title,
                "ingredients": row.ingredients,
                "directions": row.directions,
                "match_score": row.match_score
            })

        query_time = time.time() - start_time
        logger.info(f"Query executed in {query_time:.2f} seconds")
        
        # Track total results for rate limiting
        throttler.add_results(len(recipes))
        
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
        logger.error(f"Error in recipe search: {e}")
        # Return mock data if BigQuery query fails
        error_result = {
            "recipes": [
                {
                    "title": f"Mock Recipe with {', '.join(request.ingredients[:2])} (Error fallback)",
                    "ingredients": request.ingredients + ["salt", "pepper", "olive oil"],
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
        logger.error(f"Error analyzing image: {e}")
        # Return mock data for testing if analysis fails
        return {"ingredients": ["tomato", "onion", "garlic"], "error": str(e)}

@app.post("/search-by-image")
async def search_by_image(
    file: UploadFile = File(...), 
    max_results: int = 200,
    page: int = 1,
    page_size: int = 20,
    req: Request = None
):
    """
    Analyze a food image and search for recipes based on the detected ingredients.
    
    This combines the image analysis and recipe search in one convenient endpoint.
    """
    # Apply absolute hard limits
    page = max(1, min(100, page))
    page_size = max(1, min(20, page_size))
    max_results = min(ABSOLUTE_MAX_RESULTS, max_results)
    
    client_ip = req.state.client_ip if req else "unknown"
    logger.info(f"IMAGE SEARCH from {client_ip}: page={page}, page_size={page_size}, max_results={max_results}")
    
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
    search_result = await search_recipes(search_request, req)
    
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
        logger.error(f"Error in image similarity search: {e}")
        return {"error": str(e)}

@app.post("/image-to-recipe")
async def image_to_recipe(
    file: UploadFile = File(...), 
    max_results: int = 200,
    page: int = 1,
    page_size: int = 10,
    req: Request = None
):
    """
    A comprehensive endpoint that:
    1. Analyzes the food image to identify ingredients
    2. Searches for recipes based on the ingredients
    3. Finds visually similar recipes based on the image appearance
    
    Returns all three results combined.
    """
    # Apply absolute hard limits
    page = max(1, min(100, page))
    page_size = max(1, min(20, page_size))
    max_results = min(ABSOLUTE_MAX_RESULTS, max_results)
    
    client_ip = req.state.client_ip if req else "unknown"
    logger.info(f"IMAGE-TO-RECIPE from {client_ip}: page={page}, page_size={page_size}, max_results={max_results}")
    
    start_time = time.time()
    
    # First, analyze the image to get ingredients
    contents = await file.read()
    
    # Create a cache key for this image
    image_hash = hashlib.md5(contents).hexdigest()
    cache_key = f"img2recipe_{image_hash}_{max_results}_{page}_{page_size}"  # Include pagination in cache key
    
    # Check cache first
    cached_result = recipe_cache.get(cache_key)
    if cached_result:
        logger.info(f"Cache hit for image: {file.filename}")
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
    recipes_task = asyncio.create_task(search_recipes(search_request, req))
    
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

# Create a middleware to get client IP
@app.middleware("http")
async def get_client_ip(request: Request, call_next):
    # Get client IP
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0]
    else:
        client_ip = request.client.host
        
    # Check if request should be throttled
    if throttler.should_throttle(client_ip):
        return JSONResponse(
            status_code=429, 
            content={"detail": "Too many requests. Please try again later."}
        )
        
    # Add client IP to request state for use in endpoints
    request.state.client_ip = client_ip
    
    # Continue with the request
    response = await call_next(request)
    return response

@app.post("/batch-search")
async def batch_search_recipes(request: BatchSearchRequest, req: Request):
    """
    Process multiple search requests in a single API call to reduce frontend request volume.
    Each set of ingredients is processed as a separate search, but all within one request.
    """
    client_ip = req.state.client_ip
    logger.info(f"BATCH SEARCH from {client_ip}: {len(request.ingredientSets)} ingredient sets")
    
    # Apply global rate limiting
    if throttler.should_throttle(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please try again later."}
        )
    
    # Limit the number of searches in a batch to prevent abuse
    max_batch_size = 5
    if len(request.ingredientSets) > max_batch_size:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Maximum batch size is {max_batch_size} searches"}
        )
    
    results = {}
    
    # Process each ingredient set
    for idx, ingredients in enumerate(request.ingredientSets):
        # Skip empty ingredient sets
        if not ingredients:
            results[f"set_{idx}"] = {"recipes": [], "total": 0}
            continue
            
        # Create a search request for this ingredient set
        search_request = SearchRequest(
            ingredients=ingredients,
            page=request.page,
            page_size=request.page_size,
            min_matches=request.min_matches,
            max_results=request.max_results
        )
        
        # Use the existing search function but without additional throttling
        # We've already applied throttling at the batch level
        search_result = await search_recipes(search_request, req)
        
        # Add to results
        results[f"set_{idx}"] = search_result
    
    # Return all results
    return {
        "results": results,
        "count": len(request.ingredientSets)
    }

# Start the server for local testing
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
