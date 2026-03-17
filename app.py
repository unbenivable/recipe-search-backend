# Recipe Search Backend API
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import base64
import uvicorn
import anthropic
import time
import hashlib
import asyncio
import datetime
from fastapi.responses import JSONResponse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
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

# Initialize Anthropic client
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

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
    page = max(1, min(100, request.page))
    page_size = max(1, min(20, request.page_size))

    client_ip = req.state.client_ip
    logger.info(f"SEARCH REQUEST from {client_ip}: ingredients={request.ingredients}, page={page}, page_size={page_size}")

    if not request.ingredients or any(len(ing.strip()) < 2 for ing in request.ingredients):
        return {
            "recipes": [],
            "total": 0,
            "page": page,
            "page_size": page_size,
            "pages": 0,
            "error": "Please enter ingredients with at least 2 characters each"
        }

    search_term = ",".join(sorted([ing.lower().strip() for ing in request.ingredients]))

    if throttler.should_throttle(client_ip, search_term):
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many similar requests. Please try again later."}
        )

    cache_params = {
        "ingredients": sorted([ing.lower().strip() for ing in request.ingredients]),
        "page": page
    }
    cache_key = hashlib.md5(json.dumps(cache_params).encode()).hexdigest()

    cached_result = recipe_cache.get(cache_key)
    if cached_result:
        logger.info(f"Cache hit for ingredients: {', '.join(request.ingredients)}, page {page}")
        return cached_result

    try:
        start_time = time.time()

        ingredients_str = ", ".join(request.ingredients)

        page_instruction = ""
        if page > 1:
            page_instruction = f" This is batch #{page} — generate completely different recipes from previous batches."

        prompt = (
            f"Generate exactly {page_size} unique recipes using some or all of these ingredients: {ingredients_str}.{page_instruction}\n\n"
            f"For each recipe return a JSON object with:\n"
            f'- "title": creative recipe name (string)\n'
            f'- "ingredients": complete ingredient list with quantities (array of strings)\n'
            f'- "directions": step-by-step cooking instructions (array of strings)\n\n'
            f"Respond ONLY with a valid JSON array of {page_size} recipe objects. No markdown, no explanation, no extra text."
        )

        message = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = message.content[0].text.strip()

        # Strip markdown code fences if present
        if response_text.startswith("```"):
            response_text = response_text.split("\n", 1)[1] if "\n" in response_text else response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3].strip()

        recipes_data = json.loads(response_text)

        if not isinstance(recipes_data, list):
            recipes_data = [recipes_data]

        recipes = []
        for r in recipes_data:
            recipes.append({
                "title": r.get("title", "Untitled Recipe"),
                "ingredients": r.get("ingredients", []),
                "directions": r.get("directions", [])
            })

        query_time = time.time() - start_time
        logger.info(f"Generated {len(recipes)} recipes in {query_time:.2f}s")

        throttler.add_results(len(recipes))

        total_pages = 5
        total_count = total_pages * page_size

        result = {
            "recipes": recipes,
            "query_time_seconds": query_time,
            "pagination": {
                "total": total_count,
                "page": page,
                "page_size": page_size,
                "pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1,
                "next_page": page + 1 if page < total_pages else None,
                "prev_page": page - 1 if page > 1 else None,
                "first_page": 1,
                "last_page": total_pages,
                "page_numbers": list(range(max(1, page - 2), min(total_pages + 1, page + 3)))
            }
        }

        recipe_cache.set(cache_key, result)
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse AI response as JSON: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to generate recipes. Please try again."}
        )
    except Exception as e:
        logger.error(f"Error generating recipes: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to generate recipes: {str(e)}"}
        )

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze a food image to extract ingredients using Anthropic Claude.

    Returns a list of identified ingredients that can be used for recipe search.
    """
    contents = await file.read()

    if not contents:
        return JSONResponse(
            status_code=400,
            content={"error": "Empty file uploaded"}
        )

    # Validate content type
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/gif"}
    content_type = file.content_type or ""
    if content_type not in allowed_types:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported image format: {content_type}. Use JPEG, PNG, or WebP."}
        )

    try:
        prompt = "List only the food ingredients you can see in this image, one per line, nothing else."
        image_b64 = base64.standard_b64encode(contents).decode("utf-8")

        message = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": content_type, "data": image_b64}},
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        ingredients_text = message.content[0].text.strip()

        # Parse line-separated list into individual ingredients
        ingredients = []
        for line in ingredients_text.split('\n'):
            # Strip whitespace, bullets, dashes, numbers, and asterisks
            cleaned = line.strip().lstrip('-*•').strip()
            # Remove leading numbers like "1. " or "1) "
            if cleaned and cleaned[0].isdigit():
                for sep in ['. ', ') ', ': ', ' ']:
                    idx = cleaned.find(sep)
                    if idx != -1 and idx < 4:
                        cleaned = cleaned[idx + len(sep):]
                        break
            cleaned = cleaned.strip()
            if cleaned and len(cleaned) > 1:
                ingredients.append(cleaned)

        if not ingredients:
            return JSONResponse(
                status_code=200,
                content={"ingredients": [], "message": "No food ingredients detected. Try a clearer photo of food items."}
            )

        # Cap at 15 ingredients
        return {"ingredients": ingredients[:15]}
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to analyze image: {str(e)}"}
        )

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

    # If analyze_image returned a JSONResponse (error), propagate it
    if isinstance(analysis_result, JSONResponse):
        return analysis_result

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

@app.post("/image-to-recipe")
async def image_to_recipe(
    file: UploadFile = File(...),
    max_results: int = 200,
    page: int = 1,
    page_size: int = 10,
    req: Request = None
):
    """
    Analyze a food image, identify ingredients, and search for recipes.
    """
    page = max(1, min(100, page))
    page_size = max(1, min(20, page_size))
    max_results = min(ABSOLUTE_MAX_RESULTS, max_results)

    client_ip = req.state.client_ip if req else "unknown"
    logger.info(f"IMAGE-TO-RECIPE from {client_ip}: page={page}, page_size={page_size}")

    # Analyze the image to get ingredients
    analysis_result = await analyze_image(file)

    if isinstance(analysis_result, JSONResponse):
        return analysis_result

    ingredients = analysis_result.get("ingredients", [])

    if not ingredients:
        return {
            "detected_ingredients": [],
            "recipes_by_ingredients": [],
            "pagination": {}
        }

    search_request = SearchRequest(
        ingredients=ingredients,
        page=page,
        page_size=page_size,
        min_matches=1,
        max_results=max_results
    )

    recipes_result = await search_recipes(search_request, req)

    return {
        "detected_ingredients": ingredients,
        "recipes_by_ingredients": recipes_result.get("recipes", []),
        "pagination": recipes_result.get("pagination", {})
    }

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
