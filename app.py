# Recipe Search Backend API - With Google Vertex AI
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import base64
import sys
from google.cloud import bigquery
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.vision_models import Image
from vertexai.vision_models import MultiModalEmbeddingModel
from io import BytesIO
#
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
    credentials_info = json.loads(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON", "{}"))
    
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

@app.post("/search")
async def search_recipes(request: Request):
    data = await request.json()
    user_ingredients = [ing.lower().strip() for ing in data.get("ingredients", [])]

    if not user_ingredients:
        return {"recipes": []}

    # Build clause for each ingredient
    match_clauses = [
        f"""EXISTS (
            SELECT 1 FROM UNNEST(ingredients) AS ing
            WHERE LOWER(ing) LIKE '%{ing}%'
        )""" for ing in user_ingredients
    ]

    score_expression = " + ".join([f"CASE WHEN {clause} THEN 1 ELSE 0 END" for clause in match_clauses])

    query = f"""
        SELECT title, ingredients, directions
        FROM `recipe-data-pipeline.recipes.structured_recipes`
        WHERE ({score_expression}) >= 3
        LIMIT 20
    """

    results = client.query(query).result()

    recipes = []
    for row in results:
        recipes.append({
            "title": row.title,
            "ingredients": row.ingredients,
            "directions": row.directions
        })

    return {"recipes": recipes}

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
async def search_by_image(file: UploadFile = File(...)):
    """
    Analyze a food image and search for recipes based on the detected ingredients.
    
    This combines the image analysis and recipe search in one convenient endpoint.
    """
    # First, analyze the image to get ingredients
    analysis_result = await analyze_image(file)
    ingredients = analysis_result.get("ingredients", [])
    
    if not ingredients:
        return {"recipes": [], "detected_ingredients": []}
    
    # For local testing without BigQuery
    if client is None:
        print("Using mock data for recipe search since BigQuery is not available")
        return {
            "recipes": [
                {
                    "title": "Sample Recipe with " + ", ".join(ingredients[:2]),
                    "ingredients": ingredients + ["salt", "pepper", "olive oil"],
                    "directions": ["Mix ingredients", "Cook for 20 minutes", "Serve hot"]
                }
            ],
            "detected_ingredients": ingredients
        }
    
    # Then use those ingredients to search for recipes
    ingredient_conditions = [
        f"LOWER(ARRAY_TO_STRING(ingredients, ',')) LIKE '%{ingredient.lower()}%'"
        for ingredient in ingredients
    ]
    
    # Require at least 2 matching ingredients
    condition_query = " + ".join(
        [f"CASE WHEN {cond} THEN 1 ELSE 0 END" for cond in ingredient_conditions]
    )

    query = f"""
        SELECT title, ingredients, directions
        FROM `recipe-data-pipeline.recipes.structured_recipes`
        WHERE ({condition_query}) >= 2
        LIMIT 20
    """

    results = client.query(query).result()

    recipes = []
    for row in results:
        recipes.append({
            "title": row.title,
            "ingredients": row.ingredients,
            "directions": row.directions,
        })

    return {"recipes": recipes, "detected_ingredients": ingredients}

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
async def image_to_recipe(file: UploadFile = File(...)):
    """
    A comprehensive endpoint that:
    1. Analyzes the food image to identify ingredients
    2. Searches for recipes based on the ingredients
    3. Finds visually similar recipes based on the image appearance
    
    Returns all three results combined.
    """
    # First, analyze the image to get ingredients
    analysis_result = await analyze_image(file)
    ingredients = analysis_result.get("ingredients", [])
    
    # Rewind the file after reading it in analyze_image
    await file.seek(0)
    
    # Find visually similar recipes
    similar_result = await search_similar_recipe_images(file)
    similar_recipes = similar_result.get("similar_recipes", [])
    
    # Get recipes by ingredients
    recipes_by_ingredients = []
    if client is not None and ingredients:
        # Build query similar to search_by_image function
        ingredient_conditions = [
            f"LOWER(ARRAY_TO_STRING(ingredients, ',')) LIKE '%{ingredient.lower()}%'"
            for ingredient in ingredients
        ]
        
        condition_query = " + ".join(
            [f"CASE WHEN {cond} THEN 1 ELSE 0 END" for cond in ingredient_conditions]
        )

        query = f"""
            SELECT title, ingredients, directions
            FROM `recipe-data-pipeline.recipes.structured_recipes`
            WHERE ({condition_query}) >= 2
            LIMIT 10
        """

        results = client.query(query).result()

        for row in results:
            recipes_by_ingredients.append({
                "title": row.title,
                "ingredients": row.ingredients,
                "directions": row.directions,
            })
    
    return {
        "detected_ingredients": ingredients,
        "recipes_by_ingredients": recipes_by_ingredients,
        "similar_looking_recipes": similar_recipes
    }

# Start the server for local testing
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
