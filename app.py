from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import json
import base64
import sys
from google.cloud import bigquery
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, Part

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
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    
    # Initialize Vertex AI
    vertexai.init(
        project=credentials.project_id,
        location="us-central1",
        credentials=credentials
    )
    
    print(f"Successfully initialized Google Cloud clients for project: {credentials.project_id}")
except Exception as e:
    print(f"Warning: Unable to initialize Google Cloud clients: {e}")
    print("Using mock data for local testing")

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

# Start the server for local testing
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
