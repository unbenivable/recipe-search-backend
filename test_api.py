#!/usr/bin/env python
"""
Recipe Search Backend API Test Script

This script demonstrates how to interact with the recipe search backend API
when it's running locally.
"""

import requests
import sys
import json
from pathlib import Path

# API URL - Change this if your server is running on a different port
API_URL = "http://localhost:8000"

def test_analyze_image(image_path):
    """Test the image analysis endpoint with a food image."""
    endpoint = f"{API_URL}/analyze-image"
    
    try:
        with open(image_path, "rb") as img_file:
            files = {"file": (Path(image_path).name, img_file, "image/jpeg")}
            
            print(f"Sending image to {endpoint}...")
            response = requests.post(endpoint, files=files)
            
            if response.status_code == 200:
                result = response.json()
                print("\nDetected ingredients:")
                print(", ".join(result.get("ingredients", [])))
                return result
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_similar_recipe_images(image_path):
    """Test the image similarity search endpoint with a food image."""
    endpoint = f"{API_URL}/similar-recipe-images"
    
    try:
        with open(image_path, "rb") as img_file:
            files = {"file": (Path(image_path).name, img_file, "image/jpeg")}
            
            print(f"Sending image to {endpoint}...")
            response = requests.post(endpoint, files=files)
            
            if response.status_code == 200:
                result = response.json()
                print("\nSimilar recipes:")
                for recipe in result.get("similar_recipes", []):
                    print(f"- {recipe.get('title')} (Score: {recipe.get('similarity_score')})")
                return result
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_comprehensive_search(image_path):
    """Test the comprehensive image-to-recipe endpoint."""
    endpoint = f"{API_URL}/image-to-recipe"
    
    try:
        with open(image_path, "rb") as img_file:
            files = {"file": (Path(image_path).name, img_file, "image/jpeg")}
            
            print(f"Sending image to {endpoint}...")
            response = requests.post(endpoint, files=files)
            
            if response.status_code == 200:
                result = response.json()
                
                print("\n--- Comprehensive Results ---")
                
                print("\nDetected ingredients:")
                print(", ".join(result.get("detected_ingredients", [])))
                
                print("\nRecipes by ingredients:")
                for recipe in result.get("recipes_by_ingredients", [])[:3]:  # Show only first 3
                    print(f"- {recipe.get('title')}")
                
                print("\nSimilar looking recipes:")
                for recipe in result.get("similar_looking_recipes", []):
                    print(f"- {recipe.get('title')} (Score: {recipe.get('similarity_score')})")
                
                return result
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <image_path> [endpoint]")
        print("Endpoints: analyze, similar, comprehensive (default)")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Default to comprehensive test if no endpoint specified
    endpoint = sys.argv[2] if len(sys.argv) > 2 else "comprehensive"
    
    if endpoint == "analyze":
        test_analyze_image(image_path)
    elif endpoint == "similar":
        test_similar_recipe_images(image_path)
    elif endpoint == "comprehensive":
        test_comprehensive_search(image_path)
    else:
        print(f"Unknown endpoint: {endpoint}")
        print("Valid endpoints: analyze, similar, comprehensive") 