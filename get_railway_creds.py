#!/usr/bin/env python
"""
Format GCP Credentials for Railway Deployment
This script reads a standard GCP JSON credentials file and formats it for Railway.
"""
import json
import sys

def format_creds_for_railway(filepath):
    try:
        # Read the credentials file
        with open(filepath, 'r') as f:
            creds = json.load(f)
            
        # Validate that it has the required fields
        required_fields = ["type", "project_id", "private_key_id", "private_key", "client_email"]
        missing = [field for field in required_fields if field not in creds]
        
        if missing:
            print(f"Error: Missing required fields in credentials file: {', '.join(missing)}")
            return None
            
        # Format the JSON as a single line string with proper escaping
        formatted_json = json.dumps(creds)
        
        return formatted_json
        
    except Exception as e:
        print(f"Error processing credentials file: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_railway_creds.py path/to/credentials.json")
        sys.exit(1)
        
    filepath = sys.argv[1]
    formatted = format_creds_for_railway(filepath)
    
    if formatted:
        print("\n===== RAILWAY CREDENTIALS VALUE =====")
        print(formatted)
        print("\n===== END CREDENTIALS =====")
        print("\nCopy the above value (without the === lines) into your Railway environment variable for GOOGLE_APPLICATION_CREDENTIALS_JSON")
        
        # Also save to a file for convenience
        with open("railway_creds.txt", "w") as f:
            f.write(formatted)
            
        print("\nCredentials also saved to railway_creds.txt")
        sys.exit(0)
    else:
        print("Failed to format credentials properly.")
        sys.exit(1) 