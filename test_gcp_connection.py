#!/usr/bin/env python
"""
Test GCP Connection and Credential Parsing
"""
import os
import json
import sys
from google.cloud import bigquery
from google.oauth2 import service_account

def test_credentials():
    print("=== GCP Credentials Test ===")
    
    # Get credentials from environment
    print("Reading credentials from environment variable...")
    creds_str = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON", "{}")
    
    # Print length info
    print(f"Credentials string length: {len(creds_str)}")
    
    # Print beginning of the string (first 50 chars)
    print(f"First 50 chars: {creds_str[:50]}...")
    
    # Check for problematic character at position 136
    if len(creds_str) > 140:
        print(f"Characters around position 136-138: '{creds_str[135:140]}'")
    
    # Check if wrapped in quotes
    if creds_str.startswith("'") and creds_str.endswith("'"):
        print("Credentials wrapped in single quotes - removing")
        creds_str = creds_str[1:-1]
    elif creds_str.startswith('"') and creds_str.endswith('"'):
        print("Credentials wrapped in double quotes - removing")
        creds_str = creds_str[1:-1]
    
    # Try to find and fix JSON issues
    print("\nAttempting to parse JSON...")
    try:
        # Basic cleanup - handle escaped newlines
        creds_str = creds_str.replace('\\n', '\\\\n')
        
        # Try to parse
        credentials_info = json.loads(creds_str)
        print("✅ Successfully parsed JSON credentials")
        
        # Verify required fields
        required_fields = ["type", "project_id", "private_key", "client_email"]
        for field in required_fields:
            if field in credentials_info:
                print(f"✅ Found required field: {field}")
            else:
                print(f"❌ Missing required field: {field}")
        
        # Try to connect to BigQuery
        print("\nAttempting to connect to BigQuery...")
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        client = bigquery.Client(credentials=credentials, project=credentials_info["project_id"])
        
        # Run a simple query
        query = "SELECT 1"
        try:
            result = client.query(query).result()
            print("✅ Successfully connected to BigQuery and ran a test query")
            return True
        except Exception as e:
            print(f"❌ Query execution failed: {e}")
            return False
            
    except json.JSONDecodeError as je:
        print(f"❌ JSON parse error: {je}")
        print(f"  Position: {je.pos}, Line: {je.lineno}, Column: {je.colno}")
        
        # Show the problematic part
        start = max(0, je.pos - 10)
        end = min(len(creds_str), je.pos + 10)
        print(f"  Problem area: '...{creds_str[start:je.pos]}➡️{creds_str[je.pos:end]}...'")
        
        # Try to fix common problems
        print("\nAttempting to fix common JSON issues...")
        
        # Fix 1: Try replacing single quotes with double quotes for JSON
        fixed_creds = creds_str.replace("'", '"')
        try:
            json.loads(fixed_creds)
            print("✅ Fixed by replacing single quotes")
            return True
        except:
            pass
            
        # Fix 2: Try handling escaped newlines in private key
        if "private_key" in creds_str:
            try:
                # Save problematic credentials to a file for inspection
                with open("creds_debug.txt", "w") as f:
                    f.write(creds_str[:200] + "... (truncated)")
                print("Saved first 200 chars of credentials to creds_debug.txt for inspection")
            except:
                pass
        
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    if test_credentials():
        print("\n✅ GCP connection test PASSED")
        sys.exit(0)
    else:
        print("\n❌ GCP connection test FAILED")
        sys.exit(1) 