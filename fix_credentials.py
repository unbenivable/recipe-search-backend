#!/usr/bin/env python
"""
Fix GCP Credentials Format
"""
import os
import json

# Get the raw credentials string from the environment
raw_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON", "{}")

# Remove wrapping quotes if present
if raw_creds.startswith("'") and raw_creds.endswith("'"):
    raw_creds = raw_creds[1:-1]
elif raw_creds.startswith('"') and raw_creds.endswith('"'):
    raw_creds = raw_creds[1:-1]

# Print diagnostic info
print(f"Raw credentials length: {len(raw_creds)}")
print(f"First 50 characters: {raw_creds[:50]}...")

# Create a properly formatted credential JSON
try:
    # Extract the components of the JSON manually
    # This is a workaround for the JSON parsing issue
    if "private_key" in raw_creds:
        # Split on known fields to extract parts
        parts = raw_creds.split(',"private_key":"')
        if len(parts) >= 2:
            prefix = parts[0]
            
            # Extract and fix the private key and the rest
            rest_parts = parts[1].split('","client_email":"')
            if len(rest_parts) >= 2:
                private_key = rest_parts[0]
                suffix = '"client_email":"' + rest_parts[1]
                
                # Replace \n with actual newlines in the private key
                if "\\n" not in private_key and "\n" in private_key:
                    private_key = private_key.replace("\n", "\\n")
                
                # Reconstruct the JSON with proper escaping
                fixed_json = prefix + ',"private_key":"' + private_key + '",' + suffix
                
                # Test if the resulting string is valid JSON
                try:
                    json_obj = json.loads(fixed_json)
                    print("\n✅ Successfully fixed and parsed credentials")
                    
                    # Write the fixed credentials to an environment file
                    with open(".env.fixed", "w") as f:
                        f.write(f'GOOGLE_APPLICATION_CREDENTIALS_JSON=\'{fixed_json}\'')
                    
                    print("✅ Saved fixed credentials to .env.fixed")
                    print("\nTo use the fixed credentials, run:")
                    print("set -a; source .env.fixed; set +a")
                    
                except json.JSONDecodeError as je:
                    print(f"❌ Fixed JSON is still invalid: {je}")
            else:
                print("❌ Could not split on client_email")
        else:
            print("❌ Could not split on private_key")
    else:
        print("❌ No private_key field found in credentials")

except Exception as e:
    print(f"❌ Error fixing credentials: {e}") 