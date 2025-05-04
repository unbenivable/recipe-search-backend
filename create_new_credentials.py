#!/usr/bin/env python
"""
Create New GCP Credentials
This script helps you create a properly formatted GCP credentials file.
"""
import json
import os

# Template for a GCP service account credentials
template = {
    "type": "service_account",
    "project_id": "recipe-data-pipeline",  # Your project
    "private_key_id": "",  # Will be prompted
    "private_key": "",     # Will be prompted
    "client_email": "",    # Will be prompted
    "client_id": "",       # Will be prompted
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "",  # Will be generated
    "universe_domain": "googleapis.com"
}

def get_input(prompt, field_name, required=True):
    """Get user input for a field with validation"""
    while True:
        value = input(prompt)
        if not value and required:
            print(f"Error: {field_name} is required. Please enter a value.")
        else:
            return value

print("=== Create New GCP Credentials ===")
print("This script will help you create a properly formatted GCP credentials file.")
print("You'll need information from your service account key file.\n")

# Get values for required fields
project_id = get_input("Project ID [recipe-data-pipeline]: ", "project_id") or "recipe-data-pipeline"
template["project_id"] = project_id

private_key_id = get_input("Private Key ID: ", "private_key_id")
template["private_key_id"] = private_key_id

# Get the private key with special handling for multi-line input
print("\nPaste your private key (starts with '-----BEGIN PRIVATE KEY-----')")
print("Press Enter twice when done:")
private_key_lines = []
while True:
    line = input()
    if not line and private_key_lines and not private_key_lines[-1]:
        break  # Two empty lines in a row means we're done
    private_key_lines.append(line)

# Join lines and add proper newline escaping for JSON
private_key = "\\n".join([line for line in private_key_lines if line]) + "\\n"
template["private_key"] = private_key

client_email = get_input("Client Email (usually ends with .iam.gserviceaccount.com): ", "client_email")
template["client_email"] = client_email

client_id = get_input("Client ID: ", "client_id")
template["client_id"] = client_id

# Generate the client_x509_cert_url
template["client_x509_cert_url"] = f"https://www.googleapis.com/robot/v1/metadata/x509/{client_email.replace('@', '%40')}"

# Generate the new credentials JSON
credentials_json = json.dumps(template)

# Write to a new .env file
with open(".env.new", "w") as f:
    f.write(f'GOOGLE_APPLICATION_CREDENTIALS_JSON=\'{credentials_json}\'')

print("\n✅ Credentials created successfully!")
print("✅ Saved to .env.new file")
print("\nTo use these credentials, run:")
print("set -a; source .env.new; set +a")
print("\nYou can also copy this value to your Railway environment variables:")
print(credentials_json)

# Test if the JSON is valid
try:
    json.loads(credentials_json)
    print("\n✅ Verified: Valid JSON format")
except json.JSONDecodeError as e:
    print(f"\n❌ ERROR: Invalid JSON format: {e}") 