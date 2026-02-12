import os
import sys
import re
import json
import hashlib
import requests
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util

# ========================================
# Environment setup
# ========================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ========================================
# MongoDB setup
# ========================================
client = MongoClient("mongodb://localhost:27017/")
db = client["AISubmodule"]
users_collection = db["Users"]
collection = db["SameRolesCache"]

# ========================================
# Load Sentence Transformer
# ========================================
model = SentenceTransformer('all-MiniLM-L6-v2')

# API Key for OpenRouter
api_key = "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"

# ========================================
# Helper functions
# ========================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def signup(username=None):
    """Signup new user"""
    if not username:
        username = input("Enter a new username: ").strip()
    password = input("Enter a password: ").strip()
    hashed_pw = hash_password(password)

    users_collection.insert_one({
        "username": username,
        "password": hashed_pw,
        "projects": []  # store multiple project details
    })
    print(f"✅ Signup successful. Welcome, {username}!")
    return {"username": username, "password": hashed_pw}

def login():
    """Login existing user, signup if not found"""
    username = input("Username: ").strip()
    user = users_collection.find_one({"username": username})

    if not user:
        print("⚠️ No account found. Let's create one for you.")
        return signup(username)

    password = input("Password: ").strip()
    if user["password"] != hash_password(password):
        print("❌ Incorrect password.")
        return None
    print(f"✅ Welcome back, {username}!")
    return user

def parse_api_response(response_text):
    """Parse AI API response"""
    lines = response_text.strip().split('\n')
    project_type = ""
    team_size = 0
    roles = []

    for line in lines:
        line = line.strip()
        if line.startswith("Project Type:"):
            project_type = line.replace("Project Type:", "").strip()
        elif line.startswith("Team Size:"):
            team_size_str = line.replace("Team Size:", "").strip()
            team_size = int(re.findall(r'\d+', team_size_str)[0])
        elif " - Experienced - " in line or " - NewHire - " in line:
            if " - Experienced - " in line:
                parts = line.split(" - Experienced - ")
                experience_level = "Experienced"
            else:
                parts = line.split(" - NewHire - ")
                experience_level = "NewHire"

            role_name = parts[0].strip()
            role_name = re.sub(r'^[-•\d.\s]+', '', role_name).strip()
            skills = [skill.strip() for skill in parts[1].split(',')]

            roles.append({
                "role": role_name,
                "experience_level": experience_level,
                "skills": skills
            })

    return {
        "project_type": project_type,
        "team_size": team_size,
        "roles": roles
    }

# ========================================
# AUTHENTICATION FLOW
# ========================================
user = login()
if not user:
    sys.exit("❌ Authentication failed. Exiting.")

# ========================================
# PROJECT INPUT (Linked to user)
# ========================================
project_title = input("Enter the project title: ").strip()
project_description = input("Enter the project description: ").strip()

# Store project details inside Users collection under this username
users_collection.update_one(
    {"username": user["username"]},
    {"$push": {"projects": {
        "project_title": project_title,
        "project_description": project_description
    }}}
)

# ========================================
# MAIN AI MODULE LOGIC
# ========================================
new_embedding = model.encode(project_description, convert_to_tensor=True)

# Try exact match in cache for ANY user first
cached_entry = collection.find_one({
    "project_title": project_title,
    "project_description": project_description
})

# Try semantic similarity across all users if exact match not found
if not cached_entry:
    best_match = None
    max_sim = 0.0
    for doc in collection.find({"project_title": project_title}):
        existing_embedding = model.encode(doc["project_description"], convert_to_tensor=True)
        sim = util.pytorch_cos_sim(new_embedding, existing_embedding).item()
        if sim > 0.80 and sim > max_sim:
            max_sim = sim
            best_match = doc
    cached_entry = best_match

if cached_entry:
    print("\n[From Cache] (Found from another user)")
    print(f"Project Type: {cached_entry['project_type']}")
    print(f"Team Size: {cached_entry['team_size']}")
    print("\nRoles:")

    exp_count = 0
    new_count = 0
    for role in cached_entry['roles']:
        print(f"- {role['role']} ({role['experience_level']})")
        print(f"  Skills: {', '.join(role['skills'])}")
        if role['experience_level'] == 'Experienced':
            exp_count += 1
        else:
            new_count += 1
    print(f"\nExperienced: {exp_count}")
    print(f"NewlyHired: {new_count}")

    # Also store it under the current user's name so they have a copy
    if not collection.find_one({
        "username": user["username"],
        "project_title": project_title,
        "project_description": project_description
    }):
        collection.insert_one({
            "username": user["username"],
            "project_title": cached_entry["project_title"],
            "project_description": cached_entry["project_description"],
            "project_type": cached_entry["project_type"],
            "team_size": cached_entry["team_size"],
            "roles": cached_entry["roles"]
        })

else:
    print("\n[Calling OpenRouter API...]")
    # (keep your existing API call + parsing code here)

    prompt = f"""
You are an intelligent AI assistant. Given a software project title and description, your task is to suggest a software project type, team size, and a list of roles.

Strictly follow this structure and constraints:

Project Type: <value>
Team Size: <exact number between 4–10>  

Roles:(excluding Project Manager or Team Lead)
Each role must follow this structure:
Avoid repeating the same role at both Experienced and NewHire levels unless clearly necessary for project scope diversity.

<Role> - Experience level - <comma-separated skills>  

SKILL GUIDELINES:
- Skills should be specific but not overly technical or niche
- For EXPERIENCED roles, include advanced/specialized skills
- For NEWHIRE roles, include fundamental/basic skills
- Use full names, not abbreviations (e.g., "JavaScript" not "JS", "PostgreSQL" not "Postgres")
- Skills should match the experience level appropriately

Total number of role entries (Experienced + NewHire) must be equal to the Team Size. Do not exceed this number.

Now process the following:

Project Title: {project_title}  
Project Description: {project_description}
"""

    res = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "mistralai/mistral-7b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
    )

    result = res.json()["choices"][0]["message"]["content"].strip()
    print("Raw API Response:")
    print(result)

    structured_data = parse_api_response(result)

    # Save AI results in SameRolesCache with username
    document = {
        "username": user["username"],
        "project_title": project_title,
        "project_description": project_description,
        "project_type": structured_data["project_type"],
        "team_size": structured_data["team_size"],
        "roles": structured_data["roles"],
    }
    collection.insert_one(document)

    print(f"\n[Structured Output]")
    print(f"Project Type: {structured_data['project_type']}")
    print(f"Team Size: {structured_data['team_size']}")
    print("\nRoles:")

    exp_count = 0
    new_count = 0
    for role in structured_data['roles']:
        print(f"- {role['role']} ({role['experience_level']})")
        print(f"  Skills: {', '.join(role['skills'])}")
        if role['experience_level'] == 'Experienced':
            exp_count += 1
        else:
            new_count += 1
    print(f"\nExperienced: {exp_count}")
    print(f"NewlyHired: {new_count}")