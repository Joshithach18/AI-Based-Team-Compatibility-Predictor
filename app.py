from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from pymongo import MongoClient
import hashlib
import os
import sys
import re
import json
import requests
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import itertools
from collections import defaultdict
import heapq
import time
import subprocess
import tempfile
import pickle
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['TRANSFORMERS_NO_TF'] = '1'

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'your-secret-key-here-change-this'
CORS(app)

# MongoDB Setup
client = MongoClient("mongodb://localhost:27017/")
db = client["AISubmodule"]
users_collection = db["Users"]
collection = db["SameRolesCache"]
top_scores_col = db["TopTechnicalScores"]
behavior_col = db["BehavioralScores"]
same_roles_col = db["SameRolesCache"]
compatible_col = db["CompatibleTeams"]
user_teams_collection = db["UserTeams"]
project_slack_channels_collection = db["ProjectSlackChannels"]
slack_messages_collection = db["SlackMessages"]
team_analytics_collection = db["TeamAnalytics"]

# Slack Bot Token - Update this with your actual token
SLACK_BOT_TOKEN = "xoxb-10313122964370-10341631935168-B02UyxaXrHvVqIr4FlZbZSOX"
def get_available_slack_channels():
    """
    Fetch all Slack channels that the bot has access to
    Filters out channels that are already linked to other teams
    Also filters out default/common workspace channels
    Returns: List of available channels
    """
    url = "https://slack.com/api/conversations.list"
    headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
    params = {
        "types": "public_channel,private_channel",
        "exclude_archived": True,
        "limit": 200
    }
    
    # Default channels to exclude (common workspace channels)
    DEFAULT_CHANNELS = {
        'general', 'random', 'social', 'new-channel', 
        'all-compatibleteams', 'announcements', 'watercooler'
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()
        
        if data.get("ok") and "channels" in data:
            all_channels = data["channels"]
            
            # Get all already-linked channel IDs from ProjectSlackChannels collection
            linked_channels = set()
            for link in project_slack_channels_collection.find({}, {"channel_id": 1}):
                linked_channels.add(link["channel_id"])
            
            # Filter out already-linked channels and default channels
            available_channels = []
            for channel in all_channels:
                channel_id = channel["id"]
                channel_name = channel.get("name", "").lower()
                
                # Skip if already linked to another team
                if channel_id in linked_channels:
                    continue
                
                # Skip default/common workspace channels
                if channel_name in DEFAULT_CHANNELS:
                    continue
                
                available_channels.append({
                    "id": channel_id,
                    "name": channel.get("name", ""),
                    "is_private": channel.get("is_private", False),
                    "is_member": channel.get("is_member", False)
                })
            
            return available_channels
        else:
            print(f"❌ Failed to fetch channels: {data.get('error', 'Unknown error')}")
            return []
            
    except Exception as e:
        print(f"❌ Error fetching Slack channels: {str(e)}")
        return []
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Warning: Could not load SentenceTransformer: {e}")
    model = None

API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "mistralai/mistral-7b-instruct"

# Behavioral scoring model configuration
TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
MODEL_PATH = "behavioral_model.pkl"
_model_cache = None

def load_behavioral_model():
    """Load trained behavioral model from disk (cached)."""
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Please train the model first using: python behavioral_score.py train")
        with open(MODEL_PATH, "rb") as f:
            _model_cache = pickle.load(f)
        print(f"✅ Behavioral model loaded successfully")
    return _model_cache["embedder"], _model_cache["regressor"], _model_cache["traits"]

def get_personality_scores_fast(summary: str, embedder, regressor) -> dict:
    """Predict Big Five scores using the trained model (no API call)."""
    # Generate embedding
    emb = embedder.encode([summary], convert_to_numpy=True)
    
    # Predict scores
    scores_array = regressor.predict(emb)[0]
    
    # Clip to [0, 1] and convert to dict
    scores = {}
    for i, trait in enumerate(TRAITS):
        scores[trait] = float(np.clip(scores_array[i], 0, 1))
    
    return scores

def create_fallback_candidate(c):
    """Create candidate with default behavioral scores."""
    default_scores = {
        "Openness": 0.7,
        "Conscientiousness": 0.75,
        "Extraversion": 0.6,
        "Agreeableness": 0.7,
        "Neuroticism": 0.3
    }
    neuro_adj = 1 - default_scores["Neuroticism"]
    overall_score = (default_scores["Openness"] + default_scores["Conscientiousness"] + 
                    default_scores["Extraversion"] + default_scores["Agreeableness"] + neuro_adj) / 5
    
    return {
        "name": c["name"],
        "role": c["role"],
        "experience_level": c.get("experience_level", "Any"),
        "employee_role": c.get("employee_role", ""),
        "technical_score": c.get("technical_score", 60),
        "experience": c.get("experience", 2),
        "personality": {
            "openness": round(default_scores["Openness"] * 5, 1),
            "conscientiousness": round(default_scores["Conscientiousness"] * 5, 1),
            "extraversion": round(default_scores["Extraversion"] * 5, 1),
            "agreeableness": round(default_scores["Agreeableness"] * 5, 1),
            "neuroticism": round(default_scores["Neuroticism"] * 5, 1)
        },
        "scores": default_scores,
        "behavioral_score": round(overall_score * 100, 1)
    }

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def parse_api_response(response_text):
    print(f"Raw API Response for parsing:\n{response_text}\n")
    
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
            team_size_match = re.findall(r'\d+', team_size_str)
            if team_size_match:
                team_size = int(team_size_match[0])
        elif " - Experienced - " in line or " - NewHire - " in line:
            if " - Experienced - " in line:
                parts = line.split(" - Experienced - ")
                experience_level = "Experienced"
            else:
                parts = line.split(" - NewHire - ")
                experience_level = "NewHire"

            role_name = parts[0].strip()
            role_name = re.sub(r'^[-ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢\d.\s]+', '', role_name).strip()
            
            if len(parts) > 1:
                skills_text = parts[1].strip()
                skills = [skill.strip() for skill in skills_text.split(',') if skill.strip()]
            else:
                skills = []

            if role_name:
                roles.append({
                    "role": role_name,
                    "experience_level": experience_level,
                    "skills": skills
                })

    print(f"Parsed data:")
    print(f"Project Type: {project_type}")
    print(f"Team Size: {team_size}")
    print(f"Roles: {json.dumps(roles, indent=2)}")

    return {
        "project_type": project_type,
        "team_size": team_size,
        "roles": roles
    }

# ============================================================================
# PHASE 1: ROUTING - Base Pages
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/team-formation')
def team_formation():
    return render_template('team-formation.html')

@app.route('/monitor')
def monitor():
    """Phase 2: Team monitoring page with Slack integration"""
    return render_template('monitor.html')

# ============================================================================
# PHASE 1: AUTHENTICATION
# ============================================================================

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    
    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password required'})
    
    user = users_collection.find_one({"username": username})
    
    if not user:
        hashed_pw = hash_password(password)
        users_collection.insert_one({
            "username": username,
            "password": hashed_pw,
            "projects": []
        })
        session['username'] = username
        return jsonify({'success': True, 'message': 'Account created successfully'})
    
    if user["password"] != hash_password(password):
        return jsonify({'success': False, 'message': 'Incorrect password'})
    
    session['username'] = username
    return jsonify({'success': True, 'message': 'Login successful'})

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    
    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password required'})
    
    if users_collection.find_one({"username": username}):
        return jsonify({'success': False, 'message': 'Username already exists'})
    
    hashed_pw = hash_password(password)
    users_collection.insert_one({
        "username": username,
        "password": hashed_pw,
        "projects": []
    })
    
    session['username'] = username
    return jsonify({'success': True, 'message': 'Account created successfully'})

# ============================================================================
# PHASE 1: PROJECT REQUIREMENTS (AI-Powered)
# ============================================================================

@app.route('/api/project-requirements', methods=['POST'])
def project_requirements():
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})
    
    data = request.json
    project_title = data.get('projectTitle', '').strip()
    project_description = data.get('projectDescription', '').strip()
    
    if not project_title or not project_description:
        return jsonify({'success': False, 'message': 'Project title and description are required'})
    
    try:
        prompt = f"""
You are an expert project manager. Based on the following project description, suggest the ideal team composition.

Project Title: {project_title}
Project Description: {project_description}

Provide your response in EXACTLY this format:
Project Type: [type]
Team Size: [number]
[Role Name] - [Experienced/NewHire] - [comma-separated skills]

Example:
Project Type: Web Development
Team Size: 4
Frontend Developer - Experienced - React, CSS, HTML
Backend Developer - Experienced - Node.js, MongoDB
UI/UX Designer - NewHire - Figma, Adobe XD
"""
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        api_response = response.json()
        ai_text = api_response['choices'][0]['message']['content']
        
        parsed_data = parse_api_response(ai_text)
        
        parsed_data['projectTitle'] = project_title
        parsed_data['projectDescription'] = project_description
        
        return jsonify({'success': True, 'data': parsed_data})
        
    except Exception as e:
        print(f"Error in project_requirements: {str(e)}")
        return jsonify({'success': False, 'message': f'Error processing request: {str(e)}'})

# ============================================================================
# PHASE 1: TECHNICAL SCORING
# ============================================================================

@app.route('/api/technical-scores', methods=['POST'])
def technical_scores():
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})
    
    data = request.json
    project_title = data.get('projectTitle')
    roles = data.get('roles', [])
    
    if not project_title or not roles:
        return jsonify({'success': False, 'message': 'Invalid data'})
    
    try:
        # Save role requirements
        collection.delete_many({
            "username": session['username'],
            "project_title": project_title
        })
        collection.insert_one({
            "username": session['username'],
            "project_title": project_title,
            "roles": roles
        })
        
        # Find matching candidates from database
        csv_path = "UpdatedDataset.csv"
        if not os.path.exists(csv_path):
            return jsonify({'success': False, 'message': 'Dataset not found'})
        
        df = pd.read_csv(csv_path)
        
        all_candidates = []
        
        for role_spec in roles:
            required_role = role_spec["role"]
            required_exp_level = role_spec.get("experience_level", "Any")
            required_skills = [s.lower().strip() for s in role_spec.get("skills", [])]
            
            # Filter candidates
            role_matches = df[df['Employee_Role'].str.lower() == required_role.lower()]
            
            if required_exp_level != "Any":
                role_matches = role_matches[role_matches['Experience_Level'] == required_exp_level]
            
            for _, row in role_matches.iterrows():
                candidate_skills = [s.lower().strip() for s in str(row['Skills']).split(',')]
                
                # Calculate skill match score
                if required_skills:
                    matched = sum(1 for s in required_skills if any(s in cs for cs in candidate_skills))
                    skill_score = (matched / len(required_skills)) * 100
                else:
                    skill_score = 80
                
                # Technical score based on experience and skill match
                experience_score = min(float(row['Experience_Years']) * 10, 100)
                technical_score = (skill_score * 0.6) + (experience_score * 0.4)
                
                candidate = {
                    "name": row['Employee_Name'],
                    "role": row['Employee_Role'],
                    "experience_level": row['Experience_Level'],
                    "skills": candidate_skills,
                    "experience": int(row['Experience_Years']),
                    "technical_score": round(technical_score, 1),
                    "employee_role": row['Employee_Role']
                }
                
                all_candidates.append(candidate)
        
        # Save candidates
        top_scores_col.delete_many({
            "username": session['username'],
            "project_title": project_title
        })
        top_scores_col.insert_one({
            "username": session['username'],
            "project_title": project_title,
            "candidates": all_candidates
        })
        
        return jsonify({'success': True, 'data': all_candidates})
        
    except Exception as e:
        print(f"Error in technical_scores: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

# ============================================================================
# PHASE 1: BEHAVIORAL SCORING
# ============================================================================

@app.route('/api/behavioral-scores', methods=['POST'])
def behavioral_scores():
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})
    
    data = request.json
    project_title = data.get('projectTitle')
    
    if not project_title:
        return jsonify({'success': False, 'message': 'Project title required'})
    
    try:
        # Load behavioral model
        try:
            embedder, regressor, _ = load_behavioral_model()
            print("Using trained behavioral model")
        except FileNotFoundError:
            print("Behavioral model not found, using fallback scores")
            embedder = None
            regressor = None
        
        # Get candidates from technical scoring
        tech_doc = top_scores_col.find_one({
            "username": session['username'],
            "project_title": project_title
        })
        
        if not tech_doc:
            return jsonify({'success': False, 'message': 'No technical scores found. Please run technical scoring first.'})
        
        candidates = tech_doc["candidates"]
        
        # Load dataset for behavioral data
        csv_path = "UpdatedDataset.csv"
        if not os.path.exists(csv_path):
            return jsonify({'success': False, 'message': 'Dataset not found'})
        
        df = pd.read_csv(csv_path)
        
        enriched_candidates = []
        
        for c in candidates:
            # Find matching row in dataset
            match = df[
                (df['Employee_Name'] == c['name']) & 
                (df['Employee_Role'].str.lower() == c['role'].lower())
            ]
            
            if not match.empty:
                row = match.iloc[0]
                
                # Get behavioral summary
                summary = str(row.get('Combined_Behavioral_Summary', ''))
                
                if embedder and regressor and summary:
                    # Use trained model
                    scores = get_personality_scores_fast(summary, embedder, regressor)
                else:
                    # Use fallback
                    fallback = create_fallback_candidate(c)
                    scores = fallback["scores"]
                
                # Calculate behavioral score
                neuro_adj = 1 - scores["Neuroticism"]
                behavioral_score = (
                    scores["Openness"] + 
                    scores["Conscientiousness"] + 
                    scores["Extraversion"] + 
                    scores["Agreeableness"] + 
                    neuro_adj
                ) / 5
                
                enriched_candidate = {
                    **c,
                    "personality": {
                        "openness": round(scores["Openness"] * 5, 1),
                        "conscientiousness": round(scores["Conscientiousness"] * 5, 1),
                        "extraversion": round(scores["Extraversion"] * 5, 1),
                        "agreeableness": round(scores["Agreeableness"] * 5, 1),
                        "neuroticism": round(scores["Neuroticism"] * 5, 1)
                    },
                    "scores": scores,
                    "behavioral_score": round(behavioral_score * 100, 1)
                }
            else:
                # No match found, use fallback
                enriched_candidate = create_fallback_candidate(c)
            
            enriched_candidates.append(enriched_candidate)
        
        # Save enriched candidates
        behavior_col.delete_many({
            "username": session['username'],
            "project_title": project_title
        })
        behavior_col.insert_one({
            "username": session['username'],
            "project_title": project_title,
            "candidates": enriched_candidates
        })
        
        return jsonify({'success': True, 'data': enriched_candidates})
        
    except Exception as e:
        print(f"Error in behavioral_scores: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

# ============================================================================
# PHASE 1: TEAM FORMATION (Compatibility Algorithm)
# ============================================================================

def compute_personality_similarity(p1_scores, p2_scores):
    """Compute cosine similarity between two personality score dictionaries."""
    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    
    v1 = np.array([p1_scores.get(t, 0.5) for t in traits])
    v2 = np.array([p2_scores.get(t, 0.5) for t in traits])
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.5
    
    return float(np.dot(v1, v2) / (norm1 * norm2))

def compute_team_compatibility(team):
    """Calculate overall team compatibility score."""
    if len(team) < 2:
        return 0
    
    # Technical score average
    technical_avg = sum(m.get("technical_score", 60) for m in team) / len(team)
    
    # Behavioral score average
    behavioral_avg = sum(m.get("behavioral_score", 60) for m in team) / len(team)
    
    # Pairwise personality compatibility
    similarities = []
    for i in range(len(team)):
        for j in range(i + 1, len(team)):
            p1 = team[i].get("scores", {})
            p2 = team[j].get("scores", {})
            sim = compute_personality_similarity(p1, p2)
            similarities.append(sim)
    
    personality_compatibility = (sum(similarities) / len(similarities)) * 100 if similarities else 60
    
    # Weighted final score
    final_score = (
        technical_avg * 0.35 +
        behavioral_avg * 0.35 +
        personality_compatibility * 0.30
    )
    
    return final_score

def greedy_beam_search(role_candidates, required_roles, beam_width=5):
    """Greedy beam search for optimal team formation."""
    if len(required_roles) < 2:
        return []
    
    # Initialize beams with first role candidates
    first_role = required_roles[0]
    current_beams = [(0, [c], {first_role: 1}) for c in role_candidates[first_role][:beam_width]]
    
    # Iteratively add members from remaining roles
    for role in required_roles[1:]:
        next_beams = []
        
        for _, team, used_counts in current_beams:
            for candidate in role_candidates[role]:
                # Skip if candidate already in team
                if any(m["name"] == candidate["name"] for m in team):
                    continue
                
                new_team = team + [candidate]
                score = compute_team_compatibility(new_team)
                new_counts = used_counts.copy()
                new_counts[role] = new_counts.get(role, 0) + 1
                
                next_beams.append((score, new_team, new_counts))
        
        # Keep top beam_width beams
        next_beams.sort(key=lambda x: x[0], reverse=True)
        current_beams = next_beams[:beam_width]
        
        if not current_beams:
            break
    
    return [(score, team) for score, team, _ in current_beams]

@app.route('/api/form-teams', methods=['POST'])
def form_teams():
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})
    
    data = request.json
    project_title = data.get('projectTitle', '').strip()
    
    try:
        project_doc = behavior_col.find_one({
            "username": session['username'],
            "project_title": project_title
        })
        if not project_doc:
            return jsonify({'success': False, 'message': 'No behavioral scores found. Please calculate behavioral scores first.'})

        candidates = project_doc["candidates"]
        print(f"Forming teams with {len(candidates)} candidates")
        
        roles_doc = collection.find_one({
            "username": session['username'],
            "project_title": project_title
        })
        if not roles_doc:
            return jsonify({'success': False, 'message': 'No role requirements found.'})

        required_role_specs = roles_doc["roles"]
        print(f"Required roles: {required_role_specs}")
        
        role_candidates = {}
        for role_spec in required_role_specs:
            role_name = role_spec["role"]
            experience_level = role_spec.get("experience_level", "Any")
            role_key = f"{role_name} ({experience_level})"
            role_candidates[role_key] = []
        
        for c in candidates:
            role_name = c["role"]
            experience_level = c.get("experience_level", "Any")
            role_key = f"{role_name} ({experience_level})"
            if role_key in role_candidates:
                role_candidates[role_key].append(c)

        print(f"Candidates by role: {[(role, len(cands)) for role, cands in role_candidates.items()]}")

        available_roles = [role for role in role_candidates.keys() if role_candidates[role]]
        
        if len(available_roles) < 2:
            return jsonify({'success': False, 'message': 'Not enough roles with candidates available.'})
        
        print(f"Available roles for team formation: {available_roles}")
        
        team_results = greedy_beam_search(role_candidates, available_roles, beam_width=5)

        if not team_results:
            return jsonify({'success': False, 'message': 'No valid teams found.'})

        team_results.sort(key=lambda x: x[0], reverse=True)

        teams = []
        for rank, (score, team) in enumerate(team_results[:3], 1):
            teams.append({
                "rank": rank,
                "compatibility_score": round(score, 1),
                "recommended": rank == 1,
                "members": [
                    {
                        "name": m["name"], 
                        "role": m["role"],
                        "technical_score": m.get("technical_score", 0),
                        "behavioral_score": m.get("behavioral_score", 0),
                        "experience": m.get("experience", 0),
                        "skills": m.get("skills", [])
                    } for m in team
                ]
            })

        print(f"Generated {len(teams)} team options")

        highest_team = teams[0] if teams else None
        compatible_col.delete_many({
            "username": session['username'],
            "project_title": project_title
        })
        compatible_col.insert_one({
            "username": session['username'],
            "project_title": project_title,
            "highest_scored_team": highest_team,
            "top_teams": teams,
        })

        return jsonify({'success': True, 'data': teams})
        
    except Exception as e:
        print(f"Error in form_teams: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error forming teams: {str(e)}'})

@app.route('/api/save-team', methods=['POST'])
def save_team():
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})
    
    data = request.json
    
    try:
        team_data = {
            "id": str(int(time.time())),
            "username": session['username'],
            "project_title": data.get('projectTitle'),
            "project_description": data.get('projectDescription', ''),
            "project_type": data.get('projectType', ''),
            "members": data.get('members', []),
            "compatibility_score": data.get('compatibilityScore', 0),
            "team_size": len(data.get('members', [])),
            "status": 'active',
            "created_date": time.time()
        }
        
        db["UserTeams"].insert_one(team_data)
        
        return jsonify({'success': True, 'message': 'Team saved successfully', 'team_id': team_data['id']})
        
    except Exception as e:
        print(f"Error in save_team: {str(e)}")
        return jsonify({'success': False, 'message': f'Error saving team: {str(e)}'})

# ============================================================================
# PHASE 1: USER INFO & TEAMS
# ============================================================================

@app.route('/api/get-user-info', methods=['GET'])
def get_user_info():
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})
    
    try:
        username = session['username']
        
        user = users_collection.find_one({"username": username})
        if not user:
            return jsonify({'success': False, 'message': 'User not found'})
        
        teams = list(db["UserTeams"].find(
            {"username": username},
            {"_id": 0}
        ))
        
        projects = user.get("projects", [])
        
        active_teams = [t for t in teams if t.get('status') == 'active']
        completed_teams = [t for t in teams if t.get('status') == 'completed']
        total_members = sum(len(team.get('members', [])) for team in teams)
        
        compatibility_scores = [team.get('compatibility_score', team.get('compatibilityScore', 0)) for team in teams]
        avg_compatibility = sum(compatibility_scores) / len(compatibility_scores) if compatibility_scores else 0
        
        user_info = {
            'username': username,
            'total_projects': len(projects),
            'total_teams': len(teams),
            'active_teams': len(active_teams),
            'completed_teams': len(completed_teams),
            'total_members': total_members,
            'avg_compatibility': round(avg_compatibility, 1),
            'teams': teams,
            'recent_projects': sorted(projects, key=lambda x: x.get('created_at', 0), reverse=True)[:5]
        }
        
        return jsonify({'success': True, 'data': user_info})
        
    except Exception as e:
        print(f"Error in get_user_info: {str(e)}")
        return jsonify({'success': False, 'message': f'Error fetching user info: {str(e)}'})

@app.route('/api/user-teams', methods=['GET'])
@app.route('/api/get-user-teams', methods=['GET'])
def get_user_teams():
    """Get all teams - works with or without session for testing"""
    try:
        # Try to get teams with session first
        if 'username' in session:
            print(f"Getting teams for user: {session['username']}")
            teams = list(user_teams_collection.find(
                {"username": session['username']},
                {"_id": 0}
            ))
        else:
            # For testing without session, get all teams
            print("No session - getting all teams")
            teams = list(user_teams_collection.find({}, {"_id": 0}))
        
        print(f"Found {len(teams)} teams")
        for team in teams:
            print(f"  Team ID: {team.get('id')}, Title: {team.get('project_title')}")
        
        return jsonify({
            'success': True, 
            'teams': teams, 
            'data': teams,
            'count': len(teams)
        })
        
    except Exception as e:
        print(f"Error in get_user_teams: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'message': f'Error fetching teams: {str(e)}'
        }), 500

# ============================================================================
# PHASE 2: SLACK INTEGRATION API ENDPOINTS
# ============================================================================

@app.route('/api/slack/get-channels', methods=['GET'])
def get_slack_channels():
    """
    Get list of available Slack channels
    DEPRECATED: Use /api/slack/available-channels instead
    This endpoint now redirects to the new one for backward compatibility
    """
    try:
        # Get the exclude_team_id parameter if provided
        exclude_team_id = request.args.get('exclude_team_id')
        
        # Use the new function that filters out already-linked channels
        all_channels = get_available_slack_channels()
        
        # If a team_id is provided, include its currently linked channel
        if exclude_team_id:
            current_link = project_slack_channels_collection.find_one({"team_id": exclude_team_id})
            if current_link:
                current_channel_id = current_link.get("channel_id")
                
                # Check if current channel is in the list, if not, fetch and add it
                if not any(ch["id"] == current_channel_id for ch in all_channels):
                    # Fetch channel info from Slack API
                    url = "https://slack.com/api/conversations.info"
                    headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
                    params = {"channel": current_channel_id}
                    
                    try:
                        response = requests.get(url, headers=headers, params=params, timeout=5)
                        data = response.json()
                        
                        if data.get("ok") and "channel" in data:
                            channel = data["channel"]
                            all_channels.append({
                                "id": current_channel_id,
                                "name": channel.get("name", ""),
                                "is_private": channel.get("is_private", False),
                                "is_member": channel.get("is_member", False),
                                "currently_linked": True
                            })
                    except Exception as e:
                        print(f"Error fetching current channel info: {str(e)}")
        
        return jsonify({
            "success": True,
            "data": all_channels,
            "count": len(all_channels)
        })
            
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error fetching channels: {str(e)}"
        }), 500

@app.route('/api/slack/link-channel', methods=['POST'])
def link_channel():
    """Link a Slack channel to a project/team"""
    try:
        data = request.json
        project_id = data.get("project_id")
        channel_id = data.get("channel_id")
        channel_name = data.get("channel_name")
        
        if not project_id or not channel_id:
            return jsonify({
                "success": False,
                "message": "project_id and channel_id are required"
            }), 400
        
        print(f"Linking channel {channel_id} ({channel_name}) to project {project_id}")
        
        # Get team info - try both with id and team_id field
        team = user_teams_collection.find_one({"id": project_id})
        if not team:
            team = user_teams_collection.find_one({"team_id": project_id})
        if not team:
            # Last resort - search by project_title if project_id looks like a title
            team = user_teams_collection.find_one({"project_title": project_id})
        
        if not team:
            print(f"Team not found for project_id: {project_id}")
            return jsonify({
                "success": False,
                "message": f"Team not found with id: {project_id}"
            }), 404
        
        print(f"Found team: {team.get('project_title')}")
        
        # Create/update the link in ProjectSlackChannels
        link_doc = {
            "project_id": project_id,
            "project_title": team.get("project_title", project_id),
            "team_id": project_id,
            "lead_username": team.get("username", ""),
            "channel_id": channel_id,
            "channel_name": channel_name,
            "linked_at": time.time(),
            "linked_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        project_slack_channels_collection.update_one(
            {"project_id": project_id},
            {"$set": link_doc},
            upsert=True
        )
        
        # Also update the team document
        user_teams_collection.update_one(
            {"id": project_id},
            {"$set": {
                "slack_channel_id": channel_id,
                "slack_channel_name": channel_name,
                "slack_linked_at": time.time()
            }}
        )
        
        print(f"Successfully linked channel to project")
        
        return jsonify({
            "success": True,
            "message": "Channel linked successfully",
            "data": link_doc
        })
        
    except Exception as e:
        print(f"Error linking channel: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

@app.route('/api/slack/available-channels', methods=['GET'])
def get_available_channels():
    """
    Get list of Slack channels that are NOT already linked to other teams
    Optionally exclude a specific team's current channel (for re-linking)
    """
    try:
        # Optional: team_id to exclude its current channel from the "already linked" filter
        exclude_team_id = request.args.get('exclude_team_id')
        
        # Get all channels from Slack
        all_channels = get_available_slack_channels()
        
        # If a team_id is provided, we want to include its currently linked channel
        # even if it's technically "linked" (so user can see their current selection)
        if exclude_team_id:
            current_link = project_slack_channels_collection.find_one({"team_id": exclude_team_id})
            if current_link:
                current_channel_id = current_link.get("channel_id")
                
                # Check if current channel is in the list, if not, fetch and add it
                if not any(ch["id"] == current_channel_id for ch in all_channels):
                    # Fetch channel info from Slack API
                    url = "https://slack.com/api/conversations.info"
                    headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
                    params = {"channel": current_channel_id}
                    
                    try:
                        response = requests.get(url, headers=headers, params=params, timeout=5)
                        data = response.json()
                        
                        if data.get("ok") and "channel" in data:
                            channel = data["channel"]
                            all_channels.append({
                                "id": current_channel_id,
                                "name": channel.get("name", ""),
                                "is_private": channel.get("is_private", False),
                                "is_member": channel.get("is_member", False),
                                "currently_linked": True  # Flag to indicate this is the current selection
                            })
                    except Exception as e:
                        print(f"Error fetching current channel info: {str(e)}")
        
        return jsonify({
            "success": True,
            "channels": all_channels,
            "count": len(all_channels)
        })
        
    except Exception as e:
        print(f"❌ Error fetching available channels: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/slack/get-linked-channel/<project_id>', methods=['GET'])
def get_linked_channel(project_id):
    """Get the linked Slack channel for a project"""
    try:
        print(f"Getting linked channel for project: {project_id}")
        
        link = project_slack_channels_collection.find_one(
            {"project_id": project_id},
            {"_id": 0}
        )
        
        if not link:
            # Also try with team_id
            link = project_slack_channels_collection.find_one(
                {"team_id": project_id},
                {"_id": 0}
            )
        
        if link:
            print(f"Found linked channel: {link.get('channel_name')}")
            return jsonify({
                "success": True,
                "data": link
            })
        else:
            print(f"No channel linked to project {project_id}")
            return jsonify({
                "success": False,
                "message": "No channel linked to this project"
            }), 404
            
    except Exception as e:
        print(f"Error getting linked channel: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

@app.route('/api/slack/get-messages/<project_id>', methods=['GET'])
def get_messages(project_id):
    """Get Slack messages for a project"""
    try:
        print(f"Getting messages for project: {project_id}")
        
        # Check if channel is linked
        link = project_slack_channels_collection.find_one({"project_id": project_id})
        if not link:
            link = project_slack_channels_collection.find_one({"team_id": project_id})
        
        if not link:
            print(f"No channel linked to project {project_id}")
            return jsonify({
                "success": False,
                "message": "No channel linked to this project"
            }), 404
        
        limit = int(request.args.get('limit', 50))
        
        # Get messages from MongoDB - try both project_id and team_id
        messages = list(slack_messages_collection.find(
            {"$or": [{"project_id": project_id}, {"team_id": project_id}]},
            {"_id": 0}
        ).sort("timestamp", -1).limit(limit))
        
        print(f"Found {len(messages)} messages")
        
        return jsonify({
            "success": True,
            "data": messages,
            "count": len(messages)
        })
        
    except Exception as e:
        print(f"Error getting messages: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

@app.route('/api/slack/get-analytics/<team_id>', methods=['GET'])
def get_analytics(team_id):
    """Get team analytics from the monitoring service"""
    try:
        analytics = team_analytics_collection.find_one(
            {"team_id": team_id, "period": "last_24h"},
            {"_id": 0}
        )
        
        if analytics:
            return jsonify({
                "success": True,
                "data": analytics
            })
        else:
            return jsonify({
                "success": False,
                "message": "No analytics available yet"
            }), 404
            
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

# ============================================================================
# HEALTH CHECK & TEST ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Team Compatibility Predictor",
        "timestamp": time.time()
    })

@app.route('/test')
def test():
    return jsonify({"message": "Flask backend is working!", "timestamp": time.time()})

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "success": False,
        "message": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "success": False,
        "message": "Internal server error"
    }), 500

# ============================================================================
# STARTUP & MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 TEAM COMPATIBILITY PREDICTOR - COMPLETE SYSTEM")
    print("="*70)
    print("\n📋 System Features:")
    print("   ✓ Phase 1: AI-Powered Team Formation")
    print("   ✓ Phase 2: Real-time Slack Monitoring")
    print(f"\n📊 MongoDB Status:")
    print(f"   ✓ Database: {db.name}")
    print(f"   ✓ Collections Ready:")
    print(f"      - Users, UserTeams, SameRolesCache")
    print(f"      - TopTechnicalScores, BehavioralScores")
    print(f"      - CompatibleTeams, ProjectSlackChannels")
    print(f"      - SlackMessages, TeamAnalytics")
    
    # Try to load behavioral model at startup
    try:
        load_behavioral_model()
        print(f"\n✅ Behavioral scoring model loaded successfully")
    except FileNotFoundError as e:
        print(f"\n⚠️  Warning: {e}")
        print("   Behavioral scoring will use fallback values until model is trained.")
        print("   To train the model, run: python behavioral_score.py train")
    
    print(f"\n🌐 Starting Flask server on http://0.0.0.0:5000")
    print(f"   Dashboard: http://localhost:5000/dashboard")
    print(f"   Team Formation: http://localhost:5000/team-formation")
    print(f"   Monitor: http://localhost:5000/monitor")
    print(f"\n⚠️  For Slack monitoring, also run: python app1.py (on port 5001)")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)