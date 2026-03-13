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
from datetime import datetime
import subprocess
import tempfile
import pickle
import numpy as np

# RoBERTa / HuggingFace imports for behavioral scoring
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
proj_req_cache_col = db["ProjectRequirementsCache"]   # dedicated cache for AI analysis results
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

# ============================================================================
# BEHAVIORAL SCORING — RoBERTa model (model/ folder)
# ============================================================================

TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
ROBERTA_MODEL_DIR = "model"   # folder containing the trained RoBERTa model artifacts
_model_cache = None


def load_behavioral_model():
    """
    Load the trained RoBERTa behavioral model from the model/ folder (cached).
    Returns the tokenizer (as 'embedder') and the RoBERTa model (as 'regressor')
    so that all downstream call-sites stay consistent with the original signature.
    """
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(ROBERTA_MODEL_DIR):
            raise FileNotFoundError(
                f"Model directory not found: '{ROBERTA_MODEL_DIR}'. "
                "Please ensure the trained RoBERTa model is saved inside the model/ folder."
            )
        tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_DIR)
        roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL_DIR)
        roberta_model.eval()
        if torch.cuda.is_available():
            roberta_model = roberta_model.cuda()
        _model_cache = {
            "embedder": tokenizer,       # kept as 'embedder' for API consistency
            "regressor": roberta_model,  # kept as 'regressor' for API consistency
            "traits": TRAITS
        }
        print("✅ RoBERTa behavioral model loaded successfully from model/ folder")
    return _model_cache["embedder"], _model_cache["regressor"], _model_cache["traits"]


def get_personality_scores_fast(summary: str, embedder, regressor) -> dict:
    """
    Predict Big Five personality scores using the RoBERTa model.

    Parameters
    ----------
    summary  : behavioural text summary for the candidate
    embedder : HuggingFace tokenizer loaded from model/
    regressor: RoBERTa AutoModelForSequenceClassification with 5 output nodes

    Returns
    -------
    dict  {trait_name: float}  with values clipped to [0.0, 1.0]
    """
    device = next(regressor.parameters()).device

    inputs = embedder(
        summary,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = regressor(**inputs)
        logits = outputs.logits  # shape: (1, num_labels==5)

    # Sigmoid maps raw logits → [0, 1] for each trait
    raw_scores = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    scores = {}
    for i, trait in enumerate(TRAITS):
        scores[trait] = float(np.clip(raw_scores[i], 0.0, 1.0))

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

# ============================================================================
# TECHNICAL SCORING HELPERS  (ported from technicalscore.py)
# ============================================================================

# Maps every AI-generated role variant → the canonical role name in the CSV
ROLE_MAPPINGS = {
    'ui/ux designer': 'ui/ux designer',
    'ux/ui designer': 'ui/ux designer',
    'ui designer': 'ui/ux designer',
    'ux designer': 'ui/ux designer',
    'user experience designer': 'ui/ux designer',
    'user interface designer': 'ui/ux designer',
    'devops engineer': 'devops engineer',
    'dev ops engineer': 'devops engineer',
    'devops': 'devops engineer',
    'backend developer': 'backend developer',
    'back end developer': 'backend developer',
    'back-end developer': 'backend developer',
    'server side developer': 'backend developer',
    'backend engineer': 'backend developer',
    'frontend developer': 'frontend developer',
    'front end developer': 'frontend developer',
    'front-end developer': 'frontend developer',
    'client side developer': 'frontend developer',
    'frontend engineer': 'frontend developer',
    'qa engineer': 'qa engineer',
    'qa tester': 'qa engineer',
    'quality assurance engineer': 'qa engineer',
    'quality assurance tester': 'qa engineer',
    'test engineer': 'qa engineer',
    'testing engineer': 'qa engineer',
    'software tester': 'qa engineer',
    'automation engineer': 'qa engineer',
    'test automation engineer': 'qa engineer',
    'data scientist': 'data scientist',
    'data analyst': 'data analyst',
    'ml engineer': 'machine learning engineer',
    'machine learning engineer': 'machine learning engineer',
    'ai engineer': 'ai engineer',
    'ai specialist': 'ai engineer',
    'artificial intelligence engineer': 'ai engineer',
    'software engineer': 'software engineer',
    'software developer': 'software engineer',
    'application developer': 'software engineer',
    'programmer': 'software engineer',
    'full stack developer': 'full stack developer',
    'fullstack developer': 'full stack developer',
    'full-stack developer': 'full stack developer',
    'full stack engineer': 'full stack developer',
    'mobile developer': 'mobile app developer',
    'mobile app developer': 'mobile app developer',
    'android developer': 'mobile app developer',
    'ios developer': 'mobile app developer',
    'hardware engineer': 'hardware engineer',
    'electronics engineer': 'hardware engineer',
    'embedded engineer': 'hardware engineer',
    'cloud engineer': 'cloud engineer',
    'security analyst': 'security analyst',
    'system architect': 'system architect',
    'business analyst': 'business analyst',
    'product manager': 'product manager',
    'blockchain developer': 'blockchain developer',
}


def normalize_skill(skill: str) -> str:
    """Normalize a skill string for consistent matching."""
    if not skill:
        return ""
    skill = str(skill).strip().lower()
    skill = re.sub(r'[^\w\s]', '', skill)
    skill = re.sub(r'\s+', ' ', skill)
    return skill


def normalize_role(role: str) -> str:
    """Map an AI-suggested role name to the canonical dataset role name."""
    if not role:
        return ""
    key = str(role).strip().lower()
    return ROLE_MAPPINGS.get(key, key)


def extract_skills_from_text(skills_text: str) -> list:
    """Parse comma-separated skills string into a normalized list."""
    if not skills_text or (isinstance(skills_text, float)):
        return []
    return [normalize_skill(s) for s in str(skills_text).split(',') if normalize_skill(s)]


def calculate_skill_match_score(employee_skills: list, required_skills: list):
    """
    3-tier skill matching: exact → substring → word-level.
    Returns (score_0_to_100, list_of_matched_skill_names).
    """
    if not required_skills or not employee_skills:
        return 0.0, []

    norm_required = [normalize_skill(s) for s in required_skills if s]
    norm_employee  = [normalize_skill(s) for s in employee_skills  if s]

    if not norm_required or not norm_employee:
        return 0.0, []

    matched = 0
    matched_names = []
    for req in norm_required:
        for emp in norm_employee:
            if req == emp:                                                          # exact
                matched += 1; matched_names.append(req.title()); break
            elif req in emp or emp in req:                                          # substring
                matched += 1; matched_names.append(req.title()); break
            elif (any(w in emp.split() for w in req.split() if len(w) > 2) or      # word-level
                  any(w in req.split() for w in emp.split() if len(w) > 2)):
                matched += 1; matched_names.append(req.title()); break

    score = round((matched / len(norm_required)) * 100, 2)
    return score, list(set(matched_names))


def is_experience_match(emp_exp, required_level: str) -> bool:
    """
    NewHire  → 0 years of experience
    Experienced / Senior / Mid-level → > 0 years
    """
    try:
        years = int(emp_exp) if emp_exp is not None and not (isinstance(emp_exp, float) and np.isnan(emp_exp)) else 0
    except (ValueError, TypeError):
        years = 0

    level = str(required_level).lower().strip()
    if level in ('experienced', 'senior', 'mid-level'):
        return years > 0
    elif level in ('newhire', 'new hire', 'junior', 'entry level'):
        return years == 0
    return True   # 'Any' or unknown — include everyone


def get_top_candidates_for_roles(roles: list, df: pd.DataFrame, top_n: int = 4) -> dict:
    """
    Core scoring engine.  For each role spec, normalise the AI role name,
    find matching employees (role + experience-level), score them with
    3-tier skill matching, and return the top `top_n` per role.

    Returns: { "Role Name (Level)": [candidate_dict, ...] }
    """
    candidates_by_role = {}
    all_candidates_flat = []

    for role_spec in roles:
        ai_role          = role_spec["role"]
        required_level   = role_spec.get("experience_level", "Any")
        required_skills  = role_spec.get("skills", [])
        canonical_role   = normalize_role(ai_role)
        role_key         = f"{ai_role} ({required_level})"

        role_results = []
        for _, row in df.iterrows():
            try:
                emp_role_raw  = str(row.get('Role', ''))
                emp_canonical = normalize_role(emp_role_raw)
                emp_exp       = row.get('Experience', 0)
                emp_skills    = extract_skills_from_text(row.get('TechnicalSkills', ''))

                # Role match: canonical equality OR one contains the other
                role_match = (
                    emp_canonical == canonical_role or
                    emp_canonical in canonical_role or
                    canonical_role in emp_canonical
                )

                if role_match and is_experience_match(emp_exp, required_level):
                    tech_score, matched_skills = calculate_skill_match_score(emp_skills, required_skills)
                    exp_label = 'NewHire' if int(emp_exp) == 0 else 'Experienced'

                    candidate = {
                        "name":             row.get('Name', 'N/A'),
                        "role":             emp_role_raw,           # actual CSV role
                        "assigned_role":    ai_role,                # AI-requested role (used for team grouping)
                        "experience_level": exp_label,
                        "skills":           emp_skills,
                        "matched_skills":   matched_skills,
                        "experience":       int(emp_exp),
                        "technical_score":  tech_score,
                        "employee_role":    emp_role_raw,
                    }
                    role_results.append(candidate)

            except Exception as e:
                print(f"  [warn] skipping {row.get('Name','?')}: {e}")
                continue

        # Sort by score then experience, keep top N
        role_results.sort(key=lambda x: (x['technical_score'], x['experience']), reverse=True)
        top_for_role = role_results[:top_n]
        candidates_by_role[role_key] = top_for_role

        # ONLY add the top-N to the flat list — this is what gets saved to MongoDB
        # and consumed by behavioral scoring, so it must match what the UI shows
        all_candidates_flat.extend(top_for_role)

    return candidates_by_role, all_candidates_flat


def strip_markdown(text: str) -> str:
    """Remove common markdown formatting from a string."""
    # Remove bold/italic markers: **, __, *, _
    text = re.sub(r'\*{1,3}|_{1,3}', '', text)
    # Remove inline code backticks
    text = re.sub(r'`+', '', text)
    # Normalise en-dash / em-dash / bullet chars to plain hyphen-space
    text = re.sub(r'[–—•]', '-', text)
    # Collapse multiple spaces
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def parse_api_response(response_text):
    print(f"Raw API Response for parsing:\n{response_text}\n")

    lines = response_text.strip().split('\n')
    project_type = ""
    team_size = 0
    roles = []

    for raw_line in lines:
        # Strip markdown formatting first so all comparisons work on plain text
        line = strip_markdown(raw_line).strip()

        # ── Project Type ──────────────────────────────────────────────────────
        if re.match(r'project\s*type\s*:', line, re.IGNORECASE):
            project_type = re.sub(r'project\s*type\s*:', '', line, flags=re.IGNORECASE).strip()
            continue

        # ── Team Size ─────────────────────────────────────────────────────────
        if re.match(r'team\s*size\s*:', line, re.IGNORECASE):
            nums = re.findall(r'\d+', line)
            if nums:
                team_size = int(nums[0])
            continue

        # ── Role lines ────────────────────────────────────────────────────────
        # Detect experience level keyword anywhere in the line (case-insensitive)
        exp_match = re.search(r'\b(Experienced|NewHire|New\s*Hire)\b', line, re.IGNORECASE)
        if not exp_match:
            continue

        experience_level = "NewHire" if re.search(r'new\s*hire', exp_match.group(), re.IGNORECASE) else "Experienced"

        # Split on the experience keyword (plus surrounding separators like - / –)
        # Pattern: <role> [sep] Experienced/NewHire [sep] <skills>
        split_pattern = r'\s*[-/|]\s*(?:Experienced|NewHire|New\s*Hire)\s*[-/|]\s*'
        parts = re.split(split_pattern, line, maxsplit=1, flags=re.IGNORECASE)

        if len(parts) == 2:
            role_raw, skills_raw = parts
        else:
            # Fallback: split at the keyword position
            idx = exp_match.start()
            role_raw = line[:idx]
            skills_raw = line[exp_match.end():]
            # Strip leading separators from skills_raw
            skills_raw = re.sub(r'^[\s\-/|]+', '', skills_raw)

        # Clean role name: strip leading numbering, bullets, separators
        role_name = re.sub(r'^[\d]+[\.\)]\s*', '', role_raw)   # "1. " or "1) "
        role_name = re.sub(r'^[-\s•*]+', '', role_name)         # leading bullets/dashes
        role_name = role_name.strip()

        # Parse skills — split by comma; also handle parenthetical notes inline
        skills = [s.strip() for s in re.split(r',\s*', skills_raw) if s.strip()]
        # Drop empty or very-short tokens that are just punctuation artifacts
        skills = [s for s in skills if len(s) > 1]

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
        # ── 1. Exact cache hit (same title + same description, any user) ──────
        cached = proj_req_cache_col.find_one({
            "project_title": project_title,
            "project_description": project_description
        })

        if not cached:
            # ── 2. Semantic similarity cache (same title, description ≥ 0.80) ──
            if model is not None:
                new_emb = model.encode(project_description, convert_to_tensor=True)
                best_match = None
                best_sim = 0.0
                for doc in proj_req_cache_col.find({"project_title": project_title}):
                    try:
                        doc_emb = model.encode(doc["project_description"], convert_to_tensor=True)
                        sim = util.pytorch_cos_sim(new_emb, doc_emb).item()
                        if sim > 0.80 and sim > best_sim:
                            best_sim = sim
                            best_match = doc
                    except Exception:
                        continue
                if best_match:
                    cached = best_match
                    print(f"[Cache] Semantic hit — similarity {best_sim:.3f}")

        if cached:
            print(f"[Cache] Returning cached analysis for '{project_title}'")
            result = {
                "project_type": cached["project_type"],
                "team_size": cached["team_size"],
                "roles": cached["roles"],
                "projectTitle": project_title,
                "projectDescription": project_description,
                "from_cache": True
            }
            # Ensure the current user also has a personal copy in the cache
            if not proj_req_cache_col.find_one({
                "username": session['username'],
                "project_title": project_title,
                "project_description": project_description
            }):
                proj_req_cache_col.insert_one({
                    "username": session['username'],
                    "project_title": project_title,
                    "project_description": project_description,
                    "project_type": cached["project_type"],
                    "team_size": cached["team_size"],
                    "roles": cached["roles"]
                })
            return jsonify({'success': True, 'data': result})

        # ── 3. No cache — call the AI API ─────────────────────────────────────
        print(f"[API] No cache hit for '{project_title}' — calling OpenRouter...")

        prompt = f"""
You are an intelligent AI assistant. Given a software project title and description, your task is to suggest a software project type, team size, and a list of roles.

Strictly follow this structure and constraints:

Project Type: <value>
Team Size: <exact number between 4-10>

Roles: (excluding Project Manager or Team Lead)
Each role must follow this exact format on its own line:
<Role> - <Experienced/NewHire> - <comma-separated skills>

ROLE GUIDELINES:
- Use only these role names: Frontend Developer, Backend Developer, Full Stack Developer, Mobile App Developer, UI/UX Designer, DevOps Engineer, QA Engineer, Data Scientist, Data Analyst, Machine Learning Engineer, AI Engineer, Software Engineer, Cloud Engineer, Security Analyst, System Architect, Business Analyst, Blockchain Developer, Hardware Engineer, Product Manager
- Avoid repeating the same role at both Experienced and NewHire levels unless clearly necessary
- Total number of role entries (Experienced + NewHire combined) must equal the Team Size exactly

SKILL GUIDELINES:
- Skills should be specific but not overly technical or niche
- For EXPERIENCED roles, include advanced/specialized skills
- For NEWHIRE roles, include fundamental/basic skills
- Use full names, not abbreviations (e.g. "JavaScript" not "JS", "PostgreSQL" not "Postgres")
- Skills should match the experience level appropriately

Now process the following:

Project Title: {project_title}
Project Description: {project_description}
"""
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        api_response = response.json()
        ai_text = api_response['choices'][0]['message']['content']
        
        parsed_data = parse_api_response(ai_text)

        # Save to dedicated cache for future requests
        proj_req_cache_col.insert_one({
            "username": session['username'],
            "project_title": project_title,
            "project_description": project_description,
            "project_type": parsed_data["project_type"],
            "team_size": parsed_data["team_size"],
            "roles": parsed_data["roles"]
        })
        print(f"[Cache] Saved analysis for '{project_title}' to ProjectRequirementsCache")

        parsed_data['projectTitle'] = project_title
        parsed_data['projectDescription'] = project_description
        parsed_data['from_cache'] = False
        
        return jsonify({'success': True, 'data': parsed_data})
        
    except Exception as e:
        print(f"Error in project_requirements: {str(e)}")
        import traceback
        traceback.print_exc()
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
        return jsonify({'success': False, 'message': 'Invalid data: projectTitle and roles are required'})
    
    try:
        # Save role requirements to SameRolesCache (used later by team formation)
        collection.delete_many({
            "username": session['username'],
            "project_title": project_title
        })
        collection.insert_one({
            "username": session['username'],
            "project_title": project_title,
            "roles": roles
        })

        # Load dataset — columns: Name, Role, TechnicalSkills, BehaviouralSummary, Experience
        csv_path = "UpdatedDataset.csv"
        if not os.path.exists(csv_path):
            return jsonify({'success': False, 'message': 'Dataset not found'})

        df = pd.read_csv(csv_path)
        print(f"[Technical] Dataset loaded: {len(df)} employees, "
              f"roles in dataset: {sorted(df['Role'].unique())}")

        # Run the full scoring engine (normalize roles, 3-tier skill match, top 4 per role)
        candidates_by_role, all_candidates_flat = get_top_candidates_for_roles(roles, df, top_n=4)

        # Log what was found
        for rk, cands in candidates_by_role.items():
            print(f"  [{rk}] → {len(cands)} candidates found")
            if not cands:
                # Role mapping may have missed — show what was searched
                ai_role = rk.split(' (')[0]
                print(f"    [warn] No match for AI role '{ai_role}' "
                      f"(normalized: '{normalize_role(ai_role)}'). "
                      f"Dataset roles: {sorted(df['Role'].str.lower().unique())}")

        # Persist flat list to MongoDB for downstream behavioral scoring
        top_scores_col.delete_many({
            "username": session['username'],
            "project_title": project_title
        })
        top_scores_col.insert_one({
            "username": session['username'],
            "project_title": project_title,
            "candidates": all_candidates_flat
        })

        return jsonify({'success': True, 'data': candidates_by_role})

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
        # Load the RoBERTa behavioral model from model/ folder
        try:
            embedder, regressor, _ = load_behavioral_model()
            print("Using RoBERTa behavioral model from model/ folder")
        except FileNotFoundError as model_err:
            print(f"RoBERTa model not found — using fallback scores. Reason: {model_err}")
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
        
        # Load dataset for behavioural summary text
        csv_path = "UpdatedDataset.csv"
        if not os.path.exists(csv_path):
            return jsonify({'success': False, 'message': 'Dataset not found'})
        
        df = pd.read_csv(csv_path)
        # CSV columns: Name, Role, TechnicalSkills, BehaviouralSummary, Experience

        enriched_candidates = []

        for c in candidates:
            # Match by Name and Role (actual CSV column names)
            match = df[
                (df['Name'] == c['name']) &
                (df['Role'].str.lower() == c['role'].lower())
            ]

            if not match.empty:
                row = match.iloc[0]

                # pandas Series — use direct indexing, not .get()
                summary = str(row['BehaviouralSummary']) if 'BehaviouralSummary' in row.index else ''
                
                if embedder is not None and regressor is not None and summary.strip():
                    # Use the RoBERTa model for prediction
                    scores = get_personality_scores_fast(summary, embedder, regressor)
                else:
                    # Fall back to defaults when model or summary is unavailable
                    fallback = create_fallback_candidate(c)
                    scores = fallback["scores"]
                
                # Calculate final behavioral score (same formula as before)
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
                # No dataset match — use fallback defaults
                enriched_candidate = create_fallback_candidate(c)
            
            enriched_candidates.append(enriched_candidate)
        
        # Persist enriched candidates to MongoDB
        behavior_col.delete_many({
            "username": session['username'],
            "project_title": project_title
        })
        behavior_col.insert_one({
            "username": session['username'],
            "project_title": project_title,
            "candidates": enriched_candidates
        })

        print(f"[Behavioral] Enriched {len(enriched_candidates)} candidates for '{project_title}'")

        # Return as {candidates: [...]} — matches displayBehavioralResults(result.data).candidates
        return jsonify({'success': True, 'data': {'candidates': enriched_candidates}})
        
    except Exception as e:
        print(f"Error in behavioral_scores: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

# ============================================================================
# PHASE 1: TEAM FORMATION (Compatibility Algorithm)
# ============================================================================

def pairwise_compatibility(candidate_a, candidate_b):
    """
    Pairwise personality compatibility (from compatibilitycheck.py):
    - Similarity traits:    Conscientiousness, Agreeableness  (closer = better)
    - Complementary traits: Openness, Extraversion, Neuroticism (balanced = better)
    Returns a 0–1 score.
    """
    traits_a = candidate_a.get("scores", {})
    traits_b = candidate_b.get("scores", {})

    similarity_traits     = ["Conscientiousness", "Agreeableness"]
    complementary_traits  = ["Openness", "Extraversion", "Neuroticism"]

    score = 0.0
    count = 0

    for trait in similarity_traits:
        score += 1 - abs(traits_a.get(trait, 0.5) - traits_b.get(trait, 0.5))
        count += 1

    for trait in complementary_traits:
        score += 1 - abs((traits_a.get(trait, 0.5) + traits_b.get(trait, 0.5)) - 1)
        count += 1

    return score / count if count else 0.5


def team_personality_compatibility(team):
    """Average pairwise compatibility across all pairs in a team (returns 0–100)."""
    pairs = list(itertools.combinations(team, 2))
    if not pairs:
        return 0.0
    return (sum(pairwise_compatibility(a, b) for a, b in pairs) / len(pairs)) * 100


def compute_team_compatibility(team):
    """
    Final weighted team score combining:
      35% technical average
      35% behavioral average
      30% personality compatibility (pairwise similarity + complementary)
    """
    if len(team) < 2:
        return 0

    technical_avg  = sum(m.get("technical_score",  60) for m in team) / len(team)
    behavioral_avg = sum(m.get("behavioral_score", 60) for m in team) / len(team)
    personality_score = team_personality_compatibility(team)

    return (technical_avg * 0.35 +
            behavioral_avg * 0.35 +
            personality_score * 0.30)

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
            role_name        = role_spec["role"]
            experience_level = role_spec.get("experience_level", "Any")
            role_key         = f"{role_name} ({experience_level})"
            role_candidates[role_key] = []

        for c in candidates:
            # Use assigned_role (AI-requested role) for grouping — falls back to raw role
            role_name        = c.get("assigned_role") or c["role"]
            experience_level = c.get("experience_level", "Any")
            role_key         = f"{role_name} ({experience_level})"
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

@app.route('/api/slack/fetch-history/<team_id>', methods=['POST'])
def fetch_slack_history(team_id):
    """
    Manually fetch Slack message history for a team's linked channel.
    Called by "Sync Slack Messages" button. Works without app1.py running.
    """
    try:
        channel_link = project_slack_channels_collection.find_one({"team_id": team_id})
        if not channel_link:
            return jsonify({"success": False, "message": "No Slack channel linked to this team"})

        channel_id = channel_link.get("channel_id")
        lead_username = channel_link.get("lead_username", "")
        project_title = channel_link.get("project_title", channel_link.get("project_id", ""))

        url = "https://slack.com/api/conversations.history"
        headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
        oldest = str(time.time() - 7 * 86400)  # last 7 days
        params = {"channel": channel_id, "oldest": oldest, "limit": 200}

        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()

        if not data.get("ok"):
            error = data.get("error", "unknown")
            msg = f"Slack API error: {error}"
            if error == "not_in_channel":
                msg += ". Bot is not in this channel — run /invite @YourBot in Slack."
            return jsonify({"success": False, "message": msg})

        messages = data.get("messages", [])
        new_count = 0

        for msg in messages:
            if msg.get("subtype") or not msg.get("user"):
                continue
            ts = msg.get("ts", "")
            existing = slack_messages_collection.find_one({
                "channel_id": channel_id, "timestamp": float(ts)
            })
            if existing:
                continue

            user_id = msg["user"]
            # Resolve Slack user ID → display name
            user_info_resp = requests.get(
                "https://slack.com/api/users.info",
                headers=headers, params={"user": user_id}, timeout=5
            ).json()
            user_name = "Unknown"
            if user_info_resp.get("ok") and "user" in user_info_resp:
                profile = user_info_resp["user"].get("profile", {})
                user_name = profile.get("display_name") or profile.get("real_name") or "Unknown"

            text = msg.get("text", "")
            message_doc = {
                "slack_user_id": user_id,
                "user_name": user_name,
                "channel_id": channel_id,
                "project_title": project_title,
                "project_id": project_title,
                "team_id": team_id,
                "lead_username": lead_username,
                "message": text,
                "text": text,
                "timestamp": float(ts),
                "datetime": datetime.fromtimestamp(float(ts)).isoformat(),
                "created_at": float(ts),
                "date": datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d"),
                "time": datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S"),
                "source": "manual_sync"
            }
            slack_messages_collection.insert_one(message_doc)
            new_count += 1

        # Recompute analytics after sync
        _recompute_analytics(team_id)

        channel_name = channel_link.get("channel_name", channel_id)
        return jsonify({
            "success": True,
            "message": f"Synced {new_count} new messages from #{channel_name}",
            "new_messages": new_count
        })

    except Exception as e:
        print(f"Error in fetch_slack_history: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)})


def _score_messages_behaviorally(messages_text: str):
    """Keyword-based behavioral scoring. Returns score 0-100 or None if no text."""
    if not messages_text or not messages_text.strip():
        return None
    text = messages_text.lower()
    base = 70.0
    positive = ["thanks", "thank you", "great", "good job", "well done", "appreciate",
                "agree", "helpful", "sure", "happy to", "absolutely", "excellent",
                "nice work", "awesome", "perfect", "will do", "on it", "done",
                "finished", "completed", "delivered", "sounds good", "yes", "correct",
                "good point", "i can help", "absolutely", "great idea"]
    negative = ["no", "won't", "can't", "refuse", "disagree", "wrong", "not my job",
                "whatever", "doubt", "problem", "issue", "fail", "failed", "mistake",
                "error", "terrible", "hate", "bad", "ugh", "annoying", "frustrated",
                "late", "delay", "missed", "not done", "incomplete"]
    pos_hits = sum(1 for kw in positive if kw in text)
    neg_hits = sum(1 for kw in negative if kw in text)
    return round(min(max(base + pos_hits * 1.5 - neg_hits * 2.0, 0.0), 100.0), 1)


def _recompute_analytics(team_id):
    """Recompute and store analytics for a team. Called after syncing messages."""
    try:
        team_doc = user_teams_collection.find_one({"id": team_id})
        if not team_doc:
            return
        team_members = team_doc.get("members", [])
        initial_compat = float(team_doc.get("compatibility_score", 0))
        project_title = team_doc.get("project_title", "")
        team_owner = team_doc.get("username", "")

        all_messages = list(slack_messages_collection.find({"team_id": team_id}))
        cutoff = time.time() - 86400
        recent = [m for m in all_messages if m.get("created_at", 0) >= cutoff]

        user_msg_counts = defaultdict(int)
        for m in recent:
            user_msg_counts[m.get("user_name", "Unknown")] += 1

        messages_by_user = defaultdict(list)
        for m in all_messages:
            messages_by_user[m.get("user_name", "Unknown")].append(m.get("text", ""))

        def norm(n): return n.lower().strip() if n else ""

        member_behavioral_analysis = []
        for member in team_members:
            member_name = member.get("name", "")
            initial_score = float(member.get("behavioral_score", 70.0))
            norm_name = norm(member_name)

            user_texts = messages_by_user.get(member_name, [])
            if not user_texts:
                for slack_name, texts in messages_by_user.items():
                    if norm(slack_name) == norm_name or norm_name in norm(slack_name) or norm(slack_name) in norm_name:
                        user_texts = texts
                        break

            has_data = len(user_texts) >= 1
            messages_analyzed = len(user_texts)

            if has_data:
                current_score = _score_messages_behaviorally(" ".join(user_texts[:20]))
                if current_score is None:
                    current_score = initial_score
                    has_data = False
            else:
                current_score = initial_score

            fluctuation = round(current_score - initial_score, 1)
            fluctuation_pct = round((fluctuation / initial_score * 100), 1) if initial_score > 0 else 0.0
            trend = ("improving" if has_data and messages_analyzed >= 2 and fluctuation > 3
                     else "declining" if has_data and messages_analyzed >= 2 and fluctuation < -3
                     else "stable")

            member_behavioral_analysis.append({
                "user_name": member_name,
                "initial_behavioral_score": round(initial_score, 1),
                "current_behavioral_score": round(current_score, 1),
                "fluctuation": fluctuation,
                "fluctuation_pct": fluctuation_pct,
                "trend": trend,
                "alert": has_data and fluctuation < -10,
                "has_data": has_data,
                "messages_analyzed": messages_analyzed
            })

        members_with_data = [m for m in member_behavioral_analysis if m["has_data"]]
        if members_with_data:
            avg_fluctuation = sum(m["fluctuation"] for m in members_with_data) / len(members_with_data)
            current_compat = round(max(0.0, min(100.0, initial_compat + avg_fluctuation * 0.5)), 1)
        else:
            current_compat = initial_compat

        compat_change = round(current_compat - initial_compat, 1)
        compat_trend = ("improving" if compat_change > 2 else "declining" if compat_change < -2 else "stable")

        analytics_doc = {
            "team_id": team_id,
            "team_owner": team_owner,
            "project_title": project_title,
            "period": "last_24h",
            "total_messages": len(recent),
            "active_members": len(user_msg_counts),
            "member_behavioral_analysis": member_behavioral_analysis,
            "team_compatibility": {
                "initial_score": round(initial_compat, 1),
                "current_score": current_compat,
                "change": compat_change,
                "trend": compat_trend
            },
            "last_updated": time.time(),
            "last_updated_iso": datetime.now().isoformat()
        }

        team_analytics_collection.update_one(
            {"team_id": team_id, "period": "last_24h"},
            {"$set": analytics_doc},
            upsert=True
        )
        print(f"📊 Analytics recomputed for team {team_id}")
    except Exception as e:
        print(f"❌ _recompute_analytics error: {e}")


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
    
    # Try to load RoBERTa behavioral model at startup
    try:
        load_behavioral_model()
        print(f"\n✅ RoBERTa behavioral model loaded successfully from model/ folder")
    except FileNotFoundError as e:
        print(f"\n⚠️  Warning: {e}")
        print("   Behavioral scoring will use fallback values until the model/ folder is present.")
    
    print(f"\n🌐 Starting Flask server on http://0.0.0.0:5000")
    print(f"   Dashboard: http://localhost:5000/dashboard")
    print(f"   Team Formation: http://localhost:5000/team-formation")
    print(f"   Monitor: http://localhost:5000/monitor")
    print(f"\n⚠️  For Slack monitoring, also run: python app1.py (on port 5001)")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)