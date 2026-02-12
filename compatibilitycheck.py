import itertools
from pymongo import MongoClient
import heapq
from collections import defaultdict

# -----------------------
# Mongo Config
# -----------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["AISubmodule"]
behavior_col = db["BehavioralScores"]
same_roles_col = db["SameRolesCache"]
compatible_col = db["CompatibleTeams"]

# -----------------------
# Compatibility Functions
# -----------------------

def avg_behavioral_score(scores):
    return sum(scores.values()) / len(scores)

def compatibility_score(candidate_a, candidate_b):
    traits_a = candidate_a["scores"]
    traits_b = candidate_b["scores"]

    similarity_traits = ["Conscientiousness", "Agreeableness"]
    complementary_traits = ["Openness", "Extraversion", "Neuroticism"]

    score = 0.0
    count = 0

    # Similarity fit
    for trait in similarity_traits:
        score += 1 - abs(traits_a[trait] - traits_b[trait])
        count += 1

    # Complementary fit
    for trait in complementary_traits:
        score += 1 - abs((traits_a[trait] + traits_b[trait]) - 1)
        count += 1

    return score / count  # 0–1 range

def team_compatibility(team):
    total = 0
    pairs = 0
    for a, b in itertools.combinations(team, 2):
        total += compatibility_score(a, b)
        pairs += 1
    return (total / pairs) * 100 if pairs > 0 else 0  # return percentage

def candidate_team_compatibility(candidate, existing_team):
    """Calculate compatibility between a candidate and existing team members"""
    if not existing_team:
        return avg_behavioral_score(candidate["scores"])
    
    total_compatibility = 0
    for team_member in existing_team:
        total_compatibility += compatibility_score(candidate, team_member)
    
    return total_compatibility / len(existing_team)

def find_missing_role_candidates(role, project_title, existing_team, excluded_names):
    """
    Find candidates for missing roles, prioritizing same project then other projects
    Selection based on compatibility with existing team
    """
    candidates = []
    
    # First, try to find candidates in the same project
    project_doc = behavior_col.find_one({"project_title": project_title})
    if project_doc:
        for cand in project_doc["candidates"]:
            if cand["role"] == role and cand["name"] not in excluded_names:
                compatibility = candidate_team_compatibility(cand, existing_team)
                candidates.append((compatibility, cand))
    
    # If no candidates found in same project, search other projects
    if not candidates:
        for doc in behavior_col.find({"project_title": {"$ne": project_title}}):
            for cand in doc["candidates"]:
                if cand["role"] == role and cand["name"] not in excluded_names:
                    compatibility = candidate_team_compatibility(cand, existing_team)
                    candidates.append((compatibility, cand))
    
    # Sort by compatibility score (descending)
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [cand for _, cand in candidates]

def greedy_beam_search(role_candidates, required_roles, beam_width=3):
    """
    Optimized team formation using greedy beam search
    Instead of brute force, build teams incrementally keeping top candidates
    """
    if not required_roles or not role_candidates:
        return []
    
    # Start with candidates for the first role
    first_role = required_roles[0]
    current_beams = []
    
    for candidate in role_candidates[first_role]:
        team = [candidate]
        score = avg_behavioral_score(candidate["scores"])  # Initial score
        current_beams.append((score, team, {candidate["name"]}))
    
    # Keep only top beam_width candidates
    current_beams = sorted(current_beams, key=lambda x: x[0], reverse=True)[:beam_width]
    
    # Iteratively add candidates for remaining roles
    for role in required_roles[1:]:
        next_beams = []
        
        for current_score, current_team, used_names in current_beams:
            # Try each candidate for this role
            for candidate in role_candidates[role]:
                if candidate["name"] not in used_names:
                    new_team = current_team + [candidate]
                    new_used = used_names | {candidate["name"]}
                    
                    # Calculate team compatibility for the new team
                    team_score = team_compatibility(new_team)
                    next_beams.append((team_score, new_team, new_used))
        
        # Keep only top beam_width teams
        current_beams = sorted(next_beams, key=lambda x: x[0], reverse=True)[:beam_width]
        
        if not current_beams:
            break
    
    # Return all beams as potential teams
    return [(score, team) for score, team, _ in current_beams]

# -----------------------
# Main Execution
# -----------------------

def main():
    project_title = input("Enter project title: ").strip()

    # Load behavioral scores for project
    project_doc = behavior_col.find_one({"project_title": project_title})
    if not project_doc:
        print(f"No behavioral scores found for '{project_title}'")
        return

    candidates = project_doc["candidates"]
    print(f"Total candidates in project: {len(candidates)}")

    # Load required roles from SameRolesCache
    roles_doc = same_roles_col.find_one({"project_title": project_title})
    if not roles_doc:
        print(f"No role requirements found for '{project_title}' in SameRolesCache.")
        return

    required_roles = [r["role"] for r in roles_doc["roles"]]
    team_size = len(required_roles)
    print(f"Team size: {team_size}")
    print(f"Required roles in this project: {required_roles}\n")

    # Group candidates by role (from this project only)
    role_candidates = {role: [] for role in required_roles}
    for c in candidates:
        if c["role"] in role_candidates:
            role_candidates[c["role"]].append(c)

    print("Checking for missing roles and finding replacements...")
    
    # Handle missing roles with improved logic
    existing_team = []  # Track team members as we build
    used_names = set()
    
    for role in required_roles:
        if not role_candidates[role]:
            print(f"No candidates in project for role '{role}', searching for replacements...")
            
            # Find candidates based on compatibility with existing team
            replacement_candidates = find_missing_role_candidates(
                role, project_title, existing_team, used_names
            )
            
            if replacement_candidates:
                chosen = replacement_candidates[0]  # Best compatibility match
                role_candidates[role].append(chosen)
                existing_team.append(chosen)
                used_names.add(chosen["name"])
                
                # Check if from same project or external
                source = "same project" if any(
                    c["name"] == chosen["name"] for c in candidates
                ) else "other projects"
                print(f" -> Added {chosen['name']} from {source} for role '{role}'")
            else:
                print(f" -> No available candidate found for '{role}'. Skipping role.")

    # Remove roles with no candidates
    role_candidates = {role: cands for role, cands in role_candidates.items() if cands}
    available_roles = list(role_candidates.keys())

    if len(available_roles) < team_size:
        print(f"Adjusted team size due to missing roles: {len(available_roles)} roles available.")

    if len(available_roles) == 0:
        print("No roles have available candidates. Cannot form team.")
        return

    print(f"\nUsing optimized beam search to find best teams...")
    print(f"Roles with candidates: {available_roles}")

    # Use optimized beam search instead of brute force
    team_results = greedy_beam_search(role_candidates, available_roles, beam_width=5)

    if not team_results:
        print("No valid teams found.")
        return

    # Sort teams by compatibility score
    team_results.sort(key=lambda x: x[0], reverse=True)

    print(f"\nTop {min(5, len(team_results))} Compatible Teams:\n")
    top_teams_data = []
    
    for rank, (score, team) in enumerate(team_results[:5], start=1):
        print(f"{rank}) Compatibility Score: {score:.4f}")
        members_list = [{"name": m["name"], "role": m["role"]} for m in team]
        for m in members_list:
            print(f"   - {m['name']} ({m['role']})")
        print()
        top_teams_data.append({
            "score": score,
            "members": members_list
        })

    # Store in MongoDB
    highest_team = top_teams_data[0] if top_teams_data else None
    compatible_col.update_one(
        {"project_title": project_title},
        {
            "$set": {
                "project_title": project_title,
                "highest_scored_team": highest_team,
                "top_teams": top_teams_data,
            }
        },
        upsert=True
    )

    print(f"✅ Compatible teams saved for '{project_title}'")

if __name__ == "__main__":
    main()