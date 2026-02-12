import pandas as pd
from pymongo import MongoClient
import re
from tabulate import tabulate

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["AISubmodule"]
collection = db["SameRolesCache"]

def normalize_skill(skill):
    """Normalize skill names for better matching"""
    if not skill or pd.isna(skill):
        return ""
    
    skill = str(skill).strip().lower()
    skill = re.sub(r'[^\w\s]', '', skill)
    skill = re.sub(r'\s+', ' ', skill)
    return skill

def normalize_role(role):
    """Normalize role names for better matching"""
    if not role or pd.isna(role):
        return ""
    
    role = str(role).strip().lower()
    role_mappings = {
        # UI/UX Designer variations
        'ui/ux designer': 'ui/ux designer',
        'ux/ui designer': 'ui/ux designer',
        'ui designer': 'ui/ux designer',
        'ux designer': 'ui/ux designer',
        'user experience designer': 'ui/ux designer',
        'user interface designer': 'ui/ux designer',
        
        # DevOps Engineer variations
        'devops engineer': 'devops engineer',
        'dev ops engineer': 'devops engineer',
        'devops': 'devops engineer',
        
        # Backend Developer variations
        'backend developer': 'backend developer',
        'back end developer': 'backend developer',
        'back-end developer': 'backend developer',
        'server side developer': 'backend developer',
        'backend engineer': 'backend developer',
        
        # Frontend Developer variations
        'frontend developer': 'frontend developer',
        'front end developer': 'frontend developer',
        'front-end developer': 'frontend developer',
        'client side developer': 'frontend developer',
        'frontend engineer': 'frontend developer',
        
        # QA/Testing role variations - ALL MAP TO 'qa engineer'
        'qa engineer': 'qa engineer',
        'qa tester': 'qa engineer',
        'quality assurance engineer': 'qa engineer',
        'quality assurance tester': 'qa engineer',
        'test engineer': 'qa engineer',
        'testing engineer': 'qa engineer',
        'software tester': 'qa engineer',
        'automation engineer': 'qa engineer',
        'test automation engineer': 'qa engineer',
        
        # Data Science variations
        'data scientist': 'data scientist',
        'data analyst': 'data scientist',
        
        # ML/AI Engineer variations
        'ml engineer': 'machine learning engineer',
        'machine learning engineer': 'machine learning engineer',
        'ai engineer': 'machine learning engineer',
        'ai specialist': 'machine learning engineer',
        'artificial intelligence engineer': 'machine learning engineer',
        
        # Software Engineer variations
        'software engineer': 'software engineer',
        'software developer': 'software engineer',
        'application developer': 'software engineer',
        'programmer': 'software engineer',
        
        # Full Stack Developer variations
        'full stack developer': 'fullstack developer',
        'fullstack developer': 'fullstack developer',
        'full-stack developer': 'fullstack developer',
        'full stack engineer': 'fullstack developer',
        
        # Mobile Developer variations
        'mobile developer': 'mobile developer',
        'mobile app developer': 'mobile developer',
        'android developer': 'mobile developer',
        'ios developer': 'mobile developer',
        
        # Hardware Engineer variations
        'hardware engineer': 'hardware engineer',
        'electronics engineer': 'hardware engineer',
        'embedded engineer': 'hardware engineer',
    }
    
    return role_mappings.get(role, role)

def extract_skills_from_text(skills_text):
    """Extract individual skills from a comma-separated string"""
    if pd.isna(skills_text) or not skills_text:
        return []
    
    skills = []
    for skill in str(skills_text).split(','):
        normalized_skill = normalize_skill(skill)
        if normalized_skill:
            skills.append(normalized_skill)
    
    return skills

def calculate_skill_match_score(employee_skills, required_skills):
    """Calculate the technical score based on skill matching"""
    if not required_skills or not employee_skills:
        return 0.0, []
    
    normalized_required = [normalize_skill(skill) for skill in required_skills if skill]
    normalized_employee = [normalize_skill(skill) for skill in employee_skills if skill]
    
    if not normalized_required or not normalized_employee:
        return 0.0, []
    
    matched_skills = 0
    matched_skill_names = []
    
    for req_skill in normalized_required:
        skill_matched = False
        for emp_skill in normalized_employee:
            # Check for exact match first
            if req_skill == emp_skill:
                matched_skills += 1
                matched_skill_names.append(req_skill.title())
                skill_matched = True
                break
            # Check for partial matches
            elif (req_skill in emp_skill or emp_skill in req_skill):
                matched_skills += 1
                matched_skill_names.append(req_skill.title())
                skill_matched = True
                break
            # Check for word-level matches
            elif (any(word in emp_skill.split() for word in req_skill.split() if len(word) > 2) or
                  any(word in req_skill.split() for word in emp_skill.split() if len(word) > 2)):
                matched_skills += 1
                matched_skill_names.append(req_skill.title())
                skill_matched = True
                break
        
        if skill_matched:
            continue
    
    score = (matched_skills / len(normalized_required)) * 100
    return round(score, 2), list(set(matched_skill_names))

def is_experience_match(employee_exp, required_level):
    """Check if employee experience matches the required level"""
    try:
        emp_exp = int(employee_exp) if not pd.isna(employee_exp) else 0
    except (ValueError, TypeError):
        emp_exp = 0
    
    required_level = str(required_level).lower().strip()
    
    if required_level in ['experienced', 'senior', 'mid-level']:
        return emp_exp > 0
    elif required_level in ['newhire', 'new hire', 'junior', 'entry level']:
        return emp_exp == 0
    else:
        return True  # If unclear, include all

def get_project_roles():
    """Get all unique project titles from the MongoDB collection (avoid duplicates)"""
    try:
        # Fetch all project titles (distinct values)
        unique_titles = collection.distinct("project_title")
        
        if not unique_titles:
            print("No projects found in database. Please run aisubmodule.py first.")
            return None
        
        print("Available projects in database:")
        for i, title in enumerate(unique_titles, 1):
            print(f"{i}. {title}")
        
        while True:
            try:
                choice = int(input(f"\nSelect a project (1-{len(unique_titles)}): ")) - 1
                if 0 <= choice < len(unique_titles):
                    selected_title = unique_titles[choice]
                    print(f"\nSelected project: {selected_title}")
                    
                    # Fetch one document for this title to get roles
                    selected_project_doc = collection.find_one({"project_title": selected_title})
                    return selected_project_doc['roles'] if selected_project_doc else None
                else:
                    print(f"Please enter a number between 1 and {len(unique_titles)}")
            except ValueError:
                print("Please enter a valid number")
                
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def load_employee_data():
    """Load employee data from CSV file"""
    try:
        df = pd.read_csv('UpdatedDataset.csv')
        print(f"Loaded {len(df)} employees from newdataset.csv")
        
        # Validate required columns
        required_columns = ['Name', 'Role', 'Experience', 'TechnicalSkills']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns in CSV: {missing_columns}")
            return None
            
        return df
        
    except FileNotFoundError:
        print("newdataset.csv not found. Please ensure the file exists.")
        return None
    except Exception as e:
        print(f"Error loading employee data: {e}")
        return None

def calculate_technical_scores():
    employees_df = load_employee_data()
    if employees_df is None:
        return
    
    # Get project title and roles
    try:
        unique_titles = collection.distinct("project_title")
        print("Available projects in database:")
        for i, title in enumerate(unique_titles, 1):
            print(f"{i}. {title}")
        choice = int(input(f"\nSelect a project (1-{len(unique_titles)}): ")) - 1
        if choice < 0 or choice >= len(unique_titles):
            print("Invalid selection.")
            return
        project_title = unique_titles[choice]
        print(f"\nSelected project: {project_title}")
        selected_project_doc = collection.find_one({"project_title": project_title})
        project_roles = selected_project_doc['roles'] if selected_project_doc else None
    except Exception as e:
        print(f"Error fetching project: {e}")
        return

    if not project_roles:
        return

    print("\nRequired roles and skills:")
    for role_info in project_roles:
        skills_str = ', '.join(role_info.get('skills', []))
        print(f"- {role_info['role']} ({role_info.get('experience_level', 'Any')}): {skills_str}")
    
    print("\n" + "="*100)
    print("TOP 4 TECHNICAL SCORERS FOR EACH REQUIRED ROLE")
    print("="*100)
    
    overall_results = []
    top_scores_dict = {}  # NEW: to store top 4 per role for DB

    for role_info in project_roles:
        required_role = normalize_role(role_info['role'])
        required_experience_level = role_info.get('experience_level', 'Any')
        required_skills = role_info.get('skills', [])

        print(f"\nROLE: {role_info['role']} ({required_experience_level})")
        print(f"Required Skills: {', '.join(required_skills)}")
        print("-" * 80)

        role_results = []
        for _, employee in employees_df.iterrows():
            try:
                employee_role = normalize_role(employee.get('Role', ''))
                employee_exp = employee.get('Experience', 0)
                employee_skills = extract_skills_from_text(employee.get('TechnicalSkills', ''))

                role_match = (employee_role == required_role or 
                              employee_role in required_role or 
                              required_role in employee_role)
                
                if role_match and is_experience_match(employee_exp, required_experience_level):
                    technical_score, matched_skills = calculate_skill_match_score(employee_skills, required_skills)
                    role_results.append({
                        'Name': employee.get('Name', 'N/A'),
                        'Role': employee.get('Role', 'N/A'),
                        'Experience': employee_exp,
                        'Technical_Score': technical_score,
                        'Matched_Skills': ', '.join(matched_skills)
                    })
            except Exception as e:
                print(f"Error processing employee {employee.get('Name', 'Unknown')}: {e}")
                continue

        role_results.sort(key=lambda x: (x['Technical_Score'], x['Experience']), reverse=True)
        top_4_for_role = role_results[:4]
        top_scores_dict[role_info['role']] = top_4_for_role  # NEW: store for DB

        if top_4_for_role:
            table_data = []
            for i, result in enumerate(top_4_for_role, 1):
                table_data.append([
                    i,
                    result['Name'],
                    result['Role'],
                    f"{result['Experience']} years",
                    f"{result['Technical_Score']}%",
                    result['Matched_Skills'] if result['Matched_Skills'] else 'None'
                ])
            headers = ['Rank', 'Name', 'Current Role', 'Experience', 'Technical Score', 'Matched Skills']
            print(tabulate(table_data, headers=headers, tablefmt='grid'))
        else:
            print("No matching employees found for this role")

        overall_results.extend(role_results)

    if overall_results:
        print(f"Total matching candidates found: {len(overall_results)}")
        save_top_scores_to_db(project_title, top_scores_dict)  # NEW: Save to Mongo
    else:
        print("No matching candidates found for any role.")

def get_top_scorers_by_role(project_roles=None, employees_df=None):
    """
    Alternative function that returns results as structured data instead of printing
    Useful for API integration or further processing
    """
    if employees_df is None:
        employees_df = load_employee_data()
    if employees_df is None:
        return {}
    
    if project_roles is None:
        project_roles = get_project_roles()
    if not project_roles:
        return {}
    
    results_by_role = {}
    
    for role_info in project_roles:
        required_role = normalize_role(role_info['role'])
        required_experience_level = role_info.get('experience_level', 'Any')
        required_skills = role_info.get('skills', [])
        
        role_results = []
        
        for _, employee in employees_df.iterrows():
            try:
                employee_role = normalize_role(employee.get('Role', ''))
                employee_exp = employee.get('Experience', 0)
                employee_skills = extract_skills_from_text(employee.get('TechnicalSkills', ''))
                
                role_match = (employee_role == required_role or 
                            employee_role in required_role or 
                            required_role in employee_role)
                
                if role_match and is_experience_match(employee_exp, required_experience_level):
                    technical_score, matched_skills = calculate_skill_match_score(employee_skills, required_skills)
                    
                    role_results.append({
                        'name': employee.get('Name', 'N/A'),
                        'role': employee.get('Role', 'N/A'),
                        'experience': employee_exp,
                        'technical_score': technical_score,
                        'employee_skills': employee_skills,
                        'matched_skills': matched_skills
                    })
                    
            except Exception as e:
                continue
        
        # Sort and get top 4
        role_results.sort(key=lambda x: (x['technical_score'], x['experience']), reverse=True)
        results_by_role[role_info['role']] = role_results[:4]
    
    return results_by_role
def save_top_scores_to_db(project_title, top_scores_by_role):
    """
    Save the top technical scorers per role into the TopTechnicalScores collection.
    Only stores: project_title, roles -> name, technical_score
    """
    try:
        top_scores_collection = db["TopTechnicalScores"]

        # Prepare document
        document = {
            "project_title": project_title,
            "roles": []
        }

        for role_name, candidates in top_scores_by_role.items():
            role_entry = {
                "role": role_name,
                "top_candidates": [
                    {
                        "name": c["Name"] if "Name" in c else c.get("name", "N/A"),
                        "technical_score": c["Technical_Score"] if "Technical_Score" in c else c.get("technical_score", 0)
                    }
                    for c in candidates
                ]
            }
            document["roles"].append(role_entry)

        # Insert into DB
        top_scores_collection.insert_one(document)
        print(f"✅ Top technical scores saved for project: {project_title}")

    except Exception as e:
        print(f"❌ Error saving top scores to DB: {e}")

if __name__ == "__main__":
    print("Technical Score Calculator - Role-wise Analysis")
    print("=" * 60)
    calculate_technical_scores()