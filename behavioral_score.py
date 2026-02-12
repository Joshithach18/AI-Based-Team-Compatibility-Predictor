import os
import sys

# CRITICAL: Disable TensorFlow before ANY imports
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Block tensorflow imports
class TFBlocker:
    def find_module(self, fullname, path=None):
        if fullname.startswith('tensorflow'):
            return self
        return None
    
    def load_module(self, fullname):
        raise ImportError(f"TensorFlow is disabled for this script")

sys.meta_path.insert(0, TFBlocker())

import json
import pickle
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# -----------------------
# Config
# -----------------------
TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, CPU-friendly, 384-dim embeddings
MODEL_PATH = "behavioral_model.pkl"

# Mongo
client = MongoClient("mongodb://localhost:27017/")
db = client["AISubmodule"]
top_scores_col = db["TopTechnicalScores"]
behavior_col = db["BehavioralScores"]

# -----------------------
# Training Functions
# -----------------------
def train_model():
    """
    Train a regression model to predict Big Five scores from behavioral summaries.
    Run this ONCE with labeled training data.
    """
    print("Loading training data...")
    df = pd.read_csv("TrainingData.csv")  # Must have: BehavioralSummary, Openness, Conscientiousness, etc.
    
    # Validate columns
    required_cols = ["BehavioralSummary"] + TRAITS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in TrainingData.csv: {missing}")
    
    # Remove rows with missing data
    df = df.dropna(subset=required_cols)
    print(f"Training samples: {len(df)}")
    
    # Load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    
    # Generate embeddings
    print("Generating embeddings...")
    summaries = df["BehavioralSummary"].tolist()
    X = embedder.encode(summaries, show_progress_bar=True, convert_to_numpy=True)
    
    # Target scores
    y = df[TRAITS].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train one Ridge regressor for all traits (multi-output)
    print("Training Ridge regressor...")
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error on test set: {mae:.4f}")
    
    # Per-trait MAE
    for i, trait in enumerate(TRAITS):
        trait_mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        print(f"  {trait}: {trait_mae:.4f}")
    
    # Save model and embedder
    print(f"Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"embedder": embedder, "regressor": model, "traits": TRAITS}, f)
    
    print("✅ Training complete!")

# -----------------------
# Inference Functions
# -----------------------
def load_model():
    """Load trained model from disk."""
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["embedder"], data["regressor"], data["traits"]

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

def calculate_overall_score(scores: dict) -> float:
    """Calculate overall behavioral score (inverted neuroticism)."""
    if not scores:
        return None
    try:
        openness = scores.get("Openness", 0)
        conscientiousness = scores.get("Conscientiousness", 0)
        extraversion = scores.get("Extraversion", 0)
        agreeableness = scores.get("Agreeableness", 0)
        neuro_adj = 1 - scores.get("Neuroticism", 0)

        overall = (openness + conscientiousness + extraversion + agreeableness + neuro_adj) / 5
        return round(overall, 3)
    except Exception:
        return None

def select_project_title() -> str:
    """Interactive project selection."""
    titles = top_scores_col.distinct("project_title")
    if not titles:
        raise RuntimeError("No project titles found in TopTechnicalScores.")
    print("\nAvailable Projects:")
    for i, t in enumerate(titles, 1):
        print(f"{i}. {t}")
    idx = int(input(f"\nSelect a project (1-{len(titles)}): ")) - 1
    if idx < 0 or idx >= len(titles):
        raise RuntimeError("Invalid selection.")
    return titles[idx]

def load_employee_summaries(names: list) -> pd.DataFrame:
    """Load behavioral summaries for given employees."""
    df = pd.read_csv("UpdatedDataset.csv")
    df.rename(columns={"BehaviouralSummary": "BehavioralSummary"}, inplace=True)
    needed_cols = ["Name", "BehavioralSummary"]
    for col in needed_cols:
        if col not in df.columns:
            raise RuntimeError(f"Column '{col}' not found in UpdatedDataset.csv")
    return df[df["Name"].isin(names)][needed_cols].copy()

# -----------------------
# Main Inference Pipeline
# -----------------------
def main():
    """Score candidates for a selected project using the trained model."""
    # Load model
    print("Loading trained model...")
    embedder, regressor, traits = load_model()
    print(f"✅ Model loaded (traits: {traits})")
    
    # Select project
    project_title = select_project_title()
    print(f"\nSelected Project: {project_title}")

    # Get top technical scorers
    doc = top_scores_col.find_one({"project_title": project_title})
    if not doc or "roles" not in doc:
        print("No roles found for this project.")
        return

    # Collect candidates
    candidates_info = []
    all_names = set()
    for role in doc["roles"]:
        for cand in role.get("top_candidates", []):
            if cand.get("name"):
                candidates_info.append({"name": cand["name"], "role": role["role"]})
                all_names.add(cand["name"])

    if not candidates_info:
        print("No top candidates found.")
        return

    # Load summaries
    df_summaries = load_employee_summaries(list(all_names))

    # Score candidates (FAST - no API calls!)
    final_candidates = []
    print("\nScoring candidates...")
    for c in candidates_info:
        summary_row = df_summaries[df_summaries["Name"] == c["name"]]
        if summary_row.empty:
            print(f"⚠️  Skipping {c['name']}: no BehavioralSummary found.")
            continue

        summary = str(summary_row.iloc[0]["BehavioralSummary"]).strip()
        if not summary:
            print(f"⚠️  Skipping {c['name']}: empty BehavioralSummary.")
            continue

        try:
            scores = get_personality_scores_fast(summary, embedder, regressor)
            overall_score = calculate_overall_score(scores)
            final_candidates.append({
                "name": c["name"],
                "role": c["role"],
                "scores": scores,
                "overall_behavioral_score": overall_score
            })
            print(f"✓ {c['name']}: {overall_score:.3f}")
        except Exception as e:
            print(f"❌ Error scoring {c['name']}: {e}")

    # Save to MongoDB
    behavior_col.update_one(
        {"project_title": project_title},
        {"$set": {
            "project_title": project_title,
            "candidates": final_candidates,
        }},
        upsert=True
    )

    print(f"\n✅ Done! Scored {len(final_candidates)} candidates in milliseconds.")
    print(f"   Results saved to BehavioralScores collection.")

# -----------------------
# Entry Point
# -----------------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Training mode: python script.py train
        train_model()
    else:
        # Inference mode (default)
        main()