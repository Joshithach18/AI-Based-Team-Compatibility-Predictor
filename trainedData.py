import json
import time
import requests
import pandas as pd
from typing import Optional
import os

# -----------------------
# Config
# -----------------------
API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "mistralai/mistral-7b-instruct"
TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

# Output files
TRAINING_DATA_FILE = "TrainingData.csv"
CHECKPOINT_FILE = "bootstrap_checkpoint.csv"

# -----------------------
# API Functions
# -----------------------
def get_personality_scores(summary: str, retry_count: int = 3) -> Optional[dict]:
    """Get Big Five scores from Mistral with retry logic."""
    prompt = f"""
You are a personality analysis expert.
Given the following behavioral summary of a person, return their Big Five personality trait scores
as JSON with keys: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism.
Each value must be a floating-point number between 0 and 1.

Behavioral Summary:
\"\"\"{summary}\"\"\"

Return only JSON, no explanations.
"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }
    
    for attempt in range(retry_count):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 429:  # Rate limit
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"   Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            if response.status_code != 200:
                print(f"   API error {response.status_code}: {response.text[:100]}")
                return None

            content = response.json()["choices"][0]["message"]["content"].strip()
            
            # Clean up the response - handle various formats
            # Remove <s> tags
            if content.startswith("<s>"):
                content = content[3:].strip()
            
            # Remove [OUT] tags
            if "[OUT]" in content:
                content = content.split("[OUT]")[1].split("[/OUT]")[0].strip()
            elif "[/OUT]" in content:
                content = content.split("[/OUT]")[0].strip()
            
            # Remove markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Remove any remaining special tokens
            content = content.replace("<s>", "").replace("</s>", "").strip()
            
            scores = json.loads(content)
            
            # Validate all traits exist and are in [0, 1]
            valid = True
            for trait in TRAITS:
                if trait not in scores:
                    valid = False
                    break
                if not isinstance(scores[trait], (int, float)) or not (0 <= scores[trait] <= 1):
                    valid = False
                    break
            
            if valid:
                return scores
            else:
                print(f"   Invalid scores format: {scores}")
                return None
                
        except json.JSONDecodeError as e:
            print(f"   JSON parse error: {e}")
            print(f"   Response: {content[:200]}")
            return None
        except Exception as e:
            print(f"   Error: {e}")
            if attempt == retry_count - 1:
                return None
            time.sleep(1)
    
    return None

# -----------------------
# Data Loading
# -----------------------
def load_dataset(file_path: str = "UpdatedDataset.csv") -> pd.DataFrame:
    """Load the dataset with behavioral summaries."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please provide the dataset.")
    
    df = pd.read_csv(file_path)
    
    # Rename column if needed
    if "BehaviouralSummary" in df.columns:
        df.rename(columns={"BehaviouralSummary": "BehavioralSummary"}, inplace=True)
    
    if "BehavioralSummary" not in df.columns:
        raise ValueError("Dataset must have 'BehavioralSummary' column")
    
    # Filter out empty summaries
    df = df[df["BehavioralSummary"].notna()].copy()
    df["BehavioralSummary"] = df["BehavioralSummary"].astype(str).str.strip()
    df = df[df["BehavioralSummary"] != ""]
    
    return df

def load_checkpoint() -> set:
    """Load already processed names from checkpoint."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    
    df = pd.read_csv(CHECKPOINT_FILE)
    if "Name" in df.columns:
        return set(df["Name"].tolist())
    return set()

def save_checkpoint(df: pd.DataFrame):
    """Save progress to checkpoint file."""
    df.to_csv(CHECKPOINT_FILE, index=False)
    print(f"   Checkpoint saved: {len(df)} samples")

# -----------------------
# Sampling Strategy
# -----------------------
def select_diverse_samples(df: pd.DataFrame, n_samples: int = 200, existing_names: set = None) -> pd.DataFrame:
    """
    Select diverse behavioral summaries for training.
    Uses simple length-based diversity to capture different summary styles.
    """
    if existing_names is None:
        existing_names = set()
    
    # Filter out already processed
    if "Name" in df.columns:
        df = df[~df["Name"].isin(existing_names)].copy()
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Add summary length as diversity metric
    df["summary_length"] = df["BehavioralSummary"].str.len()
    
    # Sort by length to get diverse samples
    df = df.sort_values("summary_length")
    
    # Take samples from different length quantiles
    if len(df) <= n_samples:
        selected = df
    else:
        # Take evenly spaced samples
        indices = [int(i * len(df) / n_samples) for i in range(n_samples)]
        selected = df.iloc[indices].copy()
    
    return selected.drop(columns=["summary_length"])

# -----------------------
# Bootstrap Process
# -----------------------
def bootstrap_training_data(n_samples: int = 200, resume: bool = True):
    """
    Generate training data by scoring behavioral summaries with the LLM.
    
    Args:
        n_samples: Number of samples to generate
        resume: Whether to resume from checkpoint
    """
    print("=" * 60)
    print("BOOTSTRAP TRAINING DATA GENERATOR")
    print("=" * 60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    df = load_dataset()
    print(f"   Found {len(df)} samples with behavioral summaries")
    
    # Load checkpoint if resuming
    existing_names = set()
    training_data = []
    
    if resume and os.path.exists(CHECKPOINT_FILE):
        print("\n2. Loading checkpoint...")
        existing_names = load_checkpoint()
        checkpoint_df = pd.read_csv(CHECKPOINT_FILE)
        training_data = checkpoint_df.to_dict('records')
        print(f"   Resuming from {len(existing_names)} completed samples")
    else:
        print("\n2. Starting fresh (no checkpoint)")
    
    # Select diverse samples
    print(f"\n3. Selecting {n_samples} diverse samples...")
    samples = select_diverse_samples(df, n_samples, existing_names)
    
    if len(samples) == 0:
        print("   No new samples to process!")
        print(f"   Training data already complete: {TRAINING_DATA_FILE}")
        return
    
    print(f"   Selected {len(samples)} new samples to score")
    
    # Score samples
    print(f"\n4. Scoring samples (this will take time)...")
    print(f"   Estimated time: ~{len(samples) * 3} seconds ({len(samples) * 3 / 60:.1f} minutes)")
    print("   Progress will be checkpointed every 10 samples\n")
    
    successful = 0
    failed = 0
    
    for idx, row in enumerate(samples.itertuples(), 1):
        name = getattr(row, "Name", f"Sample_{idx}")
        summary = row.BehavioralSummary
        
        print(f"[{idx}/{len(samples)}] Scoring: {name[:30]}...", end=" ")
        
        scores = get_personality_scores(summary)
        
        if scores:
            training_data.append({
                "Name": name,
                "BehavioralSummary": summary,
                **scores
            })
            successful += 1
            print("✓")
        else:
            failed += 1
            print("✗ (failed)")
        
        # Checkpoint every 10 samples
        if idx % 10 == 0:
            temp_df = pd.DataFrame(training_data)
            save_checkpoint(temp_df)
        
        # Rate limiting: small delay between requests
        time.sleep(0.5)
    
    # Final save
    print(f"\n5. Saving final training data...")
    final_df = pd.DataFrame(training_data)
    final_df.to_csv(TRAINING_DATA_FILE, index=False)
    
    print("\n" + "=" * 60)
    print("BOOTSTRAP COMPLETE")
    print("=" * 60)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total samples: {len(final_df)}")
    print(f"💾 Saved to: {TRAINING_DATA_FILE}")
    print("\nNext steps:")
    print(f"  1. Review {TRAINING_DATA_FILE} for quality")
    print("  2. Run: python behavioral_scoring.py train")
    print("=" * 60)
    
    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"🧹 Cleaned up checkpoint file")

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    import sys
    
    # Parse arguments
    n_samples = 200  # Default
    resume = True
    
    if len(sys.argv) > 1:
        try:
            n_samples = int(sys.argv[1])
        except ValueError:
            print(f"Invalid sample count: {sys.argv[1]}")
            sys.exit(1)
    
    if len(sys.argv) > 2 and sys.argv[2] == "--no-resume":
        resume = False
    
    print(f"\nConfiguration:")
    print(f"  Samples: {n_samples}")
    print(f"  Resume: {resume}")
    print(f"  Model: {MODEL}")
    print()
    
    try:
        bootstrap_training_data(n_samples=n_samples, resume=resume)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print(f"   Progress saved to: {CHECKPOINT_FILE}")
        print(f"   Resume with: python bootstrap_training_data.py {n_samples}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()