# match.py (or app.py)

# import os, json
# from sentence_transformers import SentenceTransformer, util
# import openai
# from dotenv import load_dotenv

# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # ─── Configuration ────────────────────────────────────────────────────────────
# MODEL_NAME = "all-MiniLM-L6-v2"
# TOP_K = 3
# model = SentenceTransformer(MODEL_NAME)

# # ─── Load Users ────────────────────────────────────────────────────────────────
# def load_users(filename="users.json"):
#     # locate this file next to the script, not CWD
#     base = os.path.dirname(__file__)
#     path = os.path.join(base, filename)
#     with open(path, "r", encoding="utf-8") as f:
#         users = json.load(f)
#     print(f"[DEBUG] Loaded {len(users)} users from {path}")
#     return users

# # ─── Matching Logic ─────────────────────────────────────────────────────────────
# def find_matches(current_user, all_users, top_k=TOP_K):
#     input_emb = model.encode(current_user["description"], convert_to_tensor=True)
#     candidates = []
#     for user in all_users:
#         if user["name"] == current_user["name"]:
#             continue
#         if user["gender"] != current_user["looking_for"]:
#             continue
#         emb = model.encode(user["description"], convert_to_tensor=True)
#         score = util.pytorch_cos_sim(input_emb, emb).item()
#         candidates.append((user, score))
#     candidates.sort(key=lambda x: x[1], reverse=True)
#     print(f"[DEBUG] Found {len(candidates)} candidates matching gender '{current_user['looking_for']}'")
#     return candidates[:top_k]

# # ─── GPT Helpers ────────────────────────────────────────────────────────────────
# def generate_match_explanation(a, b):
#     # …
#     # same as before
#     return "…"  # stub

# def generate_convo_starter(a, b):
#     # …
#     return "…"

# # ─── Main ───────────────────────────────────────────────────────────────────────
# def main():
#     users = load_users()
#     current_user = {
#         "name": "You",
#         "age": 21,
#         "gender": "Male",
#         "interests": ["poetry", "travel", "reading"],
#         "description": "I love deep conversations, stargazing, and exploring unknown cities.",
#         "looking_for": "Female"
#     }

#     matches = find_matches(current_user, users)
#     print(f"\nTop {TOP_K} matches for {current_user['name']}:\n" + "-"*40)
#     if not matches:
#         print("⚠️ No matches found! Check your users.json and gender filters.")
#     for idx, (user, score) in enumerate(matches, 1):
#         print(f"{idx}. {user['name']} — {round(score*100,2)}%")

# if __name__ == "__main__":
#     main()
# import os
# import json
# import numpy as np
# from typing import List, Dict
# from sentence_transformers import SentenceTransformer, util
# from openai import OpenAI
# from sklearn.preprocessing import MinMaxScaler
# from dotenv import load_dotenv

# # Load environment & initialise client
# load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # ─── Configuration ────────────────────────────────────────────────────────────
# EMBEDDING_MODEL = "all-mpnet-base-v2"
# GPT_MODEL = "gpt-3.5-turbo"
# MATCH_WEIGHTS   = {"interests": 0.3, "bio_similarity": 0.4, "personality": 0.3}

# # Load embedding model
# embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# def normalize_scores(scores: List[float]) -> List[float]:
#     arr = np.array(scores).reshape(-1, 1)
#     return MinMaxScaler().fit_transform(arr).flatten().tolist()


# class MatchingAgent:
#     def __init__(self, users_data: List[Dict]):
#         self.users = users_data

#     def _get_embedding(self, text: str) -> np.ndarray:
#         return embedding_model.encode(text, convert_to_tensor=False)

#     def _calculate_bio_similarity(self, user1: Dict, user2: Dict) -> float:
#         emb1 = self._get_embedding(user1["bio"])
#         emb2 = self._get_embedding(user2["bio"])
#         return util.pytorch_cos_sim(emb1, emb2).item()

#     def _calculate_interests_match(self, user1: Dict, user2: Dict) -> float:
#         """Jaccard similarity between interest tags."""
#         interests1 = set(user1.get("interests", []))
#         interests2 = set(user2.get("interests", []))
#         if not interests1 or not interests2:
#             return 0.0
#         return len(interests1 & interests2) / len(interests1 | interests2)

#     def _gpt_personality_analysis(self, user1: Dict, user2: Dict) -> float:
#         prompt = f"""
# Analyze compatibility between two dating profiles:
# User A: {user1["bio"]} | Interests: {", ".join(user1["interests"])}
# User B: {user2["bio"]} | Interests: {", ".join(user2["interests"])}
# Rate their compatibility (0-1) based on values, communication style, and long-term potential.
# Return ONLY a number between 0 and 1.
# """
#         resp = client.chat.completions.create(
#             model=GPT_MODEL,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.0
#         )
#         return float(resp.choices[0].message.content.strip())

#     def get_matches(self, target_user: Dict, top_k: int = 5) -> List[Dict]:
#         candidates = []
#         for user in self.users:
#             if user["id"] == target_user["id"]:
#                 continue

#             scores = {
#                 "bio_similarity": self._calculate_bio_similarity(target_user, user),
#                 "interests": self._calculate_interests_match(target_user, user),
#                 "personality": self._gpt_personality_analysis(target_user, user)
#             }

#             composite = sum(scores[f] * MATCH_WEIGHTS[f] for f in MATCH_WEIGHTS)
#             candidates.append({"user": user, "score": composite, "details": scores})

#         candidates.sort(key=lambda x: x["score"], reverse=True)
#         return candidates[:top_k]
# if __name__ == "__main__":

#     # Load your JSON users (or your real dataset)
#     from pathlib import Path

# # Get path to the current script directory
#     BASE_DIR = Path(__file__).resolve().parent
#     USERS_PATH = BASE_DIR / "users.json"

#     with open(USERS_PATH, "r", encoding="utf-8") as f:
#         users = json.load(f)


#     target_user = {
#         "id": 0,
#         "name": "You",
#         "bio": "I love the outdoors and live music festivals.",
#         "interests": ["hiking", "music", "festivals"],
#         "gender": "Male",
#         "looking_for": "Female"
#     }

#     agent = MatchingAgent(users)
#     matches = agent.get_matches(target_user, top_k=3)

#     print(f"\nTop matches for {target_user['name']}:\n" + "-"*40)
#     for i, match in enumerate(matches, 1):
#         usr = match["user"]
#         score = match["score"]
#         details = match["details"]
#         print(f"{i}. {usr['name']} — {score:.2%}")
#         print(f"   • Bio similarity:   {details['bio_similarity']:.2%}")
#         print(f"   • Interests match:  {details['interests']:.2%}")

#%% import os
# import sys
# import json
# import numpy as np
# from pathlib import Path
# from typing import List, Dict
# from sentence_transformers import SentenceTransformer, util
# from sklearn.preprocessing import MinMaxScaler
# from openai import OpenAI, RateLimitError, APIConnectionError, APIError

# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # 1. ABSOLUTELY RELIABLE API KEY LOADING
# # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# def load_api_key():
#     """Triple-check key loading from all possible sources"""
#     # Method 1: Direct from .env file (bypasses dotenv)
#     env_path = Path(__file__).parent / '.env'
#     if env_path.exists():
#         with open(env_path, 'r') as f:
#             for line in f:
#                 if line.strip() and 'OPENAI_API_KEY' in line:
#                     key = line.split('=', 1)[1].strip()
#                     os.environ['OPENAI_API_KEY'] = key
#                     return key

#     # Method 2: Environment variables
#     if 'OPENAI_API_KEY' in os.environ:
#         return os.environ['OPENAI_API_KEY']

#     # Method 3: Hardcoded fallback (TEMPORARY - remove after testing)
#     TEMP_KEY = "sk-your-actual-key-here"  # REPLACE WITH YOUR KEY
#     if TEMP_KEY.startswith("sk-"):
#         os.environ['OPENAI_API_KEY'] = TEMP_KEY
#         return TEMP_KEY

#     raise RuntimeError("""
#     ❌ No OpenAI API key found!
#     1. Create a .env file with OPENAI_API_KEY=your-key-here
#     2. Or set environment variable
#     3. Or temporarily uncomment the TEMP_KEY above
#     """)

# # Initialize clients
# try:
#     api_key = load_api_key()
#     client = OpenAI(api_key=api_key)
    
#     # Immediate connection test
#     client.models.list(timeout=5)  # Fail fast if key is invalid
#     print("✅ OpenAI connection successfully established!")
    
# except Exception as e:
#     print(f"❌ Critical OpenAI initialization error: {str(e)}")
#     print("Possible solutions:")
#     print("1. Verify your .env file exists in the same folder as match.py")
#     print("2. Check key validity at https://platform.openai.com/account/api-keys")
#     print("3. Ensure no network restrictions/firewalls")
#     sys.exit(1)

# embedding_model = SentenceTransformer("all-mpnet-base-v2")

# # ─── Matching Agent Class ────────────────────────────────────────────────────
# class MatchingAgent:
#     def __init__(self, users_data: List[Dict]):
#         self.users = users_data

#     def _get_embedding(self, text: str) -> np.ndarray:
#         """Convert text to embedding using SentenceTransformer."""
#         return embedding_model.encode(text, convert_to_tensor=False)

#     def _calculate_bio_similarity(self, u1: Dict, u2: Dict) -> float:
#         """Compute cosine similarity between bios."""
#         emb1 = self._get_embedding(u1["bio"])
#         emb2 = self._get_embedding(u2["bio"])
#         return util.pytorch_cos_sim(emb1, emb2).item()

#     def _calculate_interests_match(self, u1: Dict, u2: Dict) -> float:
#         """Jaccard similarity between interest tags."""
#         i1, i2 = set(u1.get("interests", [])), set(u2.get("interests", []))
#         return len(i1 & i2) / len(i1 | i2) if (i1 and i2) else 0.0

#     def _batch_personality_scores(self, pairs: List[tuple]) -> List[float]:
#         """Batch process personality analysis with GPT-4."""
#         try:
#             prompts = [
#                 f"Analyze compatibility (0-1 scale):\n"
#                 f"User A: {u1['bio']}\nInterests: {', '.join(u1.get('interests', []))}\n"
#                 f"User B: {u2['bio']}\nInterests: {', '.join(u2.get('interests', []))}\n"
#                 "Return ONLY a number between 0 and 1."
#                 for u1, u2 in pairs
#             ]
            
#             responses = client.chat.completions.create(
#                 model="gpt-4",
#                 messages=[{"role": "user", "content": p} for p in prompts],
#                 temperature=0.0,
#                 max_tokens=10
#             )
#             return [float(r.choices[0].message.content.strip()) for r in responses]
            
#         except (RateLimitError, APIConnectionError, APIError) as e:
#             print(f"⚠️ OpenAI error: {str(e)} - Using default scores")
#             return [0.5] * len(pairs)  # Fallback

#     def get_matches(self, target_user: Dict, top_k: int = 5) -> List[Dict]:
#         """Optimized matching pipeline with batch processing."""
#         # Step 1: Pre-filter candidates
#         candidates = [
#             u for u in self.users 
#             if u["id"] != target_user["id"] 
#             and u["gender"] == target_user["looking_for"]
#         ]
        
#         # Step 2: Compute fast metrics (embeddings + interests)
#         for u in candidates:
#             u["_bio_sim"] = self._calculate_bio_similarity(target_user, u)
#             u["_int_sim"] = self._calculate_interests_match(target_user, u)
        
#         # Step 3: Shortlist top candidates to limit GPT calls
#         candidates.sort(
#             key=lambda x: 0.6*x["_bio_sim"] + 0.4*x["_int_sim"], 
#             reverse=True
#         )
#         shortlist = candidates[:min(top_k*2, len(candidates))]  # Wider pool
        
#         # Step 4: Batch process personality analysis
#         pairs = [(target_user, u) for u in shortlist]
#         pers_scores = self._batch_personality_scores(pairs)
        
#         # Step 5: Final scoring
#         results = []
#         for u, p in zip(shortlist, pers_scores):
#             composite_score = 0.5*u["_bio_sim"] + 0.3*u["_int_sim"] + 0.2*p
#             results.append({
#                 "user": u,
#                 "score": composite_score,
#                 "details": {
#                     "bio_similarity": u["_bio_sim"],
#                     "interests": u["_int_sim"],
#                     "personality": p
#                 }
#             })
        
#         return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

# # ─── Example Usage ────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     # Load user data
#     base_dir = Path(__file__).parent
#     with open(base_dir / "users.json", "r", encoding="utf-8") as f:
#         users = json.load(f)
    
#     # Define target user
#     target_user = {
#         "id": 0,
#         "name": "You",
#         "bio": "I love the outdoors and live music festivals.",
#         "interests": ["hiking", "music", "festivals"],
#         "gender": "Male",
#         "looking_for": "Female"
#     }
    
#     # Get matches
#     agent = MatchingAgent(users)
#     matches = agent.get_matches(target_user, top_k=3)
    
#     # Print results
#     print(f"\nTop matches for {target_user['name']}:\n" + "-"*40)
#     for i, match in enumerate(matches, 1):
#         print(f"{i}. {match['user']['name']} — Score: {match['score']:.2%}")
#         print(f"   • Bio Similarity: {match['details']['bio_similarity']:.2%}")
#         print(f"   • Interests Match: {match['details']['interests']:.2%}")
#         print(f"   • Personality Fit: {match['details']['personality']:.2%}\n")
#%%
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MinMaxScaler
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError  # Correct imports for latest version

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. INITIALIZATION (WITH ERROR HANDLING)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_api_key():
    """Load API key from .env file or environment variables"""
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip() and 'OPENAI_API_KEY' in line:
                    return line.split('=', 1)[1].strip()
    
    return os.getenv("OPENAI_API_KEY")

try:
    api_key = load_api_key()
    if not api_key:
        raise ValueError("API key not found in .env or environment variables")
    
    client = OpenAI(api_key=api_key)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Test API connection
    test_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say 'TEST'"}],
        max_tokens=5,
        timeout=10
    )
    print("✅ API connection test passed")
    
except (APIConnectionError, RateLimitError, APIStatusError) as e:
    print(f"❌ API Error: {str(e)}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Initialization error: {str(e)}")
    sys.exit(1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. MATCHING AGENT (WITH PROPER ERROR HANDLING)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MatchingAgent:
    def __init__(self, users_data: List[Dict]):
        self.users = users_data

    def _get_embedding(self, text: str) -> np.ndarray:
        return embedding_model.encode(text, convert_to_tensor=False)

    def _calculate_bio_similarity(self, u1: Dict, u2: Dict) -> float:
        emb1 = self._get_embedding(u1["bio"])
        emb2 = self._get_embedding(u2["bio"])
        return util.pytorch_cos_sim(emb1, emb2).item()

    def _calculate_interests_match(self, u1: Dict, u2: Dict) -> float:
        i1, i2 = set(u1.get("interests", [])), set(u2.get("interests", []))
        return len(i1 & i2) / len(i1 | i2) if (i1 and i2) else 0.0

    def _batch_personality_scores(self, pairs: List[tuple]) -> List[float]:
        """Fixed error handling for GPT-3.5 Turbo"""
        try:
            scores = []
            for u1, u2 in pairs:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "Rate compatibility 0-1 based on profiles. Respond ONLY with a number."
                        },
                        {
                            "role": "user",
                            "content": f"""
                            User A: {u1['bio']}
                            Interests: {', '.join(u1.get('interests', []))}
                            
                            User B: {u2['bio']}
                            Interests: {', '.join(u2.get('interests', []))}
                            
                            Compatibility score (0-1)?
                            """
                        }
                    ],
                    temperature=0.0,
                    max_tokens=10,
                    timeout=15
                )
                scores.append(float(response.choices[0].message.content.strip()))
            return scores
            
        except (APIConnectionError, RateLimitError, APIStatusError) as e:
            print(f"⚠️ API Error: {str(e)} - Using default scores")
            return [0.5] * len(pairs)
        except Exception as e:
            print(f"⚠️ Unexpected error: {str(e)} - Using default scores")
            return [0.5] * len(pairs)

    def get_matches(self, target_user: Dict, top_k: int = 5) -> List[Dict]:
        candidates = [
            u for u in self.users 
            if u["id"] != target_user["id"] 
            and u["gender"] == target_user["looking_for"]
        ]
        
        for u in candidates:
            u["_bio_sim"] = self._calculate_bio_similarity(target_user, u)
            u["_int_sim"] = self._calculate_interests_match(target_user, u)
        
        candidates.sort(key=lambda x: 0.6*x["_bio_sim"] + 0.4*x["_int_sim"], reverse=True)
        shortlist = candidates[:min(top_k*2, len(candidates))]
        
        pairs = [(target_user, u) for u in shortlist]
        pers_scores = self._batch_personality_scores(pairs)
        
        results = []
        for u, p in zip(shortlist, pers_scores):
            results.append({
                "user": u,
                "score": 0.5*u["_bio_sim"] + 0.3*u["_int_sim"] + 0.2*p,
                "details": {
                    "bio_similarity": u["_bio_sim"],
                    "interests": u["_int_sim"],
                    "personality": p
                }
            })
        
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. MAIN EXECUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    users = [
        {
            "id": 1,
            "name": "Alex",
            "bio": "Software engineer who loves hiking and indie music",
            "interests": ["programming", "hiking", "music"],
            "gender": "Female",
            "looking_for": "Male"
        },
        {
            "id": 2,
            "name": "Jordan",
            "bio": "Outdoorsy photographer who enjoys jazz festivals",
            "interests": ["photography", "travel", "music"],
            "gender": "Female",
            "looking_for": "Male"
        }
    ]
    
    target_user = {
        "id": 0,
        "name": "You",
        "bio": "Tech enthusiast who loves nature and live music",
        "interests": ["technology", "hiking", "concerts"],
        "gender": "Male",
        "looking_for": "Female"
    }
    
    try:
        agent = MatchingAgent(users)
        matches = agent.get_matches(target_user, top_k=3)
        
        print(f"\n🔍 Top matches for {target_user['name']}:")
        for i, match in enumerate(matches, 1):
            print(f"\n{i}. {match['user']['name']} ({match['score']:.0%} match)")
            print(f"   Bio: {match['user']['bio']}")
            print(f"   ➤ Bio similarity: {match['details']['bio_similarity']:.0%}")
            print(f"   ➤ Shared interests: {match['details']['interests']:.0%}")
            print(f"   ➤ Personality fit: {match['details']['personality']:.0%}")
            
    except Exception as e:
        print(f"❌ Fatal error in matching: {str(e)}")
       
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. MATCHING AGENT IMPLEMENTATION (GPT-3.5 Turbo)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

