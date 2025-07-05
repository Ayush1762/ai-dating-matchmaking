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
        


import streamlit as st
import json
import time
 # Your existing matching function

# 1. Configure page
st.set_page_config(page_title="AI Matchmaker", layout="centered")

# 2. Load sample profiles

from pathlib import Path

# 1. Get absolute path to users.json
current_dir = Path(__file__).parent.resolve()
users_path = current_dir / "users.json"

# 2. Load data with error handling
try:
    with open(users_path, "r") as f:
        users = json.load(f)
except FileNotFoundError:
    st.error("❌ users.json not found! Using demo data.")
    users = [  # Fallback data
        {
            "id": 1, "name": "Demo User",
            "bio": "Love hiking and coding",
            "interests": ["tech", "nature"],
            "gender": "Female",
            "looking_for": ["Male"]
        }
    ]
except json.JSONDecodeError:
    st.error("❌ Invalid users.json format! Using demo data.")
    users = [{"name": "Backup User", "bio": "Default profile"}]

# 3. Your Streamlit UI code
import streamlit as st
import json
import time
from pathlib import Path
from match import MatchingAgent  # Your backend class

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. PAGE CONFIG & DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="AI Matchmaker", 
    page_icon="💘", 
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_users():
    try:
        users_path = Path(__file__).parent / "users.json"
        with open(users_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.error("❌ users.json not found or invalid! Using fallback demo data.")
        return [
            {
                "id": 1, "name": "Alex",
                "bio": "Software engineer who loves hiking and indie music",
                "interests": ["programming", "hiking", "music"],
                "gender": "Female", "looking_for": "Male", "age": 24
            },
            {
                "id": 2, "name": "Jordan",
                "bio": "Outdoorsy photographer who enjoys jazz festivals",
                "interests": ["photography", "travel", "music"],
                "gender": "Female", "looking_for": "Male", "age": 26
            }
        ]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. PROFILE CREATION FORM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def profile_form():
    with st.form("profile_form", border=True):
        st.title("💖 AI Matchmaker")
        st.write("Complete your profile to find compatible matches.")
        
        cols = st.columns(2)
        with cols[0]:
            name = st.text_input("Your Name", placeholder="John Doe")
            age = st.slider("Age", 18, 99, 25)
        with cols[1]:
            gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
            looking_for = st.multiselect(
                "Interested In", 
                ["Male", "Female", "Non-binary"],
                default=["Female"]
            )
        
        bio = st.text_area(
            "Describe yourself", 
            placeholder="I'm a tech enthusiast who loves hiking and concerts...",
            height=100
        )
        
        interests = st.multiselect(
            "Your Interests",
            ["Hiking", "Music", "Tech", "Photography", "Travel", "Sports"],
            default=["Hiking", "Music"]
        )
        
        submitted = st.form_submit_button("🔍 Find My Matches", use_container_width=True)
        
        if not name or not bio:
            st.warning("Please fill required fields.")
            submitted = False
            
    return submitted, {
        "id": -1,
        "name": name,
        "age": age,
        "gender": gender,
        "bio": bio,
        "interests": interests,
        "looking_for": looking_for[0] if looking_for else ""
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. MATCH DISPLAY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def show_match(match, rank):
    with st.container(border=True):
        cols = st.columns([1, 4])
        with cols[0]:
            st.image(
                f"https://api.dicebear.com/7.x/initials/svg?seed={match['user']['name']}",
                width=100
            )
            st.button(
                f"💌 Connect with {match['user']['name']}",
                key=f"connect_{rank}",
                use_container_width=True
            )
        
        with cols[1]:
            st.subheader(f"🌟 #{rank}: {match['user']['name']} ({match['score']:.0%} match)")
            st.caption(f"**Age:** {match['user'].get('age', 'N/A')}")
            st.write(match['user']['bio'])

            st.progress(match['score'], text="Overall Compatibility")

            with st.expander("📊 Match Breakdown"):
                st.markdown(f"➤ **Bio similarity:** {match['details']['bio_similarity']:.0%}")
                st.markdown(f"➤ **Shared interests:** {match['details']['interests']:.0%}")
                st.markdown(f"➤ **Personality fit:** {match['details']['personality']:.0%}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. MAIN APP LOGIC
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    users = load_users()
    submitted, user_profile = profile_form()
    
    if submitted:
        with st.spinner("🧠 Analyzing profiles using AI..."):
            try:
                agent = MatchingAgent(users)
                matches = agent.get_matches(user_profile, top_k=3)
                time.sleep(1)
                
                st.success(f"✅ Found {len(matches)} match{'es' if len(matches) != 1 else ''}!")
                st.divider()
                
                for i, match in enumerate(matches, 1):
                    show_match(match, i)
                    
                st.balloons()
                
            except Exception as e:
                st.error(f"⚠️ Matching failed: {str(e)}")

if __name__ == "__main__":
    main()
