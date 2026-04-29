import pandas as pd
import pickle
import os

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File {file_path} not found.')
    return pd.read_csv(file_path)

def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Model {file_path} not found.')
    with open(file_path, 'rb') as f:
        return pickle.load(f)
from groq import Groq

def generate_ai_explanation(user_data, match_prob, api_key=None):
    prompt = f"You are an AI dating assistant. I am passing you profile data for a user: {user_data}. A machine learning model has predicted a {match_prob * 100:.1f}% chance that this profile will result in a match. In 2-3 short, human-sounding sentences, explain WHY this might be the case based on their features (like usage time, location, or orientation). Be encouraging but objective. Do not mention 'machine learning model'."
    if api_key:
        try:
            client = Groq(api_key=api_key)
            chat_completion = client.chat.completions.create(messages=[{'role': 'user', 'content': prompt}], model='llama-3.1-8b-instant', temperature=0.7, max_tokens=150)
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            print(f'Groq API Error: {e}')
            pass
    if match_prob > 0.7:
        explanation = f'[Fallback] Based on our analysis, this looks like a strong potential connection! The high match probability of {match_prob * 100:.1f}% is driven by shared interest patterns and compatible app usage behaviors.'
    elif match_prob > 0.4:
        explanation = f"[Fallback] There's some common ground here, with a {match_prob * 100:.1f}% match probability. A match is possible, but might require a personalized icebreaker to spark interest."
    else:
        explanation = f'[Fallback] The match probability is relatively low ({match_prob * 100:.1f}%). This is likely due to divergent activity patterns. We recommend exploring more diverse profiles.'
    return explanation