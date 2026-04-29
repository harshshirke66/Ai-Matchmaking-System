import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from utils import load_model, generate_ai_explanation, load_data
load_dotenv()
st.set_page_config(page_title='Dating Match Predictor', layout='centered')
st.markdown('\n    <style>\n    @import url(\'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;800&display=swap\');\n\n    html, body, [class*="css"] {\n        font-family: \'Inter\', sans-serif;\n    }\n\n    /* App Background - Pure Black */\n    .stApp {\n        background: #000000 !important;\n    }\n\n    /* Main Container adjust */\n    .block-container {\n        padding-top: 3rem !important;\n        max-width: 800px;\n    }\n\n    /* Form Container */\n    [data-testid="stForm"] {\n        background-color: #0a0a0a !important;\n        border: 1px solid #222222 !important;\n        border-radius: 12px !important;\n        padding: 2rem !important;\n    }\n\n    /* Headers */\n    h1 {\n        font-weight: 800 !important;\n        color: #ffffff !important;\n        text-align: center;\n        letter-spacing: -0.5px;\n        background: none !important;\n        -webkit-text-fill-color: #ffffff !important;\n    }\n\n    /* Inputs & Selects */\n    div[data-baseweb="select"] > div, input, div[data-baseweb="input"] {\n        background-color: #111111 !important;\n        border-radius: 6px !important;\n        border: 1px solid #333333 !important;\n        color: #ffffff !important;\n    }\n    \n    div[data-baseweb="select"]:hover > div, input:hover {\n        border: 1px solid #666666 !important;\n    }\n\n    /* Premium Submit Button */\n    button[kind="formSubmit"] {\n        width: 100%;\n        border-radius: 6px !important;\n        height: 3em;\n        background-color: #ffffff !important;\n        color: #000000 !important;\n        font-weight: 700 !important;\n        font-size: 1.05rem !important;\n        border: none !important;\n        transition: all 0.2s ease !important;\n        margin-top: 15px;\n    }\n\n    button[kind="formSubmit"]:hover {\n        background-color: #e0e0e0 !important;\n        transform: translateY(-1px);\n    }\n\n    button[kind="formSubmit"]:active {\n        transform: translateY(0px);\n    }\n\n    /* Match Output Boxes */\n    .match-box {\n        padding: 25px;\n        border-radius: 12px;\n        text-align: center;\n        margin-top: 30px;\n        background: #0a0a0a;\n        border: 1px solid #333333;\n        color: white;\n    }\n    \n    .match-box h2 {\n        background: none !important;\n        -webkit-text-fill-color: white !important;\n    }\n\n    /* AI Insight Box */\n    div[data-testid="stInfo"] {\n        background: #111111 !important;\n        border: 1px solid #333333 !important;\n        border-radius: 12px !important;\n        color: #cccccc !important;\n    }\n    \n    /* Text styling */\n    p {\n        color: #a0a0a0;\n    }\n    label {\n        color: #eeeeee !important;\n        font-weight: 500 !important;\n    }\n    </style>\n    ', unsafe_allow_html=True)
st.title('Dating Match Predictor')
st.markdown("<p style='text-align: center; color: #888; margin-bottom: 2rem;'>Enter profile details to predict the likelihood of a match.</p>", unsafe_allow_html=True)
groq_api_key = os.getenv('GROQ_API_KEY', '')

@st.cache_resource
def get_resources():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'match_model.pkl')
    data_path = os.path.join(base_dir, 'dating_dataset.csv')
    model = load_model(model_path)
    df = load_data(data_path)
    return (model, df)
try:
    model, original_df = get_resources()
except Exception as e:
    st.error('Model or data not found. Please run preprocessing and training first.')
    st.stop()
with st.form('profile_form'):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox('Gender', original_df['gender'].unique())
        orientation = st.selectbox('Sexual Orientation', original_df['sexual_orientation'].unique())
        location = st.selectbox('Location Type', original_df['location_type'].unique())
        income = st.selectbox('Income Bracket', original_df['income_bracket'].unique())
        education = st.selectbox('Education Level', original_df['education_level'].unique())
    with col2:
        usage_time = st.slider('Daily App Usage (min)', 0, 500, 60)
        swipe_ratio = st.slider('Swipe Right Ratio', 0.0, 1.0, 0.5)
        pics_count = st.number_input('Profile Photos', 1, 10, 3)
        bio_len = st.number_input('Bio Length (chars)', 0, 500, 150)
        time_of_day = st.selectbox('Preferred Swipe Time', original_df['swipe_time_of_day'].unique())
    submit = st.form_submit_button('Predict Match')
if submit:
    input_data = {'gender': gender, 'sexual_orientation': orientation, 'location_type': location, 'income_bracket': income, 'education_level': education, 'app_usage_time_min': usage_time, 'app_usage_time_label': 'Moderate', 'swipe_right_ratio': swipe_ratio, 'swipe_right_label': 'Balanced', 'likes_received': 50, 'mutual_matches': 5, 'profile_pics_count': pics_count, 'bio_length': bio_len, 'message_sent_count': 10, 'emoji_usage_rate': 0.2, 'last_active_hour': 20, 'swipe_time_of_day': time_of_day, 'num_interests': 5}
    input_df = pd.DataFrame([input_data])
    processed_input = input_df.copy()
    cat_cols = ['gender', 'sexual_orientation', 'location_type', 'income_bracket', 'education_level', 'app_usage_time_label', 'swipe_right_label', 'swipe_time_of_day']
    for col in cat_cols:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(original_df[col].astype(str))
        processed_input[col] = le.transform(processed_input[col].astype(str))
    training_cols = ['gender', 'sexual_orientation', 'location_type', 'income_bracket', 'education_level', 'app_usage_time_min', 'app_usage_time_label', 'swipe_right_ratio', 'swipe_right_label', 'likes_received', 'mutual_matches', 'profile_pics_count', 'bio_length', 'message_sent_count', 'emoji_usage_rate', 'last_active_hour', 'swipe_time_of_day', 'num_interests']
    processed_input = processed_input[training_cols]
    prob = model.predict_proba(processed_input)[0][1]
    is_match = model.predict(processed_input)[0]
    st.divider()
    if is_match:
        st.balloons()
        st.success(f"### It's a Potential Match! ({prob * 100:.1f}%)")
    else:
        st.warning(f'### Low Match Probability ({prob * 100:.1f}%)')
    st.subheader('🤖 AI Insight')
    explanation = generate_ai_explanation(input_data, prob, api_key=groq_api_key)
    st.info(explanation)