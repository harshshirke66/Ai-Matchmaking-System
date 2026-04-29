<div align="center">
  <h1>💖 AI Matchmaking System</h1>
  <p><strong>Intelligent Dating Compatibility Predictor powered by Machine Learning & Generative AI</strong></p>
  
  [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-matchmaking-system.streamlit.app/)
  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-Enabled-orange.svg)](https://scikit-learn.org/)
  [![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203-red.svg)](https://groq.com/)
</div>

<br>

## 🌟 Overview

Welcome to the **AI Matchmaking System**, a comprehensive end-to-end machine learning pipeline and interactive web application designed to evaluate dating profile compatibility. 

This project goes beyond raw algorithmic scoring. By combining a **Random Forest Classifier** with **Groq's Llama-3 API**, it not only predicts the statistical likelihood of a match but provides human-readable, psychologically-aware insights into *why* a connection might succeed.

**👉 [Try the Live Application Here](https://ai-matchmaking-system.streamlit.app/) 👈**

---

## 🏗️ Architecture & Features

This repository represents the complete lifecycle of an applied AI product, built entirely from scratch:

### 1. Data Engineering (`data_preprocessing.py`)
- **Robust Imputation:** Handled missing data via median strategy for numeric outliers and mode strategy for categorical features to prevent data leakage.
- **Target Engineering:** Dynamically constructed a binary `is_match` label representing user synergy.
- **Label Encoding:** Standardized all string-based categories into numeric tensors for model readiness.

### 2. Machine Learning (`model_training.py`)
- **Algorithm:** Deployed a **Random Forest Classifier** due to its resilience against non-linear datasets and minimal need for feature scaling.
- **Evaluation:** Prioritized **Precision** over Recall to mitigate "swipe fatigue", ensuring that predicted matches are of the highest quality rather than just high volume.
- **Persistence:** Serialized the optimized model weights into `match_model.pkl` for instantaneous, headless predictions.

### 3. Generative AI Layer (`utils.py`)
- **LLM Integration:** Integrated **Groq (Llama-3)** to interpret the Random Forest's numerical outputs.
- **Graceful Degradation:** Built a resilient fallback mechanism that seamlessly defaults to rule-based insights if the LLM API experiences rate limits or network issues.

### 4. Interactive Frontend (`app.py`)
- **Premium UI:** Designed a high-contrast, minimalist UI in **Streamlit** featuring custom CSS, Google Fonts, and responsive layouts.
- **Cloud Security:** Handled sensitive API keys securely via `st.secrets` natively on Streamlit Community Cloud.

---

## 🚀 How to Run Locally

If you wish to run the predictive engine and UI on your local machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/harshshirke66/Ai-Matchmaking-System.git
   cd Ai-Matchmaking-System
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY="your_groq_api_key_here"
   ```

4. **Launch the App:**
   ```bash
   streamlit run app.py
   ```

---

## 📄 Research & Future Improvements

To read my comprehensive analysis on the limitations of tabular dating models, the necessity of dyadic feature engineering, and plans for a Two-Tower Deep Neural Network, please view the [Task 4: Research and Improvement Plan](./Task%204_%20Research%20and%20Improvement%20Plan.pdf).

<br>

<div align="center">
  <i>Designed and developed for the AI / Backend Engineer Assignment.</i>
</div>
