# Dating App Matching System

A complete end-to-end solution for predicting dating app matches using Machine Learning and Streamlit.

## 🚀 Features
- **Data Preprocessing**: Handles missing values, encodes categories, and engineers features.
- **Model Training**: Trains a Random Forest classifier to predict match outcomes.
- **Interactive App**: A Streamlit dashboard for real-time predictions.
- **AI Explanations**: Natural language reasoning for every prediction.

## 🛠️ Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Preprocess the data:
   ```bash
   python data_preprocessing.py
   ```

3. Train the model:
   ```bash
   python model_training.py
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure
- `data_preprocessing.py`: Cleans and prepares the dataset.
- `model_training.py`: Trains and evaluates the ML model.
- `app.py`: Streamlit frontend.
- `utils.py`: Shared utility functions and AI logic.
- `research_notes.md`: Detailed analysis of model limitations and future improvements.
- `requirements.txt`: Project dependencies.

## 🤖 AI Integration
The app includes an AI explanation layer that interprets the model's confidence scores and feature inputs to provide human-readable insights into why a match was (or wasn't) predicted.
