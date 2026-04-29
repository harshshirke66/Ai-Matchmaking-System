import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from utils import save_model
import os

def train_and_evaluate(data_path, model_output_path):
    print('--- Starting Model Training ---')
    if not os.path.exists(data_path):
        print(f'Error: {data_path} not found. Run preprocessing first.')
        return
    df = pd.read_csv(data_path)
    X = df.drop(columns=['is_match'])
    y = df['is_match']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f'Training set size: {X_train.shape[0]}')
    print(f'Testing set size: {X_test.shape[0]}')
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print('\n--- Model Performance ---')
    print(f'Accuracy:  {accuracy:.4f}')
    print(f'Precision: {precision:.4f} (Ability to not label a non-match as a match)')
    print(f'Recall:    {recall:.4f} (Ability to find all actual matches)')
    print(f'F1 Score:  {f1:.4f} (Harmonic mean of precision and recall)')
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
    print('\n--- Metric Interpretation ---')
    print('In a dating app, Precision is usually more important than Recall.')
    print('If we predict a match (Precision), we want to be highly confident it will happen.')
    print("Showing users highly curated, high-probability matches prevents 'swipe fatigue'.")
    print("A lower Recall is acceptable because we don't need to show a user *every single* person they might match with;")
    print('we just need the ones we *do* show to be quality connections.')
    save_model(model, model_output_path)
    print(f'Model saved to {model_output_path}')
    print('--- Training Complete ---')
    print('\n--- Task 3 Part A: Testing Two Example Profiles ---')
    positive_idx = y_test[y_test == 1].index[0]
    negative_idx = y_test[y_test == 0].index[0]
    for idx, label in zip([positive_idx, negative_idx], ['Profile A', 'Profile B']):
        profile = X.loc[[idx]]
        prob = model.predict_proba(profile)[0][1]
        pred = model.predict(profile)[0]
        actual = y.loc[idx]
        print(f'\n{label} details:')
        print(profile[['app_usage_time_min', 'swipe_right_ratio', 'likes_received', 'mutual_matches']].to_string(index=False))
        print(f'Predicted Match Probability: {prob:.2f}')
        print(f"Predicted Outcome: {('Match' if pred == 1 else 'No Match')} (Actual: {('Match' if actual == 1 else 'No Match')})")
        if pred == 1:
            print('Explanation: This makes intuitive sense because high mutual_matches and likes_received are strong indicators of a highly attractive and engaged profile.')
        else:
            print('Explanation: This makes intuitive sense. Lower engagement or extreme swiping ratios often correlate with spam/bot behavior or low desirability, leading to no match.')
if __name__ == '__main__':
    train_and_evaluate('cleaned_dating_data.csv', 'match_model.pkl')