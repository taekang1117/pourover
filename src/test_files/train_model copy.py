# train_model.py
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATA_FILE = "training_data.csv"
MODEL_FILE = "bean_model.pkl"

def main():
    print(f"Loading data from {DATA_FILE}...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Run collect_data.py first!")
        return

    # Check for empty data or missing columns
    if df.empty:
        print("Dataset is empty.")
        return
        
    required_cols = ['area', 'aspect_ratio', 'circularity', 'solidity', 'perimeter', 'label']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Dataset missing columns. Required: {required_cols}")
        print(f"Found: {df.columns}")
        return

    # Features and Label
    X = df[['area', 'aspect_ratio', 'circularity', 'solidity', 'perimeter']]
    y = df['label']

    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{y.value_counts()}")

    # Split (optional if dataset is small, but good practice)
    # If dataset is very small, we might just train on all, but let's do a split for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train
    print("Training RandomForestClassifier...")
    # Using small n_estimators since we want speed, but 100 is usually fine for this data size
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    print("Evaluating on test set...")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Rock', 'Bean']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save full model (retrain on all data for best performance)
    print("Retraining on full dataset and saving...")
    clf.fit(X, y)
    
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(clf, f)
    
    print(f"Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()
