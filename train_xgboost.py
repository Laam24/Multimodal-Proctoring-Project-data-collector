import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import joblib

# 1. LOAD DATA
print("Loading dataset...")
df = pd.read_csv("FINAL_TRAINING_DATASET.csv")

# Identify features (Order is critical for the live app later!)
exclude = ['Session_ID', 'Timestamp', 'Target_Label']
feature_cols = [c for c in df.columns if c not in exclude]
print(f"Training on {len(feature_cols)} features.")

X = df[feature_cols]
y = df['Target_Label']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, "xgboost_label_encoder.pkl")
# Important: Save feature names to ensure live app uses the same order
joblib.dump(feature_cols, "feature_names.pkl")

# Split Data (Stratified to keep the 72/18/10 ratio in both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 2. COMPUTE SAMPLE WEIGHTS
# This forces the AI to care more about the 10% and 18% classes
weights = compute_sample_weight(class_weight='balanced', y=y_train)

# 3. TRAIN XGBOOST
model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=8,            # Slightly deeper to catch subtle head movements
    learning_rate=0.03,     # Slower learning usually yields better accuracy
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=len(le.classes_),
    tree_method='hist',
    early_stopping_rounds=50, # Stop if accuracy stops improving
    random_state=42
)

print("Starting training with class balancing...")
model.fit(
    X_train, y_train, 
    sample_weight=weights,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# 4. EVALUATE
y_pred = model.predict(X_test)
print("\n--- FINAL PERFORMANCE REPORT ---")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(f"Final Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# 5. SAVE
model.save_model("proctor_xgboost_v1.json")
print("\n✅ XGBoost Model saved as 'proctor_xgboost_v1.json'")