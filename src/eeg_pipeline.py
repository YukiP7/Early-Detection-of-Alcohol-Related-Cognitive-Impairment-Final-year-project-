# eeg_pipeline.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shap
import joblib

# ---------------------------
# 1. Load dataset
# ---------------------------
df = pd.read_csv("data/EEG_Alcohol_Data/Combined_EEG_dataset.csv")
channels = df.columns[:-1]
labels = df['label'].values

# ---------------------------
# 2. Windowed segmentation
# ---------------------------
window_size = 32
trials = []
trial_labels = []

for label in [0, 1]:
    class_data = df[df['label']==label][channels].values
    n_segments = class_data.shape[0] // window_size
    for i in range(n_segments):
        seg = class_data[i*window_size : (i+1)*window_size, :]
        trials.append(seg)
        trial_labels.append(label)

trials = np.array(trials)
trial_labels = np.array(trial_labels)

# ---------------------------
# 3. FFT feature extraction
# ---------------------------
def band_power(signal, sf=256):
    freqs = np.fft.rfftfreq(len(signal), 1/sf)
    fft_vals = np.fft.rfft(signal)
    psd = np.abs(fft_vals)**2
    bands = {'delta':(0.5,4), 'theta':(4,8), 'alpha':(8,13),
             'beta':(13,30), 'gamma':(30,50)}
    return [np.mean(psd[(freqs>=b[0]) & (freqs<=b[1])]) for b in bands.values()]

X_features = []
for trial in trials:
    trial_feat = []
    for ch_idx in range(trial.shape[1]):
        trial_feat.extend(band_power(trial[:, ch_idx]))
    X_features.append(trial_feat)

X_features = np.array(X_features)
bands = ['delta','theta','alpha','beta','gamma']
feature_names = [f"{ch}_{b}" for ch in channels for b in bands]

# ---------------------------
# 4. Standardize
# ---------------------------
# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# Replace NaNs if any
X_scaled = np.nan_to_num(X_scaled)

joblib.dump(scaler, "scaler.pkl")  # save scaler

# ---------------------------
# 5. Train/test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, trial_labels, test_size=0.5, random_state=42, stratify=trial_labels
)

# ---------------------------
# 6. Train classifiers
# ---------------------------
models = {
    'LogisticRegression': LogisticRegression(max_iter=500, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    
}

results = {}
trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred) * 100,
        'Precision': precision_score(y_test, y_pred , zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred)
    }
    trained_models[name] = model

results_df = pd.DataFrame(results).T
results_df.to_csv("model_results.csv")

import dataframe_image as dfi
dfi.export(results_df, 'model_results.png')

print("Model comparison results saved as model_results.csv")

# 7. Select best model
# ---------------------------
best_model_name = results_df['Accuracy'].idxmax()
best_model = trained_models[best_model_name]
joblib.dump(best_model, "best_model.pkl")
print(f"Best model saved: {best_model_name}")


# ---------------------------
# 7. SHAP analysis (best model)
# ---------------------------

explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test)
shap_mean = np.mean(np.abs(shap_values.values), axis=0)
importance_df = pd.DataFrame({
    'feature': feature_names,
    'shap_value': shap_mean
}).sort_values(by='shap_value', ascending=False)
importance_df.to_csv("top_features.csv", index=False)
print("Top features (SHAP) saved as top_features.csv")

