# eeg_report.py
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ---------------------------
# 1. Load pipeline outputs
# ---------------------------
results_df = pd.read_csv("model_results.csv", index_col=0)
synthetic_df = pd.read_csv("synthetic_at_risk_eeg.csv")
explainer, shap_values = joblib.load("shap_data.pkl")

# ---------------------------
# 2. Print summary report
# ---------------------------
print("===== EEG Classification Report =====")
print("\n--- Model Comparison ---")
print(results_df)

print("\n--- Synthetic At-Risk EEG ---")
print(synthetic_df)

# ---------------------------
# 3. SHAP plots
# ---------------------------
feature_names = synthetic_df.columns[:-2]  # exclude label and prob
X_test = explainer.data  # test data used in SHAP

shap.summary_plot(shap_values, features=X_test, feature_names=feature_names, plot_type="bar")
shap.summary_plot(shap_values, features=X_test, feature_names=feature_names)

# ---------------------------
# 4. Optional: PSD plots of synthetic EEG
# ---------------------------
import numpy as np

sf = 256
for i, feature in enumerate(feature_names[:5]):  # first 5 features
    psd = np.abs(np.fft.rfft(synthetic_df[feature].values))**2
    freqs = np.fft.rfftfreq(len(psd), 1/sf)
    plt.plot(freqs, psd, label=feature)

plt.title("PSD - Synthetic At-Risk EEG Features")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.legend()
plt.show()
