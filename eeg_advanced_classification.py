import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.decomposition import PCA
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

print("Loading EEG dataset...")
file_path = "combined_reduced_epochs (1).csv"

# Load a sample of the data for faster processing
sample_size = 100000  # Increased sample size for better model training
df = pd.read_csv(file_path)
df_sample = df.sample(sample_size, random_state=42)

print(f"Full dataset shape: {df.shape}")
print(f"Sample dataset shape: {df_sample.shape}")

# Identify feature columns (excluding metadata)
numerical_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
metadata_cols = ['epoch', 'channel', 'label', 'trial_id']
feature_cols = [col for col in numerical_cols if col not in metadata_cols]
print(f"Number of features: {len(feature_cols)}")

# Handle potential outliers and infinite values
df_clean = df_sample.replace([np.inf, -np.inf], np.nan)
df_clean = df_clean.dropna(subset=feature_cols)
print(f"Shape after cleaning: {df_clean.shape}")

# Prepare data for modeling
X = df_clean[feature_cols].values
y = df_clean['label'].values

# Standardize the features
print("\nStandardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Feature engineering: Create channel-specific features
print("\nPerforming feature engineering...")
# Check if we have channel information to use
if 'channel' in df_clean.columns:
    # Group by channel and compute statistics
    channel_features = df_clean.groupby(['label', 'channel'])[feature_cols].mean().reset_index()
    print(f"Created channel-specific features: {channel_features.shape}")
    
    # Visualize channel patterns for different classes
    power_bands = ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']
    if all(band in df_clean.columns for band in power_bands):
        # Select a subset of channels for visualization
        channels_to_plot = sorted(df_clean['channel'].unique())[:5]  # First 5 channels
        
        plt.figure(figsize=(15, 10))
        for i, band in enumerate(power_bands):
            plt.subplot(len(power_bands), 1, i+1)
            for label in sorted(df_clean['label'].unique()):
                channel_data = df_clean[df_clean['label'] == label]
                channel_means = channel_data.groupby('channel')[band].mean()
                plt.plot(channels_to_plot, channel_means[channels_to_plot], 
                         marker='o', label=f'Class {label}')
            plt.title(f'{band} by Channel and Class')
            plt.xlabel('Channel')
            plt.ylabel('Power')
            plt.legend()
        plt.tight_layout()
        plt.savefig('channel_power_analysis.png')
        plt.close()

# Feature selection using Random Forest importance
print("\nPerforming feature selection...")
# Random Forest for feature importance
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X_train, y_train)

# Get feature importances
importances = rf_selector.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), [feature_cols[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig('feature_importances.png')
plt.close()

# Select top features
print("\nSelecting top features...")
n_top_features = 10  # Select top 10 features
top_indices = indices[:n_top_features]
top_features = [feature_cols[i] for i in top_indices]
print(f"Top {n_top_features} features: {top_features}")

# Create dataset with only top features
X_train_top = X_train[:, top_indices]
X_test_top = X_test[:, top_indices]

# Apply PCA to the selected features
pca = PCA(n_components=10)  # Reduce to 10 components
X_train_pca = pca.fit_transform(X_train_top)
X_test_pca = pca.transform(X_test_top)
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

# Define models to evaluate
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1),
    "SVM": SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, alpha=0.0001, 
                                    solver='adam', random_state=42)
}

# Evaluate models with cross-validation
print("\nEvaluating models with cross-validation...")
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Original features
print("\nResults with all features:")
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    results[f"{name} (All Features)"] = cv_scores
    print(f"{name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Top features
print("\nResults with top features:")
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_top, y_train, cv=cv, scoring='accuracy')
    results[f"{name} (Top Features)"] = cv_scores
    print(f"{name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# PCA features
print("\nResults with PCA on top features:")
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_pca, y_train, cv=cv, scoring='accuracy')
    results[f"{name} (PCA)"] = cv_scores
    print(f"{name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Train the best model on the full training set
print("\nTraining best model on full training set...")
# Find the best model configuration
best_model_name = max(results, key=lambda k: results[k].mean())
print(f"Best model: {best_model_name} with accuracy: {results[best_model_name].mean():.4f}")

# Train the best model type with the best feature set
if "Top Features" in best_model_name:
    X_train_best = X_train_top
    X_test_best = X_test_top
elif "PCA" in best_model_name:
    X_train_best = X_train_pca
    X_test_best = X_test_pca
else:
    X_train_best = X_train
    X_test_best = X_test

best_model_type = best_model_name.split(" (")[0]
best_model = models[best_model_type]
best_model.fit(X_train_best, y_train)

# Evaluate on test set
y_pred = best_model.predict(X_test_best)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy with best model: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('best_model_confusion_matrix.png')
plt.close()

# Visualize model comparison
plt.figure(figsize=(15, 10))
model_names = []
cv_means = []
cv_stds = []

for name, scores in results.items():
    model_names.append(name)
    cv_means.append(scores.mean())
    cv_stds.append(scores.std())

# Sort by performance
sorted_indices = np.argsort(cv_means)[::-1]
model_names = [model_names[i] for i in sorted_indices]
cv_means = [cv_means[i] for i in sorted_indices]
cv_stds = [cv_stds[i] for i in sorted_indices]

plt.barh(range(len(model_names)), cv_means, xerr=cv_stds, align='center', alpha=0.8)
plt.yticks(range(len(model_names)), model_names)
plt.xlabel('Accuracy')
plt.title('Model Comparison')
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

print("\nAdvanced classification analysis completed. Visualization images saved.")
