import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

print("Loading EEG dataset...")
file_path = "combined_reduced_epochs (1).csv"
df = pd.read_csv(file_path)

print(f"Dataset shape: {df.shape}")
print(f"Number of unique trials: {df['trial_id'].nunique()}")
print(f"Number of unique labels: {df['label'].nunique()}")

# Identify feature columns (excluding metadata)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
metadata_cols = ['epoch', 'channel', 'label', 'trial_id']
feature_cols = [col for col in numerical_cols if col not in metadata_cols]
print(f"Number of features: {len(feature_cols)}")

# Function to extract features for a single trial
def extract_trial_features(trial_data):
    """Extract features from a trial by averaging across epochs for each channel"""
    # Group by channel and compute mean across epochs
    channel_features = trial_data.groupby('channel')[feature_cols].mean()
    # Flatten the channel features into a single vector
    features = channel_features.values.flatten()
    return features

# Extract features for all trials
print("\nExtracting features for all trials...")
print("Using feature strategy: mean_by_channel")

# Get unique trials
unique_trials = df['trial_id'].unique()

# Create features and labels
X_trials = []
y_trials = []
trial_ids = []

for trial in unique_trials:
    trial_data = df[df['trial_id'] == trial]
    
    # Skip if trial doesn't have data
    if len(trial_data) == 0:
        continue
    
    # Get the label (should be the same for all rows in the trial)
    label = trial_data['label'].iloc[0]
    
    # Extract features
    features = extract_trial_features(trial_data)
    
    X_trials.append(features)
    y_trials.append(label)
    trial_ids.append(trial)

# Convert to numpy arrays
X_trials = np.array(X_trials)
y_trials = np.array(y_trials)
trial_ids = np.array(trial_ids)

print(f"Extracted features shape: {X_trials.shape}")
print(f"Number of trials with features: {len(y_trials)}")
print(f"Class distribution: {np.bincount(y_trials)}")

# Standardize features
print("\nStandardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_trials)

# Apply PCA for dimensionality reduction
print("\nApplying PCA for dimensionality reduction...")
# First, determine optimal number of components
pca_full = PCA()
pca_full.fit(X_scaled)

# Plot explained variance ratio
plt.figure(figsize=(12, 6))
explained_variance = np.cumsum(pca_full.explained_variance_ratio_)
plt.plot(np.arange(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
plt.axhline(y=0.99, color='g', linestyle='--', label='99% Explained Variance')
plt.legend()
plt.grid(True)
plt.savefig('trial_pca_explained_variance.png')
plt.close()

# Find number of components for 95% and 99% variance explained
n_components_95 = np.argmax(explained_variance >= 0.95) + 1
n_components_99 = np.argmax(explained_variance >= 0.99) + 1

print(f"Number of components for 95% variance: {n_components_95}")
print(f"Number of components for 99% variance: {n_components_99}")

# Apply PCA with the number of components for 95% variance
pca_95 = PCA(n_components=n_components_95)
X_pca_95 = pca_95.fit_transform(X_scaled)

print(f"PCA reduced features shape: {X_pca_95.shape}")
print(f"Dimensionality reduction: {X_scaled.shape[1]} -> {X_pca_95.shape[1]} features")
print(f"Compression ratio: {X_scaled.shape[1] / X_pca_95.shape[1]:.2f}x")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_pca_95, y_trials, test_size=0.3, random_state=42, stratify=y_trials
)

print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

# Train and evaluate multiple classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
}

# Cross-validation
print("\nPerforming cross-validation with PCA features...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, clf in classifiers.items():
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
    cv_results[name] = cv_scores
    print(f"{name} CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Train the best classifier on the full training set
best_clf_name = max(cv_results, key=lambda k: cv_results[k].mean())
best_clf = classifiers[best_clf_name]
print(f"\nTraining {best_clf_name} on full training set...")
best_clf.fit(X_train, y_train)

# Evaluate on test set
y_pred = best_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Confusion Matrix - {best_clf_name} with PCA')
plt.tight_layout()
plt.savefig('trial_pca_confusion_matrix.png')
plt.close()

# Try different numbers of PCA components
print("\nTrying different numbers of PCA components...")
component_counts = [10, 20, 30, 50, 100, 200]
component_results = {}

for n_comp in component_counts:
    if n_comp > min(X_scaled.shape):
        continue
        
    print(f"\nUsing {n_comp} PCA components...")
    
    # Apply PCA
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)
    
    # Split data
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        X_pca, y_trials, test_size=0.3, random_state=42, stratify=y_trials
    )
    
    # Train and evaluate
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train_pca, y_train_pca)
    
    y_pred_pca = clf.predict(X_test_pca)
    accuracy_pca = accuracy_score(y_test_pca, y_pred_pca)
    print(f"Accuracy with {n_comp} components: {accuracy_pca:.4f}")
    component_results[n_comp] = accuracy_pca

# Plot accuracy vs number of components
plt.figure(figsize=(10, 6))
components = list(component_results.keys())
accuracies = list(component_results.values())
plt.plot(components, accuracies, marker='o')
plt.xlabel('Number of PCA Components')
plt.ylabel('Classification Accuracy')
plt.title('Accuracy vs Number of PCA Components')
plt.grid(True)
plt.savefig('accuracy_vs_components.png')
plt.close()

# Visualize the first two principal components
print("\nVisualizing first two principal components...")
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# Create DataFrame for visualization
pca_df = pd.DataFrame(data=X_pca_2d, columns=['PC1', 'PC2'])
pca_df['label'] = y_trials

# Plot
plt.figure(figsize=(12, 10))
sns.scatterplot(x='PC1', y='PC2', hue='label', data=pca_df, palette='viridis', alpha=0.8, s=100)
plt.title('PCA: First Two Principal Components')
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance explained)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance explained)')
plt.legend(title='Class')
plt.tight_layout()
plt.savefig('trial_pca_2d_visualization.png')
plt.close()

print("\nTrial-based PCA classification analysis completed. Visualization images saved.")
