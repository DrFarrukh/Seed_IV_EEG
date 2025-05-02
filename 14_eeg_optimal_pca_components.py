import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
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

# Convert to numpy arrays
X_trials = np.array(X_trials)
y_trials = np.array(y_trials)

print(f"Extracted features shape: {X_trials.shape}")
print(f"Number of trials with features: {len(y_trials)}")
print(f"Class distribution: {np.bincount(y_trials)}")

# Standardize features
print("\nStandardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_trials)

# Test different numbers of PCA components
print("\nTesting different numbers of PCA components...")
component_counts = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
results = {}

# Use cross-validation for more reliable results
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for n_comp in component_counts:
    print(f"\nEvaluating with {n_comp} PCA components...")
    
    # Apply PCA
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"Explained variance: {explained_variance:.4f}")
    
    # Cross-validation
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    cv_scores = cross_val_score(clf, X_pca, y_trials, cv=cv, scoring='accuracy')
    
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train-test split for detailed evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y_trials, test_size=0.3, random_state=42, stratify=y_trials
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Store results
    results[n_comp] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': accuracy,
        'explained_variance': explained_variance
    }
    
    # Save confusion matrix for this component count
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {n_comp} PCA Components')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{n_comp}_components.png')
    plt.close()

# Find the best number of components
best_comp = max(results, key=lambda k: results[k]['cv_mean'])
print(f"\nBest number of components: {best_comp}")
print(f"CV Accuracy: {results[best_comp]['cv_mean']:.4f} ± {results[best_comp]['cv_std']:.4f}")
print(f"Test Accuracy: {results[best_comp]['test_accuracy']:.4f}")
print(f"Explained Variance: {results[best_comp]['explained_variance']:.4f}")

# Plot results
plt.figure(figsize=(12, 8))

# Plot CV accuracy
cv_means = [results[n]['cv_mean'] for n in component_counts]
cv_stds = [results[n]['cv_std'] for n in component_counts]
plt.errorbar(component_counts, cv_means, yerr=cv_stds, fmt='-o', label='CV Accuracy')

# Plot test accuracy
test_accs = [results[n]['test_accuracy'] for n in component_counts]
plt.plot(component_counts, test_accs, 'r--o', label='Test Accuracy')

# Plot explained variance
exp_vars = [results[n]['explained_variance'] for n in component_counts]
plt.plot(component_counts, exp_vars, 'g-.o', label='Explained Variance')

plt.xlabel('Number of PCA Components')
plt.ylabel('Accuracy / Explained Variance')
plt.title('Performance vs Number of PCA Components')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('pca_component_analysis.png')
plt.close()

# Detailed analysis of the best model
print("\nDetailed analysis of the best model...")
best_pca = PCA(n_components=best_comp)
X_best_pca = best_pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(
    X_best_pca, y_trials, test_size=0.3, random_state=42, stratify=y_trials
)

best_clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the first two principal components
print("\nVisualizing first two principal components...")
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# Create DataFrame for visualization
pca_df = pd.DataFrame(data=X_pca_2d, columns=['PC1', 'PC2'])
pca_df['label'] = y_trials

# Plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='label', data=pca_df, palette='viridis', alpha=0.8, s=80)
plt.title('PCA: First Two Principal Components')
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance explained)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance explained)')
plt.legend(title='Class')
plt.tight_layout()
plt.savefig('optimal_pca_2d_visualization.png')
plt.close()

print("\nOptimal PCA component analysis completed. Visualization images saved.")
