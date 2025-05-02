import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
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

# Apply PCA with optimal number of components (20)
print("\nApplying PCA with optimal 20 components...")
n_components = 20  # Optimal number from previous analysis
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA reduced features shape: {X_pca.shape}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_trials, test_size=0.3, random_state=42, stratify=y_trials
)

print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

# Define base classifiers with optimized hyperparameters
print("\nDefining base classifiers...")
base_classifiers = [
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, 
                                 min_samples_leaf=1, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, 
                                     random_state=42)),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)),
    ('mlp', MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, alpha=0.0001, 
                         solver='adam', random_state=42))
]

# Tune hyperparameters for Random Forest
print("\nTuning Random Forest hyperparameters...")
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), 
                      rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)
print(f"Best Random Forest parameters: {rf_grid.best_params_}")
print(f"Best Random Forest CV accuracy: {rf_grid.best_score_:.4f}")

# Update Random Forest in base classifiers
base_classifiers[0] = ('rf', RandomForestClassifier(**rf_grid.best_params_, random_state=42, n_jobs=-1))

# Evaluate individual classifiers
print("\nEvaluating individual classifiers...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
individual_results = {}

for name, clf in base_classifiers:
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
    individual_results[name] = cv_scores
    print(f"{name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Create and evaluate voting ensemble
print("\nCreating voting ensemble...")
voting_clf = VotingClassifier(estimators=base_classifiers, voting='soft')
voting_cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=cv, scoring='accuracy')
print(f"Voting Ensemble CV Accuracy: {voting_cv_scores.mean():.4f} ± {voting_cv_scores.std():.4f}")

# Create and evaluate stacking ensemble
print("\nCreating stacking ensemble...")
stacking_clf = StackingClassifier(
    estimators=base_classifiers,
    final_estimator=LogisticRegression(random_state=42),
    cv=5
)
stacking_cv_scores = cross_val_score(stacking_clf, X_train, y_train, cv=cv, scoring='accuracy')
print(f"Stacking Ensemble CV Accuracy: {stacking_cv_scores.mean():.4f} ± {stacking_cv_scores.std():.4f}")

# Train the best ensemble on the full training set
if stacking_cv_scores.mean() > voting_cv_scores.mean():
    best_ensemble = stacking_clf
    best_ensemble_name = "Stacking Ensemble"
else:
    best_ensemble = voting_clf
    best_ensemble_name = "Voting Ensemble"

print(f"\nTraining {best_ensemble_name} on full training set...")
best_ensemble.fit(X_train, y_train)

# Evaluate on test set
y_pred = best_ensemble.predict(X_test)
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
plt.title(f'Confusion Matrix - {best_ensemble_name}')
plt.tight_layout()
plt.savefig('ensemble_confusion_matrix.png')
plt.close()

# Compare with individual classifiers on test set
print("\nComparing ensemble with individual classifiers on test set...")
test_results = {}

# Evaluate each individual classifier
for name, clf in base_classifiers:
    clf.fit(X_train, y_train)
    y_pred_ind = clf.predict(X_test)
    acc_ind = accuracy_score(y_test, y_pred_ind)
    test_results[name] = acc_ind
    print(f"{name}: {acc_ind:.4f}")

# Add ensemble result
test_results[best_ensemble_name] = accuracy

# Plot comparison
plt.figure(figsize=(10, 6))
models = list(test_results.keys())
accs = list(test_results.values())

# Sort by performance
sorted_indices = np.argsort(accs)[::-1]
models = [models[i] for i in sorted_indices]
accs = [accs[i] for i in sorted_indices]

plt.bar(models, accs, color='skyblue')
plt.axhline(y=0.25, color='r', linestyle='--', label='Random Chance (25%)')
plt.xlabel('Model')
plt.ylabel('Test Accuracy')
plt.title('Model Comparison on Test Set')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# Analyze misclassifications
print("\nAnalyzing misclassifications...")
misclassified_indices = np.where(y_test != y_pred)[0]
print(f"Number of misclassified samples: {len(misclassified_indices)} out of {len(y_test)}")

# Count misclassifications by true class
misclassified_by_class = {}
for i in misclassified_indices:
    true_label = y_test[i]
    pred_label = y_pred[i]
    if true_label not in misclassified_by_class:
        misclassified_by_class[true_label] = []
    misclassified_by_class[true_label].append(pred_label)

# Print misclassification analysis
print("\nMisclassification analysis by true class:")
for true_label, pred_labels in misclassified_by_class.items():
    pred_counts = np.bincount(pred_labels, minlength=4)
    print(f"Class {true_label} misclassified as: Class 0: {pred_counts[0]}, Class 1: {pred_counts[1]}, "
          f"Class 2: {pred_counts[2]}, Class 3: {pred_counts[3]}")

# Visualize feature importance (using Random Forest as a proxy)
rf_clf = base_classifiers[0][1]
if hasattr(rf_clf, 'feature_importances_'):
    # Get feature importances
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot top 10 features
    plt.figure(figsize=(10, 6))
    plt.title('PCA Component Importance')
    plt.bar(range(min(10, n_components)), importances[indices[:10]], align='center')
    plt.xticks(range(min(10, n_components)), [f'PC{i+1}' for i in indices[:10]])
    plt.xlabel('Principal Component')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('pca_component_importance.png')
    plt.close()
    
    # Print top components
    print("\nTop 5 important PCA components:")
    for i in range(min(5, n_components)):
        print(f"PC{indices[i]+1}: {importances[indices[i]]:.4f}")

print("\nEnsemble classification with optimal PCA components completed. Visualization images saved.")
