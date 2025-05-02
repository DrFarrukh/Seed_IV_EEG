import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
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
def extract_trial_features(trial_data, feature_strategy='mean_by_channel'):
    """
    Extract features from a trial using different strategies:
    - 'mean_by_channel': Average features across epochs for each channel
    - 'mean_all': Average all features across all channels and epochs
    - 'channel_epoch_stats': Statistical features across channels and epochs
    """
    if feature_strategy == 'mean_by_channel':
        # Group by channel and compute mean across epochs
        channel_features = trial_data.groupby('channel')[feature_cols].mean()
        # Flatten the channel features into a single vector
        features = channel_features.values.flatten()
        
    elif feature_strategy == 'mean_all':
        # Simple average of all features across the trial
        features = trial_data[feature_cols].mean().values
        
    elif feature_strategy == 'channel_epoch_stats':
        # Compute statistics for each feature across the trial
        features_mean = trial_data[feature_cols].mean().values
        features_std = trial_data[feature_cols].std().values
        features_min = trial_data[feature_cols].min().values
        features_max = trial_data[feature_cols].max().values
        
        # Combine all statistics
        features = np.concatenate([features_mean, features_std, features_min, features_max])
        
    return features

# Extract features for all trials
print("\nExtracting features for all trials...")
feature_strategy = 'mean_by_channel'  # This gave us ~69% accuracy in the previous analysis
print(f"Using feature strategy: {feature_strategy}")

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
    features = extract_trial_features(trial_data, feature_strategy)
    
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

# Split data into training and testing sets
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X_scaled, y_trials, trial_ids, test_size=0.3, random_state=42, stratify=y_trials
)

print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

# Train and evaluate multiple classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
}

# Cross-validation
print("\nPerforming cross-validation...")
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
plt.title(f'Confusion Matrix - {best_clf_name}')
plt.tight_layout()
plt.savefig('trial_classification_confusion_matrix.png')
plt.close()

# Analyze misclassifications
print("\nAnalyzing misclassifications...")
misclassified_indices = np.where(y_test != y_pred)[0]
misclassified_trials = ids_test[misclassified_indices]
print(f"Number of misclassified trials: {len(misclassified_indices)} out of {len(y_test)}")

# Try different feature extraction strategies
print("\nTrying different feature extraction strategies...")
strategies = ['mean_all', 'channel_epoch_stats']

for strategy in strategies:
    print(f"\nUsing strategy: {strategy}")
    
    # Extract features with this strategy
    X_strategy = []
    
    for trial in unique_trials:
        trial_data = df[df['trial_id'] == trial]
        
        if len(trial_data) == 0:
            continue
        
        features = extract_trial_features(trial_data, strategy)
        X_strategy.append(features)
    
    X_strategy = np.array(X_strategy)
    print(f"Feature shape: {X_strategy.shape}")
    
    # Standardize
    X_strategy_scaled = scaler.fit_transform(X_strategy)
    
    # Split data
    X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(
        X_strategy_scaled, y_trials, test_size=0.3, random_state=42, stratify=y_trials
    )
    
    # Train and evaluate
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train_strat, y_train_strat)
    
    y_pred_strat = clf.predict(X_test_strat)
    accuracy_strat = accuracy_score(y_test_strat, y_pred_strat)
    print(f"Accuracy with {strategy}: {accuracy_strat:.4f}")

# Feature importance analysis
print("\nAnalyzing feature importance...")
if hasattr(best_clf, 'feature_importances_'):
    # Get feature importances
    importances = best_clf.feature_importances_
    
    # If using mean_by_channel strategy, reshape to get channel-feature importance
    if feature_strategy == 'mean_by_channel':
        n_channels = 62
        n_features = len(feature_cols)
        
        # Reshape importances to (n_channels, n_features)
        try:
            importances_reshaped = importances.reshape(n_channels, n_features)
            
            # Sum importance across channels for each feature
            feature_importance = importances_reshaped.sum(axis=0)
            
            # Plot top 20 features
            plt.figure(figsize=(12, 8))
            indices = np.argsort(feature_importance)[::-1]
            plt.title('Feature Importances (Summed Across Channels)')
            plt.bar(range(20), feature_importance[indices[:20]], align='center')
            plt.xticks(range(20), [feature_cols[i] for i in indices[:20]], rotation=90)
            plt.tight_layout()
            plt.savefig('trial_feature_importances.png')
            plt.close()
            
            # Print top features
            print("\nTop 10 important features:")
            for i in range(10):
                print(f"{feature_cols[indices[i]]}: {feature_importance[indices[i]]:.4f}")
        except:
            print("Could not reshape feature importances - dimensions may not match")
    else:
        # Plot regular feature importances
        plt.figure(figsize=(12, 8))
        indices = np.argsort(importances)[::-1]
        plt.title('Feature Importances')
        plt.bar(range(20), importances[indices[:20]], align='center')
        plt.xticks(range(20), [f"Feature {i}" for i in indices[:20]], rotation=90)
        plt.tight_layout()
        plt.savefig('trial_feature_importances.png')
        plt.close()

print("\nTrial-based classification analysis completed. Visualization images saved.")
