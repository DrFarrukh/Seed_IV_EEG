import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
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
print(f"Number of unique channels: {df['channel'].nunique()}")
print(f"Number of unique epochs: {df['epoch'].nunique()}")

# Identify feature columns (excluding metadata)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
metadata_cols = ['epoch', 'channel', 'label', 'trial_id']
feature_cols = [col for col in numerical_cols if col not in metadata_cols]
print(f"Number of features: {len(feature_cols)}")

# Function to extract channel-specific features with PCA
def extract_channel_pca_features(df, n_components=5):
    """
    Extract PCA features for each channel separately, then combine
    This preserves channel structure while reducing feature dimensionality
    """
    # Get unique trials and channels
    unique_trials = sorted(df['trial_id'].unique())
    unique_channels = sorted(df['channel'].unique())
    
    n_trials = len(unique_trials)
    n_channels = len(unique_channels)
    
    print(f"Extracting channel-specific PCA features: {n_trials} trials × {n_channels} channels × {n_components} components")
    
    # Initialize array to store PCA features for each channel
    X_channel_pca = np.zeros((n_trials, n_channels * n_components))
    y_trials = np.zeros(n_trials)
    
    # For each channel, fit PCA and extract features
    for j, channel in enumerate(unique_channels):
        # Get data for this channel
        channel_data = df[df['channel'] == channel]
        
        # Create feature matrix for this channel
        X_channel = np.zeros((n_trials, len(feature_cols)))
        
        # Fill feature matrix
        for i, trial in enumerate(unique_trials):
            trial_channel_data = channel_data[channel_data['trial_id'] == trial]
            
            # Get label (should be consistent within trial)
            if len(trial_channel_data) > 0:
                y_trials[i] = trial_channel_data['label'].iloc[0]
            
            # Average features across epochs for this channel and trial
            if len(trial_channel_data) > 0:
                X_channel[i, :] = trial_channel_data[feature_cols].mean().values
        
        # Standardize features for this channel
        scaler = StandardScaler()
        X_channel_scaled = scaler.fit_transform(X_channel)
        
        # Apply PCA to this channel's features
        pca = PCA(n_components=min(n_components, X_channel_scaled.shape[1]))
        X_channel_pca_features = pca.fit_transform(X_channel_scaled)
        
        # Store PCA features for this channel
        start_idx = j * n_components
        end_idx = start_idx + X_channel_pca_features.shape[1]
        X_channel_pca[:, start_idx:end_idx] = X_channel_pca_features
    
    return X_channel_pca, y_trials.astype(int)

# Function to extract epoch-specific features with PCA
def extract_epoch_pca_features(df, n_components=5):
    """
    Extract PCA features for each epoch separately, then combine
    This preserves temporal structure while reducing feature dimensionality
    """
    # Get unique trials and epochs
    unique_trials = sorted(df['trial_id'].unique())
    unique_epochs = sorted(df['epoch'].unique())
    
    n_trials = len(unique_trials)
    n_epochs = len(unique_epochs)
    
    print(f"Extracting epoch-specific PCA features: {n_trials} trials × {n_epochs} epochs × {n_components} components")
    
    # Initialize array to store PCA features for each epoch
    X_epoch_pca = np.zeros((n_trials, n_epochs * n_components))
    y_trials = np.zeros(n_trials)
    
    # For each epoch, fit PCA and extract features
    for j, epoch in enumerate(unique_epochs):
        # Get data for this epoch
        epoch_data = df[df['epoch'] == epoch]
        
        # Create feature matrix for this epoch
        X_epoch = np.zeros((n_trials, len(feature_cols)))
        
        # Fill feature matrix
        for i, trial in enumerate(unique_trials):
            trial_epoch_data = epoch_data[epoch_data['trial_id'] == trial]
            
            # Get label (should be consistent within trial)
            if len(trial_epoch_data) > 0:
                y_trials[i] = trial_epoch_data['label'].iloc[0]
            
            # Average features across channels for this epoch and trial
            if len(trial_epoch_data) > 0:
                X_epoch[i, :] = trial_epoch_data[feature_cols].mean().values
        
        # Standardize features for this epoch
        scaler = StandardScaler()
        X_epoch_scaled = scaler.fit_transform(X_epoch)
        
        # Apply PCA to this epoch's features
        pca = PCA(n_components=min(n_components, X_epoch_scaled.shape[1]))
        X_epoch_pca_features = pca.fit_transform(X_epoch_scaled)
        
        # Store PCA features for this epoch
        start_idx = j * n_components
        end_idx = start_idx + X_epoch_pca_features.shape[1]
        X_epoch_pca[:, start_idx:end_idx] = X_epoch_pca_features
    
    return X_epoch_pca, y_trials.astype(int)

# Function to extract global PCA features (our previous best approach)
def extract_global_pca_features(df, n_components=20):
    """
    Extract global PCA features across all channels and epochs
    This is our previous best approach
    """
    # Get unique trials
    unique_trials = sorted(df['trial_id'].unique())
    
    n_trials = len(unique_trials)
    
    print(f"Extracting global PCA features: {n_trials} trials × {n_components} components")
    
    # Initialize arrays
    X_trials = []
    y_trials = []
    
    # Extract features for all trials
    for trial in unique_trials:
        trial_data = df[df['trial_id'] == trial]
        
        # Skip if trial doesn't have data
        if len(trial_data) == 0:
            continue
        
        # Get the label (should be the same for all rows in the trial)
        label = trial_data['label'].iloc[0]
        
        # Extract features by averaging across epochs for each channel
        channel_features = trial_data.groupby('channel')[feature_cols].mean()
        # Flatten the channel features into a single vector
        features = channel_features.values.flatten()
        
        X_trials.append(features)
        y_trials.append(label)
    
    # Convert to numpy arrays
    X_trials = np.array(X_trials)
    y_trials = np.array(y_trials)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_trials)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, y_trials

# Extract features using all three approaches
print("\nExtracting features using hybrid approach...")

# 1. Channel-specific PCA features
X_channel_pca, y_channel = extract_channel_pca_features(df, n_components=5)

# 2. Epoch-specific PCA features
X_epoch_pca, y_epoch = extract_epoch_pca_features(df, n_components=3)

# 3. Global PCA features
X_global_pca, y_global = extract_global_pca_features(df, n_components=20)

# Verify that labels are consistent
assert np.array_equal(y_channel, y_epoch)
assert np.array_equal(y_channel, y_global)

# Combine all features
print("\nCombining features from all approaches...")
X_combined = np.hstack((X_channel_pca, X_epoch_pca, X_global_pca))
y_combined = y_channel

print(f"Combined feature shape: {X_combined.shape}")
print(f"Number of trials: {len(y_combined)}")
print(f"Class distribution: {np.bincount(y_combined)}")

# Split data into training and testing sets
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_combined, test_size=0.3, random_state=42, stratify=y_combined
)

print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

# Create ensemble of classifiers
print("\nCreating ensemble of classifiers...")
classifiers = [
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42))
]

# Create voting classifier
voting_clf = VotingClassifier(estimators=classifiers, voting='soft')

# Cross-validation
print("\nPerforming cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate individual classifiers
print("Individual classifier performance:")
for name, clf in classifiers:
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"{name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Evaluate ensemble
cv_scores_ensemble = cross_val_score(voting_clf, X_train, y_train, cv=cv, scoring='accuracy')
print(f"Ensemble: {cv_scores_ensemble.mean():.4f} ± {cv_scores_ensemble.std():.4f}")

# Train the ensemble on the full training set
print("\nTraining ensemble on full training set...")
voting_clf.fit(X_train, y_train)

# Evaluate on test set
y_pred = voting_clf.predict(X_test)
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
plt.title('Confusion Matrix - Hybrid Model')
plt.tight_layout()
plt.savefig('hybrid_confusion_matrix.png')
plt.close()

# Compare with previous approaches
print("\nComparing with previous approaches:")
print(f"PCA + Ensemble: 67.59%")
print(f"EEGNet-inspired: 49.07%")
print(f"Hybrid Model: {accuracy:.4f}")

# Analyze feature importance (using Random Forest)
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)

# Get feature importances
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]

# Determine which feature set each feature belongs to
channel_pca_size = X_channel_pca.shape[1]
epoch_pca_size = X_epoch_pca.shape[1]
global_pca_size = X_global_pca.shape[1]

feature_types = []
for i in range(X_combined.shape[1]):
    if i < channel_pca_size:
        channel_idx = i // 5
        comp_idx = i % 5
        feature_types.append(f"Channel {channel_idx} PC{comp_idx+1}")
    elif i < channel_pca_size + epoch_pca_size:
        idx = i - channel_pca_size
        epoch_idx = idx // 3
        comp_idx = idx % 3
        feature_types.append(f"Epoch {epoch_idx} PC{comp_idx+1}")
    else:
        idx = i - (channel_pca_size + epoch_pca_size)
        feature_types.append(f"Global PC{idx+1}")

# Plot top 20 features
plt.figure(figsize=(12, 8))
plt.title('Hybrid Model Feature Importances')
plt.bar(range(20), importances[indices[:20]], align='center')
plt.xticks(range(20), [feature_types[i] for i in indices[:20]], rotation=90)
plt.tight_layout()
plt.savefig('hybrid_feature_importances.png')
plt.close()

# Print top features
print("\nTop 10 important features:")
for i in range(10):
    print(f"{feature_types[indices[i]]}: {importances[indices[i]]:.4f}")

# Analyze which approach contributes most to classification
channel_importance = sum(importances[:channel_pca_size])
epoch_importance = sum(importances[channel_pca_size:channel_pca_size+epoch_pca_size])
global_importance = sum(importances[channel_pca_size+epoch_pca_size:])

print("\nRelative importance of each approach:")
print(f"Channel-specific PCA: {channel_importance:.4f} ({channel_importance/sum(importances):.2%})")
print(f"Epoch-specific PCA: {epoch_importance:.4f} ({epoch_importance/sum(importances):.2%})")
print(f"Global PCA: {global_importance:.4f} ({global_importance/sum(importances):.2%})")

print("\nHybrid model analysis completed. Visualization images saved.")
