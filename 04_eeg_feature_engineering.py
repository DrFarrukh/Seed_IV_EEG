import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy import signal
from scipy.stats import skew, kurtosis
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

# Define EEG frequency bands
eeg_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100)
}

# Check if power band features exist in the dataset
power_bands = ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']
has_power_bands = all(band in df.columns for band in power_bands)
print(f"Dataset has power band features: {has_power_bands}")

# Function to extract domain-specific EEG features for a trial
def extract_eeg_domain_features(trial_data):
    """
    Extract domain-specific EEG features from a trial
    
    Features include:
    1. Channel connectivity features (correlation between channels)
    2. Band power ratios (e.g., theta/beta ratio)
    3. Asymmetry indices (differences between hemispheres)
    4. Spatial complexity measures
    5. Statistical features across channels
    """
    features = {}
    
    # 1. Extract basic statistical features across channels for each feature
    for feature in feature_cols:
        if feature in trial_data.columns:
            feature_data = trial_data[feature].values
            features[f"mean_{feature}"] = np.mean(feature_data)
            features[f"std_{feature}"] = np.std(feature_data)
            features[f"max_{feature}"] = np.max(feature_data)
            features[f"min_{feature}"] = np.min(feature_data)
    
    # 2. Power band ratios (if available)
    if has_power_bands:
        # Group by channel to compute averages
        channel_powers = trial_data.groupby('channel')[power_bands].mean()
        
        # Calculate band power ratios
        features['theta_beta_ratio'] = np.mean(channel_powers['theta_power'] / (channel_powers['beta_power'] + 1e-10))
        features['alpha_beta_ratio'] = np.mean(channel_powers['alpha_power'] / (channel_powers['beta_power'] + 1e-10))
        features['delta_theta_ratio'] = np.mean(channel_powers['delta_power'] / (channel_powers['theta_power'] + 1e-10))
        features['gamma_beta_ratio'] = np.mean(channel_powers['gamma_power'] / (channel_powers['beta_power'] + 1e-10))
        
        # Calculate relative band powers
        total_power = channel_powers[power_bands].sum(axis=1)
        for band in power_bands:
            features[f'mean_rel_{band}'] = np.mean(channel_powers[band] / (total_power + 1e-10))
            features[f'std_rel_{band}'] = np.std(channel_powers[band] / (total_power + 1e-10))
    
    # 3. Asymmetry indices (left vs right hemisphere)
    # Identify left and right hemisphere channels
    # Assuming channels are numbered with even on right, odd on left (common convention)
    # This is a simplification - actual channel mapping would depend on the EEG system
    left_channels = [ch for ch in trial_data['channel'].unique() if ch % 2 == 1]
    right_channels = [ch for ch in trial_data['channel'].unique() if ch % 2 == 0]
    
    if len(left_channels) > 0 and len(right_channels) > 0 and has_power_bands:
        left_data = trial_data[trial_data['channel'].isin(left_channels)]
        right_data = trial_data[trial_data['channel'].isin(right_channels)]
        
        # Calculate asymmetry indices for each power band
        for band in power_bands:
            left_power = left_data.groupby('channel')[band].mean().mean()
            right_power = right_data.groupby('channel')[band].mean().mean()
            features[f'asymmetry_{band}'] = (right_power - left_power) / (right_power + left_power + 1e-10)
    
    # 4. Channel connectivity (correlation between channels)
    # Focus on alpha and beta bands which are important for cognitive tasks
    if 'alpha_power' in trial_data.columns and 'beta_power' in trial_data.columns:
        # Create a pivot table: channels x features
        try:
            channel_alpha = trial_data.pivot(index='epoch', columns='channel', values='alpha_power')
            channel_beta = trial_data.pivot(index='epoch', columns='channel', values='beta_power')
            
            # Calculate correlation matrices
            if channel_alpha.shape[0] > 1:  # Need at least 2 epochs for correlation
                alpha_corr = channel_alpha.corr().values
                beta_corr = channel_beta.corr().values
                
                # Extract upper triangle of correlation matrix (excluding diagonal)
                mask = np.triu_indices(alpha_corr.shape[0], k=1)
                alpha_corr_upper = alpha_corr[mask]
                beta_corr_upper = beta_corr[mask]
                
                # Calculate statistics of correlations
                features['alpha_conn_mean'] = np.mean(alpha_corr_upper)
                features['alpha_conn_std'] = np.std(alpha_corr_upper)
                features['beta_conn_mean'] = np.mean(beta_corr_upper)
                features['beta_conn_std'] = np.std(beta_corr_upper)
        except:
            # If pivot fails (e.g., missing values), skip connectivity features
            pass
    
    # 5. Temporal dynamics (if multiple epochs)
    epoch_counts = trial_data['epoch'].value_counts()
    if len(epoch_counts) > 1:
        # Calculate temporal changes in power bands
        if has_power_bands:
            for band in power_bands:
                epoch_means = trial_data.groupby('epoch')[band].mean()
                if len(epoch_means) > 1:
                    # Calculate slope of band power over time
                    x = np.array(epoch_means.index)
                    y = epoch_means.values
                    if len(x) > 1 and len(y) > 1:
                        slope, _ = np.polyfit(x, y, 1)
                        features[f'{band}_temporal_slope'] = slope
    
    # 6. Complexity measures across channels
    for feature in ['sample_entropy', 'perm_entropy', 'spectral_entropy']:
        if feature in trial_data.columns:
            features[f'mean_{feature}'] = trial_data[feature].mean()
            features[f'std_{feature}'] = trial_data[feature].std()
    
    return features

# Extract domain-specific features for all trials
print("\nExtracting domain-specific EEG features for all trials...")

# Get unique trials
unique_trials = df['trial_id'].unique()

# Create features and labels
X_domain = []
y_domain = []
trial_ids = []
feature_names = []

# Process first trial to get feature names
first_trial = unique_trials[0]
first_trial_data = df[df['trial_id'] == first_trial]
first_features = extract_eeg_domain_features(first_trial_data)
feature_names = list(first_features.keys())

print(f"Number of domain-specific features: {len(feature_names)}")

# Extract features for all trials
for trial in unique_trials:
    trial_data = df[df['trial_id'] == trial]
    
    # Skip if trial doesn't have data
    if len(trial_data) == 0:
        continue
    
    # Get the label (should be the same for all rows in the trial)
    label = trial_data['label'].iloc[0]
    
    # Extract features
    features = extract_eeg_domain_features(trial_data)
    
    # Convert to vector in the same order as feature_names
    feature_vector = [features.get(name, 0) for name in feature_names]
    
    X_domain.append(feature_vector)
    y_domain.append(label)
    trial_ids.append(trial)

# Convert to numpy arrays
X_domain = np.array(X_domain)
y_domain = np.array(y_domain)
trial_ids = np.array(trial_ids)

print(f"Domain-specific features shape: {X_domain.shape}")
print(f"Number of trials with features: {len(y_domain)}")
print(f"Class distribution: {np.bincount(y_domain)}")

# Check for NaN or infinite values
nan_count = np.isnan(X_domain).sum()
inf_count = np.isinf(X_domain).sum()
print(f"NaN values: {nan_count}, Infinite values: {inf_count}")

# Replace NaN and infinite values
if nan_count > 0 or inf_count > 0:
    X_domain = np.nan_to_num(X_domain, nan=0, posinf=0, neginf=0)
    print("Replaced NaN and infinite values with 0")

# Standardize features
print("\nStandardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_domain)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_domain, test_size=0.3, random_state=42, stratify=y_domain
)

print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

# Create ensemble of classifiers
print("\nCreating ensemble of classifiers...")
classifiers = [
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, alpha=0.0001, 
                          solver='adam', random_state=42))
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
plt.title('Confusion Matrix - Ensemble Classifier')
plt.tight_layout()
plt.savefig('ensemble_confusion_matrix.png')
plt.close()

# Feature importance analysis (from Random Forest)
rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_classifier.fit(X_train, y_train)

# Get feature importances
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot top 20 features
plt.figure(figsize=(12, 8))
plt.title('Domain-Specific Feature Importances')
plt.bar(range(20), importances[indices[:20]], align='center')
plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=90)
plt.tight_layout()
plt.savefig('domain_feature_importances.png')
plt.close()

# Print top features
print("\nTop 10 important domain-specific features:")
for i in range(min(10, len(feature_names))):
    print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Compare with PCA approach
print("\nComparing with PCA approach...")
# Load the trial features from the previous analysis
print("Extracting trial features for PCA comparison...")

# Extract trial features (same as in the PCA analysis)
def extract_trial_features(trial_data):
    """Extract features from a trial by averaging across epochs for each channel"""
    # Group by channel and compute mean across epochs
    channel_features = trial_data.groupby('channel')[feature_cols].mean()
    # Flatten the channel features into a single vector
    features = channel_features.values.flatten()
    return features

# Extract features for all trials
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

# Apply PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=100)  # 100 components gave best results in previous analysis
X_pca = pca.fit_transform(StandardScaler().fit_transform(X_trials))

# Split data
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y_trials, test_size=0.3, random_state=42, stratify=y_trials
)

# Train Random Forest on PCA features
rf_pca = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_pca.fit(X_train_pca, y_train_pca)

# Evaluate
y_pred_pca = rf_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test_pca, y_pred_pca)
print(f"PCA approach accuracy: {accuracy_pca:.4f}")
print(f"Domain-specific + Ensemble accuracy: {accuracy:.4f}")

# Combine approaches
print("\nCombining domain-specific features with PCA features...")
# Create a new ensemble with both feature sets
X_combined_train = np.hstack((X_train, X_train_pca))
X_combined_test = np.hstack((X_test, X_test_pca))

# Train combined model
combined_clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
combined_clf.fit(X_combined_train, y_train)

# Evaluate
y_pred_combined = combined_clf.predict(X_combined_test)
accuracy_combined = accuracy_score(y_test, y_pred_combined)
print(f"Combined approach accuracy: {accuracy_combined:.4f}")

# Print classification report for combined approach
print("\nClassification Report (Combined Approach):")
print(classification_report(y_test, y_pred_combined))

# Plot confusion matrix for combined approach
plt.figure(figsize=(10, 8))
cm_combined = confusion_matrix(y_test, y_pred_combined)
sns.heatmap(cm_combined, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Combined Approach')
plt.tight_layout()
plt.savefig('combined_confusion_matrix.png')
plt.close()

# Update README with results
print("\nDomain-specific feature engineering and ensemble classification completed.")
print("Results saved to disk.")
