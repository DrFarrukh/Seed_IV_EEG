import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
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

# Define important channels based on previous analysis
important_channels = [61, 58, 2, 31, 23, 4, 10, 15, 30, 40]
print(f"Using top {len(important_channels)} important channels")

# Define important frequency bands based on EEG domain knowledge
power_bands = ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']
ratio_features = ['theta_beta_ratio', 'alpha_beta_ratio', 'delta_theta_ratio']

# Function to extract optimized features
def extract_optimized_features(df):
    """
    Extract optimized features based on previous insights:
    1. Focus on important channels
    2. Include power band ratios
    3. Apply PCA to reduce dimensionality
    """
    # Get unique trials
    unique_trials = sorted(df['trial_id'].unique())
    n_trials = len(unique_trials)
    
    print(f"Extracting optimized features for {n_trials} trials")
    
    # Initialize arrays for different feature sets
    X_channel = np.zeros((n_trials, len(important_channels) * len(feature_cols)))
    X_power_ratios = np.zeros((n_trials, len(important_channels) * 3))  # 3 power ratios
    y_trials = np.zeros(n_trials)
    
    # Extract features for each trial
    for i, trial in enumerate(unique_trials):
        trial_data = df[df['trial_id'] == trial]
        
        # Skip if trial doesn't have data
        if len(trial_data) == 0:
            continue
        
        # Get the label (should be the same for all rows in the trial)
        y_trials[i] = trial_data['label'].iloc[0]
        
        # Extract features for important channels
        for j, channel in enumerate(important_channels):
            channel_data = trial_data[trial_data['channel'] == channel]
            
            # Skip if channel doesn't have data
            if len(channel_data) == 0:
                continue
            
            # Average features across epochs for this channel
            start_idx = j * len(feature_cols)
            end_idx = start_idx + len(feature_cols)
            X_channel[i, start_idx:end_idx] = channel_data[feature_cols].mean().values
            
            # Calculate power band ratios if available
            if all(band in feature_cols for band in power_bands):
                # Extract power values
                delta = channel_data['delta_power'].mean()
                theta = channel_data['theta_power'].mean()
                alpha = channel_data['alpha_power'].mean()
                beta = channel_data['beta_power'].mean()
                gamma = channel_data['gamma_power'].mean()
                
                # Calculate ratios (with small epsilon to avoid division by zero)
                epsilon = 1e-10
                theta_beta = theta / (beta + epsilon)
                alpha_beta = alpha / (beta + epsilon)
                delta_theta = delta / (theta + epsilon)
                
                # Store ratios
                ratio_idx = j * 3
                X_power_ratios[i, ratio_idx] = theta_beta
                X_power_ratios[i, ratio_idx + 1] = alpha_beta
                X_power_ratios[i, ratio_idx + 2] = delta_theta
    
    # Standardize features
    scaler_channel = StandardScaler()
    X_channel_scaled = scaler_channel.fit_transform(X_channel)
    
    scaler_ratios = StandardScaler()
    X_ratios_scaled = scaler_ratios.fit_transform(X_power_ratios)
    
    # Apply PCA to channel features
    pca_channel = PCA(n_components=20)  # Reduce to 20 components
    X_channel_pca = pca_channel.fit_transform(X_channel_scaled)
    print(f"Channel PCA explained variance: {pca_channel.explained_variance_ratio_.sum():.4f}")
    
    # Combine features
    X_combined = np.hstack((X_channel_pca, X_ratios_scaled))
    
    return X_combined, y_trials.astype(int)

# Extract optimized features
print("\nExtracting optimized features...")
X_optimized, y_optimized = extract_optimized_features(df)

print(f"Optimized feature shape: {X_optimized.shape}")
print(f"Number of trials: {len(y_optimized)}")
print(f"Class distribution: {np.bincount(y_optimized)}")

# Split data into training and testing sets
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_optimized, y_optimized, test_size=0.3, random_state=42, stratify=y_optimized
)

print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

# Define specialized ensemble classifier
class SpecializedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier that uses specialized models for each class
    Each model is trained to distinguish one class from the rest
    Final prediction is based on the model with highest confidence
    """
    def __init__(self, models=None, classes=None):
        self.models = models
        self.classes = classes
        
    def fit(self, X, y):
        # If models not provided, create default models
        if self.models is None:
            self.models = [
                RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
                SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
                GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
                RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
            ]
        
        # Store classes
        self.classes = np.unique(y)
        
        # Train each model on binary classification task
        for i, model in enumerate(self.models):
            # Create binary target: 1 for current class, 0 for others
            binary_y = (y == i).astype(int)
            model.fit(X, binary_y)
        
        return self
    
    def predict_proba(self, X):
        # Get probability predictions from each model
        probas = np.zeros((X.shape[0], len(self.classes)))
        
        for i, model in enumerate(self.models):
            # Get probability of positive class (current class)
            class_probas = model.predict_proba(X)[:, 1]
            probas[:, i] = class_probas
        
        # Normalize probabilities to sum to 1
        row_sums = probas.sum(axis=1)
        normalized_probas = probas / row_sums[:, np.newaxis]
        
        return normalized_probas
    
    def predict(self, X):
        # Get probability predictions
        probas = self.predict_proba(X)
        
        # Return class with highest probability
        return np.argmax(probas, axis=1)

# Train specialized models for each class
print("\nTraining specialized models for each class...")

# Define models for each class based on previous performance
class0_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
class1_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
class2_model = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
class3_model = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)

specialized_models = [class0_model, class1_model, class2_model, class3_model]

# Create and train specialized ensemble
specialized_ensemble = SpecializedEnsembleClassifier(models=specialized_models)
specialized_ensemble.fit(X_train, y_train)

# Evaluate individual models for each class
print("\nEvaluating individual models for each class...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for i, (class_name, model) in enumerate(zip(['Class 0', 'Class 1', 'Class 2', 'Class 3'], specialized_models)):
    # Create binary target: 1 for current class, 0 for others
    binary_y = (y_train == i).astype(int)
    
    # Evaluate model
    cv_scores = cross_val_score(model, X_train, binary_y, cv=cv, scoring='roc_auc')
    print(f"{class_name} model (ROC-AUC): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Evaluate specialized ensemble
print("\nEvaluating specialized ensemble...")
cv_scores = cross_val_score(specialized_ensemble, X_train, y_train, cv=cv, scoring='accuracy')
print(f"Specialized Ensemble CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Evaluate on test set
y_pred = specialized_ensemble.predict(X_test)
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
plt.title('Confusion Matrix - Specialized Ensemble')
plt.tight_layout()
plt.savefig('specialized_ensemble_confusion_matrix.png')
plt.close()

# Compare with previous approaches
print("\nComparing with previous approaches:")
print(f"PCA + Ensemble: 67.59%")
print(f"Hybrid Model: 63.89%")
print(f"EEGNet-inspired: 49.07%")
print(f"Specialized Ensemble: {accuracy:.4f}")

# Analyze per-class performance
class_report = classification_report(y_test, y_pred, output_dict=True)
class_f1 = [class_report[str(i)]['f1-score'] for i in range(4)]

plt.figure(figsize=(10, 6))
plt.bar(['Class 0', 'Class 1', 'Class 2', 'Class 3'], class_f1, color='skyblue')
plt.axhline(y=0.5, color='r', linestyle='--', label='Baseline')
plt.xlabel('Class')
plt.ylabel('F1-Score')
plt.title('Per-Class Performance')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig('per_class_performance.png')
plt.close()

print("\nSpecialized ensemble analysis completed. Visualization images saved.")
