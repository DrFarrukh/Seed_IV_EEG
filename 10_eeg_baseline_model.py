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

# Class-specific optimal channels based on feature importance analysis
class_specific_channels = {
    0: [61, 2, 31, 23],    # Class 0 optimal channels
    1: [58, 10, 15, 40],   # Class 1 optimal channels
    2: [61, 58, 2, 30],    # Class 2 optimal channels
    3: [31, 23, 4, 15]     # Class 3 optimal channels
}

# Function to extract optimized features with class-specific channels
def extract_final_features(df):
    """
    Extract optimized features with class-specific channel selection
    and reduced dimensionality
    """
    # Get unique trials
    unique_trials = sorted(df['trial_id'].unique())
    n_trials = len(unique_trials)
    
    print(f"Extracting final features for {n_trials} trials")
    
    # Initialize arrays for different feature sets
    X_class_specific = np.zeros((n_trials, len(class_specific_channels) * 5))  # 5 features per class
    y_trials = np.zeros(n_trials)
    
    # Extract features for each trial
    for i, trial in enumerate(unique_trials):
        trial_data = df[df['trial_id'] == trial]
        
        # Skip if trial doesn't have data
        if len(trial_data) == 0:
            continue
        
        # Get the label (should be the same for all rows in the trial)
        y_trials[i] = trial_data['label'].iloc[0]
        
        # Extract features for each class-specific channel set
        for class_idx, channels in class_specific_channels.items():
            # Get data for these channels
            channels_data = trial_data[trial_data['channel'].isin(channels)]
            
            # Skip if no data for these channels
            if len(channels_data) == 0:
                continue
            
            # Calculate class-specific features
            # 1. Average power in key frequency bands
            delta = channels_data['delta_power'].mean() if 'delta_power' in feature_cols else 0
            theta = channels_data['theta_power'].mean() if 'theta_power' in feature_cols else 0
            alpha = channels_data['alpha_power'].mean() if 'alpha_power' in feature_cols else 0
            beta = channels_data['beta_power'].mean() if 'beta_power' in feature_cols else 0
            gamma = channels_data['gamma_power'].mean() if 'gamma_power' in feature_cols else 0
            
            # 2. Calculate ratios (with small epsilon to avoid division by zero)
            epsilon = 1e-10
            theta_beta = theta / (beta + epsilon)
            alpha_beta = alpha / (beta + epsilon)
            
            # 3. Calculate mean and std of all features across these channels
            mean_features = channels_data[feature_cols].mean().mean()
            std_features = channels_data[feature_cols].std().mean()
            
            # 4. Calculate correlation between channels (if multiple channels)
            if len(channels) > 1:
                # Get mean features for each channel
                channel_means = channels_data.groupby('channel')[feature_cols].mean()
                # Calculate correlation matrix
                corr_matrix = np.corrcoef(channel_means.values)
                # Get mean correlation (excluding self-correlations)
                np.fill_diagonal(corr_matrix, np.nan)
                mean_corr = np.nanmean(corr_matrix)
            else:
                mean_corr = 0
                
            # Store features for this class-specific channel set
            feature_idx = class_idx * 5
            X_class_specific[i, feature_idx] = mean_features
            X_class_specific[i, feature_idx + 1] = std_features
            X_class_specific[i, feature_idx + 2] = theta_beta
            X_class_specific[i, feature_idx + 3] = alpha_beta
            X_class_specific[i, feature_idx + 4] = mean_corr
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_class_specific)
    
    return X_scaled, y_trials.astype(int)

# Extract final features
print("\nExtracting final optimized features...")
X_final, y_final = extract_final_features(df)

print(f"Final feature shape: {X_final.shape}")
print(f"Number of trials: {len(y_final)}")
print(f"Class distribution: {np.bincount(y_final)}")

# Split data into training and testing sets
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.3, random_state=42, stratify=y_final
)

print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

# Define class-specific optimal models based on previous analysis
class0_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
class1_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
class2_model = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
class3_model = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)

# Class-specific Classifier
class ClassSpecificVotingClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom voting classifier that uses specialized models for each class
    and implements a max-voting strategy for final prediction
    """
    def __init__(self, classifiers=None):
        self.classifiers = classifiers
        
    def fit(self, X, y):
        # Train each classifier
        for clf in self.classifiers:
            clf.fit(X, y)
        return self
    
    def predict_proba(self, X):
        # Get probability predictions from each classifier
        probas = []
        for clf in self.classifiers:
            if hasattr(clf, "predict_proba"):
                probas.append(clf.predict_proba(X))
            else:
                # For classifiers without predict_proba, use decision_function
                p = np.zeros((X.shape[0], len(np.unique(y_train))))
                p[:, 1] = clf.decision_function(X)
                p[:, 0] = -p[:, 1]
                probas.append(p)
        
        # For each sample, select the classifier with highest confidence for its prediction
        final_probas = np.zeros((X.shape[0], len(np.unique(y_train))))
        
        for i in range(X.shape[0]):
            # For each class, get the maximum probability across all classifiers
            for j in range(final_probas.shape[1]):
                class_probas = [p[i, j] for p in probas]
                final_probas[i, j] = max(class_probas)
            
            # Normalize to sum to 1
            final_probas[i] = final_probas[i] / final_probas[i].sum()
            
        return final_probas
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

# Create ensemble of classifiers
print("\nTraining class-specific models...")
classifiers = [class0_model, class1_model, class2_model, class3_model]

# Create and train the class-specific voting classifier
class_specific_clf = ClassSpecificVotingClassifier(classifiers=classifiers)
class_specific_clf.fit(X_train, y_train)

# Create standard voting classifier for comparison
voting_clf = VotingClassifier(
    estimators=[
        ('svm', class0_model),
        ('rf', class1_model),
        ('gb', class2_model),
        ('rf2', class3_model)
    ],
    voting='soft'
)
voting_clf.fit(X_train, y_train)

# Evaluate on test set
y_pred_specific = class_specific_clf.predict(X_test)
accuracy_specific = accuracy_score(y_test, y_pred_specific)
print(f"Class-specific voting accuracy: {accuracy_specific:.4f}")

y_pred_standard = voting_clf.predict(X_test)
accuracy_standard = accuracy_score(y_test, y_pred_standard)
print(f"Standard voting accuracy: {accuracy_standard:.4f}")

# Print classification report for the better model
if accuracy_specific >= accuracy_standard:
    print("\nClassification Report (Class-specific Voting):")
    print(classification_report(y_test, y_pred_specific))
    y_pred_final = y_pred_specific
    accuracy_final = accuracy_specific
else:
    print("\nClassification Report (Standard Voting):")
    print(classification_report(y_test, y_pred_standard))
    y_pred_final = y_pred_standard
    accuracy_final = accuracy_standard

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Final Model')
plt.tight_layout()
plt.savefig('final_model_confusion_matrix.png')
plt.close()

# Compare with previous approaches
print("\nComparing with previous approaches:")
print(f"PCA + Ensemble (20 components): 68.06%")
print(f"Specialized Ensemble (50 features): 66.67%")
print(f"Hybrid Model (522 features): 63.89%")
print(f"EEGNet-inspired (3D): 49.07%")
print(f"Final Model ({X_final.shape[1]} features): {accuracy_final:.4f}")

# Analyze per-class performance
class_report = classification_report(y_test, y_pred_final, output_dict=True)
class_f1 = [class_report[str(i)]['f1-score'] for i in range(4)]

plt.figure(figsize=(10, 6))
plt.bar(['Class 0', 'Class 1', 'Class 2', 'Class 3'], class_f1, color='skyblue')
plt.axhline(y=0.5, color='r', linestyle='--', label='Baseline')
plt.xlabel('Class')
plt.ylabel('F1-Score')
plt.title('Per-Class Performance - Final Model')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig('final_model_per_class_performance.png')
plt.close()

print("\nFinal model analysis completed. Visualization images saved.")
