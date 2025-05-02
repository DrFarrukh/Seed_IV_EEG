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

# Identify feature columns (excluding metadata)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
metadata_cols = ['epoch', 'channel', 'label', 'trial_id']
feature_cols = [col for col in numerical_cols if col not in metadata_cols]
print(f"Number of features: {len(feature_cols)}")

# Define important channels based on previous analysis
important_channels = [61, 58, 2, 31, 23, 4, 10, 15, 30, 40]
print(f"Using top {len(important_channels)} important channels")

# Function to extract trial features with important channels only
def extract_balanced_features(df):
    """
    Extract features using only important channels and apply PCA
    to reduce dimensionality while preserving variance
    """
    # Get unique trials
    unique_trials = sorted(df['trial_id'].unique())
    n_trials = len(unique_trials)
    
    print(f"Extracting balanced features for {n_trials} trials")
    
    # Initialize arrays
    X_trials = []
    y_trials = []
    
    # Extract features for each trial
    for trial in unique_trials:
        # Get data for this trial
        trial_data = df[df['trial_id'] == trial]
        
        # Skip if trial doesn't have data
        if len(trial_data) == 0:
            continue
        
        # Get the label (should be the same for all rows in the trial)
        label = trial_data['label'].iloc[0]
        
        # Filter for important channels only
        trial_data = trial_data[trial_data['channel'].isin(important_channels)]
        
        # Extract features by averaging across epochs for each channel
        channel_features = trial_data.groupby('channel')[feature_cols].mean()
        
        # Flatten the channel features into a single vector
        features = channel_features.values.flatten()
        
        X_trials.append(features)
        y_trials.append(label)
    
    # Convert to numpy arrays
    X_trials = np.array(X_trials)
    y_trials = np.array(y_trials)
    
    print(f"Raw feature shape: {X_trials.shape}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_trials)
    
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=30)  # Balanced number of components
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Explained variance with 30 components: {pca.explained_variance_ratio_.sum():.4f}")
    
    return X_pca, y_trials

# Extract balanced features
print("\nExtracting balanced features...")
X_balanced, y_balanced = extract_balanced_features(df)

print(f"Balanced feature shape: {X_balanced.shape}")
print(f"Number of trials: {len(y_balanced)}")
print(f"Class distribution: {np.bincount(y_balanced)}")

# Split data into training and testing sets
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
)

print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

# Create class-specific classifiers
print("\nTraining class-specific models...")
class0_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
class1_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
class2_model = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
class3_model = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)

# Create voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('svm', class0_model),
        ('rf', class1_model),
        ('gb', class2_model),
        ('rf2', class3_model)
    ],
    voting='soft'
)

# Cross-validation
print("\nPerforming cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=cv, scoring='accuracy')
print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

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
plt.title('Confusion Matrix - Balanced Final Model')
plt.tight_layout()
plt.savefig('balanced_final_confusion_matrix.png')
plt.close()

# Compare with previous approaches
print("\nComparing with previous approaches:")
print(f"PCA + Ensemble (20 components): 68.06%")
print(f"Specialized Ensemble (50 features): 66.67%")
print(f"Hybrid Model (522 features): 63.89%")
print(f"Final Model (20 features): 41.67%")
print(f"Balanced Final Model (30 features): {accuracy:.4f}")

# Analyze per-class performance
class_report = classification_report(y_test, y_pred, output_dict=True)
class_f1 = [class_report[str(i)]['f1-score'] for i in range(4)]

plt.figure(figsize=(10, 6))
plt.bar(['Class 0', 'Class 1', 'Class 2', 'Class 3'], class_f1, color='skyblue')
plt.axhline(y=0.5, color='r', linestyle='--', label='Baseline')
plt.xlabel('Class')
plt.ylabel('F1-Score')
plt.title('Per-Class Performance - Balanced Final Model')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig('balanced_final_per_class_performance.png')
plt.close()

# Analyze feature importance (using Random Forest)
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)

# Get feature importances
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot top 15 features
plt.figure(figsize=(12, 8))
plt.title('Feature Importances - Balanced Final Model')
plt.bar(range(15), importances[indices[:15]], align='center')
plt.xticks(range(15), [f"PC{i+1}" for i in indices[:15]], rotation=90)
plt.tight_layout()
plt.savefig('balanced_final_feature_importances.png')
plt.close()

print("\nBalanced final model analysis completed. Visualization images saved.")
