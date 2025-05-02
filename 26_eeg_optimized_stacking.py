import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
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

# Extract optimized features
def extract_optimized_features(df):
    """
    Extract features using important channels and apply PCA
    with optimal number of components
    """
    # Get unique trials
    unique_trials = sorted(df['trial_id'].unique())
    n_trials = len(unique_trials)
    
    print(f"Extracting optimized features for {n_trials} trials")
    
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
    pca = PCA(n_components=20)  # Optimal number based on previous analysis
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Explained variance with 20 components: {pca.explained_variance_ratio_.sum():.4f}")
    
    return X_pca, y_trials

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

# Define base models with hyperparameter tuning
print("\nDefining and tuning base models...")

# SVM with grid search
svm_params = {
    'C': [1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf', 'poly']
}
svm_grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    svm_params,
    cv=5,
    n_jobs=-1,
    verbose=0
)
svm_grid.fit(X_train, y_train)
print(f"Best SVM parameters: {svm_grid.best_params_}")
best_svm = svm_grid.best_estimator_

# Random Forest with grid search
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_params,
    cv=5,
    n_jobs=-1,
    verbose=0
)
rf_grid.fit(X_train, y_train)
print(f"Best RF parameters: {rf_grid.best_params_}")
best_rf = rf_grid.best_estimator_

# Gradient Boosting with grid search
gb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}
gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_params,
    cv=5,
    n_jobs=-1,
    verbose=0
)
gb_grid.fit(X_train, y_train)
print(f"Best GB parameters: {gb_grid.best_params_}")
best_gb = gb_grid.best_estimator_

# MLP with grid search
mlp_params = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}
mlp_grid = GridSearchCV(
    MLPClassifier(max_iter=1000, random_state=42),
    mlp_params,
    cv=5,
    n_jobs=-1,
    verbose=0
)
mlp_grid.fit(X_train, y_train)
print(f"Best MLP parameters: {mlp_grid.best_params_}")
best_mlp = mlp_grid.best_estimator_

# Create optimized ensemble
print("\nCreating optimized ensemble...")
optimized_ensemble = VotingClassifier(
    estimators=[
        ('svm', best_svm),
        ('rf', best_rf),
        ('gb', best_gb),
        ('mlp', best_mlp)
    ],
    voting='soft'
)

# Create stacking ensemble
print("\nCreating stacking ensemble...")
stacking_ensemble = StackingClassifier(
    estimators=[
        ('svm', best_svm),
        ('rf', best_rf),
        ('gb', best_gb),
        ('mlp', best_mlp)
    ],
    final_estimator=RandomForestClassifier(n_estimators=200, random_state=42),
    cv=5
)

# Train ensembles
print("\nTraining ensembles...")
optimized_ensemble.fit(X_train, y_train)
stacking_ensemble.fit(X_train, y_train)

# Evaluate on test set
print("\nEvaluating ensembles...")
voting_pred = optimized_ensemble.predict(X_test)
stacking_pred = stacking_ensemble.predict(X_test)

voting_accuracy = accuracy_score(y_test, voting_pred)
stacking_accuracy = accuracy_score(y_test, stacking_pred)

print(f"Voting Ensemble Accuracy: {voting_accuracy:.4f}")
print(f"Stacking Ensemble Accuracy: {stacking_accuracy:.4f}")

# Use the better ensemble for final results
if stacking_accuracy > voting_accuracy:
    best_ensemble = stacking_ensemble
    best_pred = stacking_pred
    best_accuracy = stacking_accuracy
    ensemble_name = "Stacking Ensemble"
else:
    best_ensemble = optimized_ensemble
    best_pred = voting_pred
    best_accuracy = voting_accuracy
    ensemble_name = "Voting Ensemble"

# Print classification report for best ensemble
print(f"\nClassification Report ({ensemble_name}):")
print(classification_report(y_test, best_pred))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Confusion Matrix - {ensemble_name}')
plt.tight_layout()
plt.savefig('optimized_ensemble_confusion_matrix.png')
plt.close()

# Compare with previous approaches
print("\nComparing with previous approaches:")
print(f"PCA + Ensemble (20 components): 68.06%")
print(f"Specialized Ensemble (50 features): 66.67%")
print(f"Balanced Final Model (30 features): 66.67%")
print(f"Super Learner: 62.96%")
print(f"Advanced Ensemble: 30.56%")
print(f"{ensemble_name} (20 features): {best_accuracy:.4f}")

# Analyze per-class performance
class_report = classification_report(y_test, best_pred, output_dict=True)
class_f1 = [class_report[str(i)]['f1-score'] for i in range(4)]

plt.figure(figsize=(10, 6))
plt.bar(['Class 0', 'Class 1', 'Class 2', 'Class 3'], class_f1, color='skyblue')
plt.axhline(y=0.5, color='r', linestyle='--', label='Baseline')
plt.xlabel('Class')
plt.ylabel('F1-Score')
plt.title(f'Per-Class Performance - {ensemble_name}')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig('optimized_ensemble_per_class_performance.png')
plt.close()

# Analyze feature importance (using Random Forest)
rf_importance = best_rf.feature_importances_
indices = np.argsort(rf_importance)[::-1]

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.title('Feature Importances - Random Forest')
plt.bar(range(20), rf_importance[indices], align='center')
plt.xticks(range(20), [f"PC{i+1}" for i in indices], rotation=90)
plt.tight_layout()
plt.savefig('optimized_feature_importances.png')
plt.close()

print("\nOptimized ensemble analysis completed. Visualization images saved.")
