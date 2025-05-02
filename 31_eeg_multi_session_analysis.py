"""
EEG Multi-Session Analysis

This script analyzes EEG data from multiple sessions (1, 2, and 3) and performs:
1. Individual session analysis
2. Combined sessions analysis
3. Cross-session transfer learning

Author: Dr. Farrukh
Date: May 2025
"""

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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Define important channels based on previous analysis
IMPORTANT_CHANNELS = [61, 58, 2, 31, 23, 4, 10, 15, 30, 40]
N_PCA_COMPONENTS = 20
RANDOM_STATE = 42
TEST_SIZE = 0.3

# Define session file paths
SESSION_FILES = {
    1: "combined_reduced_epochs_session_1.csv",
    2: "combined_reduced_epochs_session_2.csv",
    3: "combined_reduced_epochs_session_3.csv"
}

def load_session_data(session_id):
    """Load data for a specific session"""
    file_path = SESSION_FILES[session_id]
    print(f"Loading Session {session_id} data from {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Session {session_id} dataset shape: {df.shape}")
        print(f"Number of unique trials: {df['trial_id'].nunique()}")
        print(f"Number of unique labels: {df['label'].nunique()}")
        
        # Add session identifier to trial_id
        df['trial_id'] = f'sess{session_id}_' + df['trial_id'].astype(str)
        
        return df
    except Exception as e:
        print(f"Error loading Session {session_id}: {e}")
        return None

def extract_features(df, scaler=None, pca=None):
    """
    Extract features using important channels and apply PCA
    Returns features, labels, trial_ids, and optionally the fitted scaler and PCA
    """
    # Identify feature columns (excluding metadata)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metadata_cols = ['epoch', 'channel', 'label', 'trial_id']
    feature_cols = [col for col in numerical_cols if col not in metadata_cols]
    
    # Get unique trials
    unique_trials = sorted(df['trial_id'].unique())
    n_trials = len(unique_trials)
    
    print(f"Extracting features for {n_trials} trials")
    
    # Initialize arrays
    X_trials = []
    y_trials = []
    trial_ids = []
    
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
        trial_data = trial_data[trial_data['channel'].isin(IMPORTANT_CHANNELS)]
        
        # Extract features by averaging across epochs for each channel
        channel_features = trial_data.groupby('channel')[feature_cols].mean()
        
        # Flatten the channel features into a single vector
        features = channel_features.values.flatten()
        
        X_trials.append(features)
        y_trials.append(label)
        trial_ids.append(trial)
    
    # Convert to numpy arrays
    X_trials = np.array(X_trials)
    y_trials = np.array(y_trials)
    trial_ids = np.array(trial_ids)
    
    print(f"Raw feature shape: {X_trials.shape}")
    
    # Standardize features
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_trials)
    else:
        X_scaled = scaler.transform(X_trials)
    
    # Apply PCA to reduce dimensionality
    if pca is None:
        pca = PCA(n_components=N_PCA_COMPONENTS)
        X_pca = pca.fit_transform(X_scaled)
        print(f"Explained variance with {N_PCA_COMPONENTS} components: {pca.explained_variance_ratio_.sum():.4f}")
    else:
        X_pca = pca.transform(X_scaled)
    
    print(f"Optimized feature shape: {X_pca.shape}")
    print(f"Class distribution: {np.bincount(y_trials)}")
    
    return X_pca, y_trials, trial_ids, scaler, pca

def train_and_evaluate_models(X_train, y_train, X_test, y_test, session_name=""):
    """Train and evaluate models on the given data"""
    print(f"\nTraining models for {session_name}...")
    print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
    
    # Define base models with hyperparameter tuning
    print("\nDefining and tuning base models...")
    
    # SVM with grid search
    svm_params = {
        'kernel': ['linear', 'rbf'],
        'C': [1, 10, 100],
        'gamma': ['scale', 0.01, 0.1]
    }
    svm_grid = GridSearchCV(
        SVC(probability=True, random_state=RANDOM_STATE),
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
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [5, 10]
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
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
        GradientBoostingClassifier(random_state=RANDOM_STATE),
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
        MLPClassifier(max_iter=1000, random_state=RANDOM_STATE),
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
    voting_ensemble = VotingClassifier(
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
        final_estimator=RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        cv=5
    )
    
    # Train ensembles
    print("\nTraining ensembles...")
    voting_ensemble.fit(X_train, y_train)
    stacking_ensemble.fit(X_train, y_train)
    
    # Evaluate on test set
    print("\nEvaluating ensembles...")
    voting_pred = voting_ensemble.predict(X_test)
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
        best_ensemble = voting_ensemble
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
    plt.title(f'Confusion Matrix - {ensemble_name} ({session_name})')
    plt.tight_layout()
    plt.savefig(f'{session_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.close()
    
    # Perform cross-validation for more robust evaluation
    print("\nPerforming 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    X_full = np.vstack((X_train, X_test))
    y_full = np.hstack((y_train, y_test))
    cv_scores = []
    
    for train_idx, test_idx in cv.split(X_full, y_full):
        # Split data
        X_cv_train, X_cv_test = X_full[train_idx], X_full[test_idx]
        y_cv_train, y_cv_test = y_full[train_idx], y_full[test_idx]
        
        # Train model
        best_ensemble.fit(X_cv_train, y_cv_train)
        
        # Evaluate
        cv_pred = best_ensemble.predict(X_cv_test)
        cv_acc = accuracy_score(y_cv_test, cv_pred)
        cv_scores.append(cv_acc)
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Save the best model
    model_filename = f'{session_name.lower().replace(" ", "_")}_best_model.pkl'
    print(f"\nSaving best model to {model_filename}...")
    joblib.dump(best_ensemble, model_filename)
    
    return {
        'best_ensemble': best_ensemble,
        'ensemble_name': ensemble_name,
        'best_accuracy': best_accuracy,
        'cv_accuracy': f"{np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}",
        'base_models': {
            'svm': best_svm,
            'rf': best_rf,
            'gb': best_gb,
            'mlp': best_mlp
        }
    }

def cross_session_analysis(sessions_data, results):
    """Analyze performance across sessions"""
    print("\n" + "="*50)
    print("CROSS-SESSION ANALYSIS")
    print("="*50)
    
    # Create a table of results
    print("\nPerformance Summary:")
    print("-"*80)
    print(f"{'Session':<20} {'Best Ensemble':<20} {'Test Accuracy':<15} {'CV Accuracy':<20}")
    print("-"*80)
    
    for session_name, result in results.items():
        print(f"{session_name:<20} {result['ensemble_name']:<20} {result['best_accuracy']:.4f} {result['cv_accuracy']:<20}")
    
    print("-"*80)
    
    # Plot performance comparison
    plt.figure(figsize=(12, 6))
    session_names = list(results.keys())
    accuracies = [results[session]['best_accuracy'] for session in session_names]
    
    plt.bar(session_names, accuracies, color='skyblue')
    plt.axhline(y=0.25, color='r', linestyle='--', label='Random Baseline')
    plt.xlabel('Session')
    plt.ylabel('Accuracy')
    plt.title('Classification Performance Across Sessions')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig('cross_session_performance.png')
    plt.close()
    
    # Analyze cross-session generalization
    print("\nCross-Session Generalization:")
    print("-"*80)
    print(f"{'Train Session':<15} {'Test Session':<15} {'Accuracy':<10}")
    print("-"*80)
    
    cross_session_accs = []
    
    for train_session, train_data in sessions_data.items():
        train_name = f"Session {train_session}"
        train_model = results[train_name]['best_ensemble']
        
        for test_session, test_data in sessions_data.items():
            if train_session != test_session:
                test_name = f"Session {test_session}"
                X_test, y_test = test_data['X'], test_data['y']
                
                # Predict on test session
                y_pred = train_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                print(f"{train_name:<15} {test_name:<15} {accuracy:.4f}")
                cross_session_accs.append((train_name, test_name, accuracy))
    
    print("-"*80)
    
    # Plot cross-session generalization
    plt.figure(figsize=(12, 8))
    train_sessions = [item[0] for item in cross_session_accs]
    test_sessions = [item[1] for item in cross_session_accs]
    accuracies = [item[2] for item in cross_session_accs]
    
    x = np.arange(len(cross_session_accs))
    plt.bar(x, accuracies, color='lightgreen')
    plt.axhline(y=0.25, color='r', linestyle='--', label='Random Baseline')
    plt.xticks(x, [f"{t}->{e}" for t, e in zip(train_sessions, test_sessions)], rotation=45)
    plt.xlabel('Train -> Test Session')
    plt.ylabel('Accuracy')
    plt.title('Cross-Session Generalization')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig('cross_session_generalization.png')
    plt.close()
    
    return cross_session_accs

def main():
    """Main function to run the analysis"""
    print("="*50)
    print("EEG MULTI-SESSION ANALYSIS")
    print("="*50)
    
    # Load data from all sessions
    session_dfs = {}
    for session_id in SESSION_FILES.keys():
        df = load_session_data(session_id)
        if df is not None:
            session_dfs[session_id] = df
    
    if not session_dfs:
        print("No session data loaded. Exiting.")
        return
    
    # Analyze each session individually
    session_results = {}
    sessions_data = {}
    
    for session_id, df in session_dfs.items():
        print("\n" + "="*50)
        print(f"SESSION {session_id} ANALYSIS")
        print("="*50)
        
        # Extract features
        X, y, trial_ids, scaler, pca = extract_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        # Train and evaluate models
        session_name = f"Session {session_id}"
        result = train_and_evaluate_models(X_train, y_train, X_test, y_test, session_name)
        session_results[session_name] = result
        
        # Save data for cross-session analysis
        sessions_data[session_id] = {
            'X': X,
            'y': y,
            'trial_ids': trial_ids,
            'scaler': scaler,
            'pca': pca
        }
    
    # Analyze combined sessions
    print("\n" + "="*50)
    print("COMBINED SESSIONS ANALYSIS")
    print("="*50)
    
    # Combine all session data
    combined_df = pd.concat(session_dfs.values(), ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Number of unique trials: {combined_df['trial_id'].nunique()}")
    print(f"Number of unique labels: {combined_df['label'].nunique()}")
    
    # Extract features from combined data
    X_combined, y_combined, trial_ids_combined, scaler_combined, pca_combined = extract_features(combined_df)
    
    # Split combined data
    X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(
        X_combined, y_combined, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_combined
    )
    
    # Train and evaluate models on combined data
    combined_result = train_and_evaluate_models(
        X_train_combined, y_train_combined, X_test_combined, y_test_combined, "Combined Sessions"
    )
    session_results["Combined Sessions"] = combined_result
    
    # Perform cross-session analysis
    cross_session_accs = cross_session_analysis(sessions_data, session_results)
    
    # Save all results
    print("\nSaving session models and results...")
    joblib.dump(session_results, 'multi_session_results.pkl')
    joblib.dump(scaler_combined, 'combined_sessions_scaler.pkl')
    joblib.dump(pca_combined, 'combined_sessions_pca.pkl')
    
    print("\nMulti-session analysis completed successfully.")
    
    # Return a summary of results
    return {
        'session_results': session_results,
        'cross_session_accs': cross_session_accs
    }

if __name__ == "__main__":
    main()
