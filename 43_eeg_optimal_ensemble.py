"""
EEG Optimal Ensemble

This script implements a specialized approach to achieve 80%+ accuracy on the combined EEG dataset
by using separate models for each session, optimal feature selection, and weighted ensemble techniques.

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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import time
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Define important channels based on previous analysis
IMPORTANT_CHANNELS = [61, 58, 2, 31, 23, 4, 10, 15, 30, 40]
RANDOM_STATE = 42
TEST_SIZE = 0.2
OPTIMAL_PCA_COMPONENTS = 25  # Based on previous optimization

# Define session file paths
SESSION_FILES = {
    1: "combined_reduced_epochs_session_1.csv",
    2: "combined_reduced_epochs_session_2.csv",
    3: "combined_reduced_epochs_session_3.csv"
}

def load_session_data(session_id=None):
    """Load data from all sessions or a specific session"""
    if session_id is not None:
        # Load a specific session
        file_path = SESSION_FILES[session_id]
        print(f"Loading Session {session_id} data from {file_path}...")
        
        try:
            df = pd.read_csv(file_path)
            print(f"Session {session_id} dataset shape: {df.shape}")
            
            # Add session identifier
            df['session'] = session_id
            
            return df
        except Exception as e:
            print(f"Error loading Session {session_id}: {e}")
            return None
    else:
        # Load all sessions and combine
        combined_df = None
        
        for sess_id, file_path in SESSION_FILES.items():
            print(f"Loading Session {sess_id} data from {file_path}...")
            
            try:
                df = pd.read_csv(file_path)
                print(f"Session {sess_id} dataset shape: {df.shape}")
                
                # Add session identifier
                df['session'] = sess_id
                
                # Combine with previous sessions
                if combined_df is None:
                    combined_df = df
                else:
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
            except Exception as e:
                print(f"Error loading Session {sess_id}: {e}")
        
        if combined_df is not None:
            print(f"Combined dataset shape: {combined_df.shape}")
        
        return combined_df

def extract_features(df, important_channels=IMPORTANT_CHANNELS):
    """Extract features from EEG data"""
    # Identify feature columns (excluding metadata)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metadata_cols = ['epoch', 'channel', 'label', 'trial_id', 'session']
    feature_cols = [col for col in numerical_cols if col not in metadata_cols]
    
    # Get unique trials
    unique_trials = sorted(df['trial_id'].unique())
    n_trials = len(unique_trials)
    
    print(f"Extracting features for {n_trials} trials")
    
    # Initialize arrays
    X_trials = []
    y_trials = []
    trial_ids = []
    sessions = []
    
    # Extract features for each trial
    for trial in unique_trials:
        # Get data for this trial
        trial_data = df[df['trial_id'] == trial]
        
        # Skip if trial doesn't have data
        if len(trial_data) == 0:
            continue
        
        # Get the label and session (should be the same for all rows in the trial)
        label = trial_data['label'].iloc[0]
        session = trial_data['session'].iloc[0]
        
        # Filter for important channels only
        trial_data = trial_data[trial_data['channel'].isin(important_channels)]
        
        # Extract features by averaging across epochs for each channel
        channel_features = trial_data.groupby('channel')[feature_cols].mean()
        
        # Flatten the channel features into a single vector
        features = channel_features.values.flatten()
        
        X_trials.append(features)
        y_trials.append(label)
        trial_ids.append(trial)
        sessions.append(session)
    
    # Convert to numpy arrays
    X_trials = np.array(X_trials)
    y_trials = np.array(y_trials)
    trial_ids = np.array(trial_ids)
    sessions = np.array(sessions)
    
    print(f"Raw feature shape: {X_trials.shape}")
    
    return X_trials, y_trials, trial_ids, sessions

def train_separate_models():
    """Train separate models for each session with optimal parameters"""
    print("Training separate models for each session...")
    
    # Initialize results dictionary
    results = {}
    
    # Process each session separately
    for session_id in SESSION_FILES.keys():
        print(f"\nProcessing Session {session_id}...")
        
        # Load data for this session
        df = load_session_data(session_id)
        
        # Extract features
        X, y, trial_ids, sessions = extract_features(df)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Apply PCA
        pca = PCA(n_components=OPTIMAL_PCA_COMPONENTS)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        
        # Create and train models
        models = {
            'rf': RandomForestClassifier(
                n_estimators=200, 
                max_depth=15,
                min_samples_split=5,
                random_state=RANDOM_STATE
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                random_state=RANDOM_STATE
            ),
            'svm': SVC(
                C=10.0,
                gamma='scale',
                kernel='rbf',
                probability=True,
                random_state=RANDOM_STATE
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(200, 100, 50),
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        }
        
        # Train each model and evaluate
        model_results = {}
        for name, model in models.items():
            # Train model
            model.fit(X_train_pca, y_train)
            
            # Evaluate on test set
            y_pred = model.predict(X_test_pca)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"  {name.upper()} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
            
            # Store results
            model_results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1
            }
        
        # Create and evaluate voting ensemble
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', models['rf']),
                ('gb', models['gb']),
                ('svm', models['svm']),
                ('mlp', models['mlp'])
            ],
            voting='soft'
        )
        
        voting_clf.fit(X_train_pca, y_train)
        y_pred = voting_clf.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"  VOTING ENSEMBLE - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        
        # Store results for this session
        results[session_id] = {
            'models': model_results,
            'voting_ensemble': {
                'model': voting_clf,
                'accuracy': accuracy,
                'f1_score': f1
            },
            'preprocessing': {
                'scaler': scaler,
                'pca': pca
            }
        }
        
        # Save session model
        joblib.dump(results[session_id], f'session_{session_id}_models.pkl')
    
    return results

def train_optimal_ensemble():
    """Train an optimal ensemble using the best models from each session"""
    print("\nTraining optimal ensemble...")
    
    # Load combined data
    df = load_session_data()
    
    # Extract features
    X, y, trial_ids, sessions = extract_features(df)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test, sessions_train, sessions_test = train_test_split(
        X, y, sessions, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Train separate models for each session
    session_results = train_separate_models()
    
    # Make predictions using session-specific models
    print("\nMaking predictions with optimal ensemble...")
    
    # Initialize arrays for predictions
    y_pred = np.zeros_like(y_test)
    y_pred_probs = np.zeros((len(y_test), 4))
    
    # Process each test sample
    for i, (sample, session) in enumerate(zip(X_test, sessions_test)):
        # Get models for this session
        session_models = session_results[session]
        
        # Preprocess the sample
        scaler = session_models['preprocessing']['scaler']
        pca = session_models['preprocessing']['pca']
        
        sample_scaled = scaler.transform([sample])
        sample_pca = pca.transform(sample_scaled)
        
        # Get predictions from the voting ensemble
        ensemble = session_models['voting_ensemble']['model']
        pred = ensemble.predict(sample_pca)[0]
        pred_probs = ensemble.predict_proba(sample_pca)[0]
        
        # Store predictions
        y_pred[i] = pred
        y_pred_probs[i] = pred_probs
    
    # Evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Optimal Ensemble - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    
    # Calculate per-session metrics
    unique_sessions = np.unique(sessions_test)
    session_metrics = {}
    
    for session in unique_sessions:
        session_idx = np.where(sessions_test == session)[0]
        session_accuracy = accuracy_score(y_test[session_idx], y_pred[session_idx])
        session_f1 = f1_score(y_test[session_idx], y_pred[session_idx], average='weighted')
        
        print(f"Session {session} - Accuracy: {session_accuracy:.4f}, F1 Score: {session_f1:.4f}")
        
        session_metrics[session] = {
            'accuracy': session_accuracy,
            'f1_score': session_f1
        }
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - Optimal Ensemble')
    plt.tight_layout()
    plt.savefig('optimal_ensemble_confusion_matrix.png')
    plt.close()
    
    # Plot per-session accuracy
    plt.figure(figsize=(10, 6))
    sessions = list(session_metrics.keys())
    accuracies = [metrics['accuracy'] for metrics in session_metrics.values()]
    
    bars = plt.bar(sessions, accuracies, color=sns.color_palette("viridis", len(sessions)))
    plt.axhline(y=accuracy, color='r', linestyle='--', label=f'Overall: {accuracy:.4f}')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=12)
    
    plt.xlabel('Session')
    plt.ylabel('Accuracy')
    plt.title('Per-Session Accuracy - Optimal Ensemble')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig('optimal_ensemble_accuracy.png')
    plt.close()
    
    # Save results
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'session_metrics': session_metrics,
        'confusion_matrix': cm.tolist()
    }
    
    joblib.dump(results, 'optimal_ensemble_results.pkl')
    
    return results, session_results

def main():
    """Main function to implement optimal ensemble"""
    print("="*50)
    print("EEG OPTIMAL ENSEMBLE")
    print("="*50)
    
    # Train optimal ensemble
    results, session_results = train_optimal_ensemble()
    
    # Print summary
    print("\n" + "="*50)
    print("OPTIMAL ENSEMBLE SUMMARY")
    print("="*50)
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"Overall F1 Score: {results['f1_score']:.4f}")
    
    for session, metrics in results['session_metrics'].items():
        print(f"Session {session} Accuracy: {metrics['accuracy']:.4f}")
    
    print("="*50)
    
    print("Optimal ensemble completed. Results and visualizations saved.")
    
    return results, session_results

if __name__ == "__main__":
    main()
