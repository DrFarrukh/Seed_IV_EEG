"""
EEG Session-Calibrated Ensemble

This script implements a specialized approach to achieve 80%+ accuracy on the combined EEG dataset
by using session-specific training, calibration, and advanced ensemble techniques.

Author: Dr. Farrukh
Date: May 2025
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Reshape, Flatten
from tensorflow.keras.layers import LSTM, Bidirectional, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import time
import joblib
from scipy import stats, signal
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

def extract_enhanced_features(df, important_channels=IMPORTANT_CHANNELS):
    """Extract enhanced features from EEG data with advanced processing"""
    # Identify feature columns (excluding metadata)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metadata_cols = ['epoch', 'channel', 'label', 'trial_id', 'session']
    feature_cols = [col for col in numerical_cols if col not in metadata_cols]
    
    # Get unique trials
    unique_trials = sorted(df['trial_id'].unique())
    n_trials = len(unique_trials)
    
    print(f"Extracting enhanced features for {n_trials} trials")
    
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
        
        # Extract additional statistical features for each channel
        channel_std = trial_data.groupby('channel')[feature_cols].std()
        channel_max = trial_data.groupby('channel')[feature_cols].max()
        channel_min = trial_data.groupby('channel')[feature_cols].min()
        
        # Extract frequency domain features
        freq_features = []
        for channel in important_channels:
            channel_data = trial_data[trial_data['channel'] == channel]
            if len(channel_data) > 0:
                # Get time series data for this channel
                time_series = channel_data[feature_cols].values.flatten()
                
                # Calculate power spectrum
                if len(time_series) > 1:
                    try:
                        f, pxx = signal.welch(time_series, fs=100, nperseg=min(256, len(time_series)))
                        # Extract key frequency bands (delta, theta, alpha, beta, gamma)
                        delta = np.mean(pxx[np.logical_and(f >= 0.5, f < 4)])
                        theta = np.mean(pxx[np.logical_and(f >= 4, f < 8)])
                        alpha = np.mean(pxx[np.logical_and(f >= 8, f < 13)])
                        beta = np.mean(pxx[np.logical_and(f >= 13, f < 30)])
                        gamma = np.mean(pxx[np.logical_and(f >= 30, f < 50)])
                        
                        freq_features.extend([delta, theta, alpha, beta, gamma])
                    except:
                        # If there's an error, add zeros
                        freq_features.extend([0, 0, 0, 0, 0])
                else:
                    freq_features.extend([0, 0, 0, 0, 0])
        
        # Combine all features
        features = np.concatenate([
            channel_features.values.flatten(),
            channel_std.values.flatten(),
            channel_max.values.flatten(),
            channel_min.values.flatten(),
            np.array(freq_features)
        ])
        
        X_trials.append(features)
        y_trials.append(label)
        trial_ids.append(trial)
        sessions.append(session)
    
    # Convert to numpy arrays
    X_trials = np.array(X_trials)
    y_trials = np.array(y_trials)
    trial_ids = np.array(trial_ids)
    sessions = np.array(sessions)
    
    print(f"Enhanced feature shape: {X_trials.shape}")
    
    return X_trials, y_trials, trial_ids, sessions

def train_session_specific_models(X, y, sessions):
    """Train separate models for each session"""
    print("Training session-specific models...")
    
    # Get unique sessions
    unique_sessions = np.unique(sessions)
    
    # Initialize models dictionary
    session_models = {}
    session_accuracies = {}
    
    # Train models for each session
    for session in unique_sessions:
        print(f"Training models for Session {session}...")
        
        # Get data for this session
        session_idx = np.where(sessions == session)[0]
        X_session = X[session_idx]
        y_session = y[session_idx]
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X_session, y_session, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_session
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Apply PCA
        pca = PCA(n_components=OPTIMAL_PCA_COMPONENTS)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        # Create and train models with optimized hyperparameters
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
        model_accuracies = {}
        for name, model in models.items():
            model.fit(X_train_pca, y_train)
            y_pred = model.predict(X_test_pca)
            accuracy = accuracy_score(y_test, y_pred)
            model_accuracies[name] = accuracy
            print(f"  {name.upper()} Accuracy: {accuracy:.4f}")
        
        # Store models and preprocessing components
        session_models[session] = {
            'models': models,
            'scaler': scaler,
            'pca': pca
        }
        
        # Store accuracies
        session_accuracies[session] = model_accuracies
    
    return session_models, session_accuracies

def create_calibration_model(X, y, sessions):
    """Create a model to calibrate predictions across sessions"""
    print("Creating session calibration model...")
    
    # Get unique sessions
    unique_sessions = np.unique(sessions)
    
    # Initialize arrays for calibration data
    X_cal = []
    y_cal = []
    
    # For each session
    for session in unique_sessions:
        # Get data for this session
        session_idx = np.where(sessions == session)[0]
        X_session = X[session_idx]
        y_session = y[session_idx]
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X_session, y_session, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_session
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Apply PCA
        pca = PCA(n_components=OPTIMAL_PCA_COMPONENTS)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        # Create and train models
        models = {
            'rf': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
            'svm': SVC(probability=True, random_state=RANDOM_STATE),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=RANDOM_STATE)
        }
        
        # Train each model
        for name, model in models.items():
            model.fit(X_train_pca, y_train)
        
        # Get predictions from each model
        test_preds = np.zeros((len(X_test_pca), 4, len(models)))
        for i, (name, model) in enumerate(models.items()):
            test_preds[:, :, i] = model.predict_proba(X_test_pca)
        
        # Flatten predictions and add to calibration data
        for i in range(len(X_test_pca)):
            X_cal.append(test_preds[i].flatten())
            y_cal.append(y_test[i])
    
    # Convert to numpy arrays
    X_cal = np.array(X_cal)
    y_cal = np.array(y_cal)
    
    # Train calibration model
    calibration_model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    calibration_model.fit(X_cal, y_cal)
    
    return calibration_model

def create_session_calibrated_ensemble(X, y, sessions):
    """Create a session-calibrated ensemble"""
    print("Creating session-calibrated ensemble...")
    
    # 1. Train session-specific models
    session_models, session_accuracies = train_session_specific_models(X, y, sessions)
    
    # 2. Create calibration model
    calibration_model = create_calibration_model(X, y, sessions)
    
    # 3. Create ensemble
    ensemble = {
        'session_models': session_models,
        'calibration_model': calibration_model,
        'session_accuracies': session_accuracies
    }
    
    return ensemble

def predict_with_ensemble(ensemble, X_test, sessions_test):
    """Make predictions using the session-calibrated ensemble"""
    print("Making predictions with session-calibrated ensemble...")
    
    # Get components
    session_models = ensemble['session_models']
    calibration_model = ensemble['calibration_model']
    
    # Initialize arrays
    n_samples = len(X_test)
    n_classes = 4
    n_models = 4  # rf, gb, svm, mlp
    
    # Initialize predictions array
    all_preds = np.zeros((n_samples, n_classes * n_models))
    
    # Process each sample
    for i in range(n_samples):
        # Get session for this sample
        session = sessions_test[i]
        
        # If session doesn't have models, use the session with highest accuracy
        if session not in session_models:
            # Find session with highest average accuracy
            session_accuracies = ensemble['session_accuracies']
            avg_accuracies = {s: np.mean(list(accs.values())) for s, accs in session_accuracies.items()}
            session = max(avg_accuracies, key=avg_accuracies.get)
        
        # Get models and preprocessing components for this session
        models = session_models[session]['models']
        scaler = session_models[session]['scaler']
        pca = session_models[session]['pca']
        
        # Preprocess the sample
        X_scaled = scaler.transform([X_test[i]])
        X_pca = pca.transform(X_scaled)
        
        # Get predictions from each model
        sample_preds = []
        for name, model in models.items():
            probs = model.predict_proba(X_pca)[0]
            sample_preds.extend(probs)
        
        # Store predictions
        all_preds[i] = np.array(sample_preds)
    
    # Apply calibration model
    y_pred = calibration_model.predict(all_preds)
    y_pred_probs = calibration_model.predict_proba(all_preds)
    
    return y_pred, y_pred_probs

def evaluate_ensemble(ensemble, X_test, y_test, sessions_test):
    """Evaluate the session-calibrated ensemble"""
    print("Evaluating session-calibrated ensemble...")
    
    # Get predictions
    y_pred, y_pred_probs = predict_with_ensemble(ensemble, X_test, sessions_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Session-Calibrated Ensemble Accuracy: {accuracy:.4f}")
    print(f"Session-Calibrated Ensemble F1 Score: {f1:.4f}")
    
    # Calculate per-session metrics
    unique_sessions = np.unique(sessions_test)
    session_metrics = {}
    
    for session in unique_sessions:
        session_idx = np.where(sessions_test == session)[0]
        if len(session_idx) > 0:
            session_accuracy = accuracy_score(y_test[session_idx], y_pred[session_idx])
            session_f1 = f1_score(y_test[session_idx], y_pred[session_idx], average='weighted')
            
            print(f"Session {session} Accuracy: {session_accuracy:.4f}")
            print(f"Session {session} F1 Score: {session_f1:.4f}")
            
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
    plt.title('Confusion Matrix - Session-Calibrated Ensemble')
    plt.tight_layout()
    plt.savefig('session_calibrated_ensemble_confusion_matrix.png')
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
    plt.title('Per-Session Accuracy - Session-Calibrated Ensemble')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig('session_calibrated_ensemble_accuracy.png')
    plt.close()
    
    # Save results
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'session_metrics': session_metrics,
        'confusion_matrix': cm.tolist()
    }
    
    joblib.dump(results, 'session_calibrated_ensemble_results.pkl')
    
    return results

def add_synthetic_samples(X, y, sessions, n_samples_per_class=50):
    """Add synthetic samples to balance the dataset and improve generalization"""
    print("Adding synthetic samples...")
    
    # Get unique classes
    unique_classes = np.unique(y)
    
    # Initialize arrays for synthetic data
    X_synthetic = []
    y_synthetic = []
    sessions_synthetic = []
    
    # For each class
    for cls in unique_classes:
        # Get samples for this class
        cls_idx = np.where(y == cls)[0]
        X_cls = X[cls_idx]
        
        # Calculate mean and covariance
        mean = np.mean(X_cls, axis=0)
        cov = np.cov(X_cls, rowvar=False)
        
        # Generate synthetic samples
        synthetic_samples = np.random.multivariate_normal(mean, cov, n_samples_per_class)
        
        # Add to synthetic data
        X_synthetic.append(synthetic_samples)
        y_synthetic.append(np.full(n_samples_per_class, cls))
        
        # Assign synthetic samples to a special "synthetic" session
        sessions_synthetic.append(np.full(n_samples_per_class, 0))  # 0 = synthetic session
    
    # Combine synthetic data
    X_synthetic = np.vstack(X_synthetic)
    y_synthetic = np.concatenate(y_synthetic)
    sessions_synthetic = np.concatenate(sessions_synthetic)
    
    # Combine with original data
    X_combined = np.vstack([X, X_synthetic])
    y_combined = np.concatenate([y, y_synthetic])
    sessions_combined = np.concatenate([sessions, sessions_synthetic])
    
    print(f"Added {len(X_synthetic)} synthetic samples")
    print(f"Combined data shape: {X_combined.shape}")
    
    return X_combined, y_combined, sessions_combined

def main():
    """Main function to implement session-calibrated ensemble"""
    print("="*50)
    print("EEG SESSION-CALIBRATED ENSEMBLE")
    print("="*50)
    
    # Load data from all sessions
    df = load_session_data()
    
    # Extract enhanced features
    X, y, trial_ids, sessions = extract_enhanced_features(df)
    
    # Add synthetic samples
    X, y, sessions = add_synthetic_samples(X, y, sessions)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test, sessions_train, sessions_test = train_test_split(
        X, y, sessions, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Create session-calibrated ensemble
    ensemble = create_session_calibrated_ensemble(X_train, y_train, sessions_train)
    
    # Evaluate ensemble
    results = evaluate_ensemble(ensemble, X_test, y_test, sessions_test)
    
    # Save ensemble
    joblib.dump(ensemble, 'session_calibrated_ensemble.pkl')
    
    # Print summary
    print("\n" + "="*50)
    print("SESSION-CALIBRATED ENSEMBLE SUMMARY")
    print("="*50)
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"Overall F1 Score: {results['f1_score']:.4f}")
    
    for session, metrics in results['session_metrics'].items():
        print(f"Session {session} Accuracy: {metrics['accuracy']:.4f}")
    
    print("="*50)
    
    print("Session-calibrated ensemble completed. Results and visualizations saved.")
    
    return results

if __name__ == "__main__":
    main()
