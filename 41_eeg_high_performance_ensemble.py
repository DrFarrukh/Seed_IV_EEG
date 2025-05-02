"""
EEG High-Performance Ensemble

This script implements an advanced ensemble approach targeting 80%+ accuracy
on the combined EEG dataset by using session-specific calibration, data augmentation,
and specialized models for each session.

Author: Dr. Farrukh
Date: May 2025
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
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

def extract_features(df, important_channels=IMPORTANT_CHANNELS):
    """Extract features from EEG data with advanced processing"""
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
        
        # Extract additional statistical features for each channel
        channel_std = trial_data.groupby('channel')[feature_cols].std().values.flatten()
        channel_max = trial_data.groupby('channel')[feature_cols].max().values.flatten()
        channel_min = trial_data.groupby('channel')[feature_cols].min().values.flatten()
        
        # Flatten the channel features into a single vector
        features = np.concatenate([
            channel_features.values.flatten(),
            channel_std,
            channel_max,
            channel_min
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
    
    print(f"Raw feature shape: {X_trials.shape}")
    
    return X_trials, y_trials, trial_ids, sessions

def apply_session_calibration(X_train, X_test, sessions_train, sessions_test):
    """Apply session-specific calibration to address session variability"""
    print("Applying session-specific calibration...")
    
    # Get unique sessions
    unique_sessions = np.unique(np.concatenate([sessions_train, sessions_test]))
    
    # Initialize calibrated arrays
    X_train_cal = np.zeros_like(X_train)
    X_test_cal = np.zeros_like(X_test)
    
    # Calibrate each session separately
    for session in unique_sessions:
        # Get indices for this session
        train_idx = np.where(sessions_train == session)[0]
        test_idx = np.where(sessions_test == session)[0]
        
        # Skip if no samples for this session
        if len(train_idx) == 0:
            continue
        
        # Fit scaler on training data for this session
        scaler = StandardScaler()
        scaler.fit(X_train[train_idx])
        
        # Transform training and test data for this session
        if len(train_idx) > 0:
            X_train_cal[train_idx] = scaler.transform(X_train[train_idx])
        
        if len(test_idx) > 0:
            X_test_cal[test_idx] = scaler.transform(X_test[test_idx])
    
    return X_train_cal, X_test_cal

def augment_data(X, y, sessions, augmentation_factor=2):
    """Augment data to increase sample size and diversity"""
    print(f"Augmenting data by factor of {augmentation_factor}...")
    
    # Initialize augmented arrays
    X_aug = [X]
    y_aug = [y]
    sessions_aug = [sessions]
    
    # Get unique sessions
    unique_sessions = np.unique(sessions)
    
    # Augment each session separately
    for session in unique_sessions:
        # Get indices for this session
        session_idx = np.where(sessions == session)[0]
        
        # Skip if not enough samples
        if len(session_idx) < 10:
            continue
        
        # Get data for this session
        X_session = X[session_idx]
        y_session = y[session_idx]
        sessions_session = sessions[session_idx]
        
        # Apply different augmentation techniques
        for i in range(augmentation_factor - 1):
            # 1. Add Gaussian noise
            noise_level = 0.05 * (i + 1)
            X_noise = X_session + np.random.normal(0, noise_level, X_session.shape)
            X_aug.append(X_noise)
            y_aug.append(y_session)
            sessions_aug.append(sessions_session)
            
            # 2. Feature-wise scaling
            scale_factors = np.random.uniform(0.9, 1.1, X_session.shape[1])
            X_scale = X_session * scale_factors
            X_aug.append(X_scale)
            y_aug.append(y_session)
            sessions_aug.append(sessions_session)
    
    # Combine augmented data
    X_combined = np.vstack(X_aug)
    y_combined = np.concatenate(y_aug)
    sessions_combined = np.concatenate(sessions_aug)
    
    print(f"Augmented data shape: {X_combined.shape}")
    
    return X_combined, y_combined, sessions_combined

def reduce_dimensions(X_train, X_test, n_components=OPTIMAL_PCA_COMPONENTS):
    """Apply PCA for dimensionality reduction"""
    print(f"Reducing dimensions to {n_components} components...")
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    return X_train_pca, X_test_pca, pca

def create_session_specific_models(X_train, y_train, sessions_train):
    """Create optimized models for each session"""
    print("Creating session-specific models...")
    
    # Get unique sessions
    unique_sessions = np.unique(sessions_train)
    
    # Initialize models dictionary
    session_models = {}
    
    # Create models for each session
    for session in unique_sessions:
        print(f"Training models for Session {session}...")
        
        # Get data for this session
        session_idx = np.where(sessions_train == session)[0]
        X_session = X_train[session_idx]
        y_session = y_train[session_idx]
        
        # Skip if not enough samples
        if len(X_session) < 20:
            print(f"Not enough samples for Session {session}, skipping...")
            continue
        
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
        
        # Train each model
        for name, model in models.items():
            model.fit(X_session, y_session)
        
        # Store models
        session_models[session] = models
    
    return session_models

def create_deep_model(input_shape, num_classes=4):
    """Create a deep neural network model"""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,), 
              kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_deep_model(X_train, y_train, X_val, y_val):
    """Train a deep neural network model"""
    print("Training deep neural network model...")
    
    # Create model
    input_shape = X_train.shape[1]
    model = create_deep_model(input_shape)
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        mode='max'
    )
    
    model_checkpoint = ModelCheckpoint(
        'deep_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, history

def create_high_performance_ensemble(X_train, y_train, sessions_train, X_val, y_val):
    """Create a high-performance ensemble of models"""
    print("Creating high-performance ensemble...")
    
    # 1. Train session-specific models
    session_models = create_session_specific_models(X_train, y_train, sessions_train)
    
    # 2. Train deep neural network model
    deep_model, _ = train_deep_model(X_train, y_train, X_val, y_val)
    
    # 3. Create ensemble
    ensemble = {
        'session_models': session_models,
        'deep_model': deep_model
    }
    
    return ensemble

def predict_with_ensemble(ensemble, X_test, sessions_test):
    """Make predictions using the high-performance ensemble"""
    print("Making predictions with high-performance ensemble...")
    
    # Get components
    session_models = ensemble['session_models']
    deep_model = ensemble['deep_model']
    
    # Initialize predictions array
    n_samples = len(X_test)
    n_classes = 4
    all_preds = np.zeros((n_samples, n_classes))
    
    # 1. Get predictions from session-specific models
    for i, session in enumerate(sessions_test):
        # Get models for this session
        if session not in session_models:
            # Use the first session's models if this session doesn't have models
            session = list(session_models.keys())[0]
        
        models = session_models[session]
        
        # Get predictions from each model
        sample_preds = np.zeros(n_classes)
        for name, model in models.items():
            probs = model.predict_proba([X_test[i]])[0]
            sample_preds += probs
        
        # Average predictions
        sample_preds /= len(models)
        all_preds[i] = sample_preds
    
    # 2. Get deep model predictions
    deep_preds = deep_model.predict(X_test, verbose=0)
    
    # 3. Combine predictions (weighted average)
    # Give more weight to session-specific models
    combined_preds = 0.7 * all_preds + 0.3 * deep_preds
    
    # Get final predictions
    y_pred = np.argmax(combined_preds, axis=1)
    
    return y_pred, combined_preds

def evaluate_ensemble(ensemble, X_test, y_test, sessions_test):
    """Evaluate the high-performance ensemble"""
    print("Evaluating high-performance ensemble...")
    
    # Get predictions
    y_pred, y_pred_probs = predict_with_ensemble(ensemble, X_test, sessions_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"High-Performance Ensemble Accuracy: {accuracy:.4f}")
    print(f"High-Performance Ensemble F1 Score: {f1:.4f}")
    
    # Calculate per-session metrics
    unique_sessions = np.unique(sessions_test)
    session_metrics = {}
    
    for session in unique_sessions:
        session_idx = np.where(sessions_test == session)[0]
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
    plt.title('Confusion Matrix - High-Performance Ensemble')
    plt.tight_layout()
    plt.savefig('high_performance_ensemble_confusion_matrix.png')
    plt.close()
    
    # Save results
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'session_metrics': session_metrics,
        'confusion_matrix': cm.tolist()
    }
    
    joblib.dump(results, 'high_performance_ensemble_results.pkl')
    
    return results

def main():
    """Main function to implement high-performance ensemble"""
    print("="*50)
    print("EEG HIGH-PERFORMANCE ENSEMBLE")
    print("="*50)
    
    # Load data from all sessions
    df = load_session_data()
    
    # Extract features with advanced processing
    X, y, trial_ids, sessions = extract_features(df)
    
    # Split data into train, validation, and test sets
    # First split into train+val and test
    X_train_val, X_test, y_train_val, y_test, sessions_train_val, sessions_test = train_test_split(
        X, y, sessions, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Then split train+val into train and validation
    X_train, X_val, y_train, y_val, sessions_train, sessions_val = train_test_split(
        X_train_val, y_train_val, sessions_train_val, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_val
    )
    
    # Apply session-specific calibration
    X_train_cal, X_test_cal = apply_session_calibration(X_train, X_test, sessions_train, sessions_test)
    X_val_cal, _ = apply_session_calibration(X_val, np.zeros_like(X_val), sessions_val, np.array([]))
    
    # Augment training data
    X_train_aug, y_train_aug, sessions_train_aug = augment_data(X_train_cal, y_train, sessions_train)
    
    # Reduce dimensions
    X_train_pca, X_test_pca, pca = reduce_dimensions(X_train_aug, X_test_cal)
    X_val_pca = pca.transform(X_val_cal)
    
    print(f"Training set: {X_train_pca.shape}, Validation set: {X_val_pca.shape}, Testing set: {X_test_pca.shape}")
    
    # Create high-performance ensemble
    ensemble = create_high_performance_ensemble(
        X_train_pca, y_train_aug, sessions_train_aug,
        X_val_pca, y_val
    )
    
    # Evaluate ensemble
    results = evaluate_ensemble(ensemble, X_test_pca, y_test, sessions_test)
    
    # Save ensemble
    joblib.dump(ensemble, 'high_performance_ensemble.pkl')
    
    # Print summary
    print("\n" + "="*50)
    print("HIGH-PERFORMANCE ENSEMBLE SUMMARY")
    print("="*50)
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"Overall F1 Score: {results['f1_score']:.4f}")
    
    for session, metrics in results['session_metrics'].items():
        print(f"Session {session} Accuracy: {metrics['accuracy']:.4f}")
    
    print("="*50)
    
    print("High-performance ensemble completed. Results and visualizations saved.")
    
    return results

if __name__ == "__main__":
    main()
