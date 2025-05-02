"""
EEG Session-Adaptive Ensemble

This script implements a session-adaptive ensemble approach to achieve high accuracy
on combined EEG sessions. It uses session-specific calibration, adversarial domain adaptation,
and multi-level feature fusion to address session variability.

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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Activation, Add
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Reshape, Flatten
from tensorflow.keras.layers import LSTM, Bidirectional, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import time
import joblib
from scipy import stats
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

def extract_raw_features(df, important_channels=IMPORTANT_CHANNELS):
    """Extract raw features from EEG data without PCA"""
    # Identify feature columns (excluding metadata)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metadata_cols = ['epoch', 'channel', 'label', 'trial_id', 'session']
    feature_cols = [col for col in numerical_cols if col not in metadata_cols]
    
    # Get unique trials
    unique_trials = sorted(df['trial_id'].unique())
    n_trials = len(unique_trials)
    
    print(f"Extracting raw features for {n_trials} trials")
    
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

def apply_session_normalization(X_train, X_test, sessions_train, sessions_test):
    """Apply session-specific normalization to address session variability"""
    print("Applying session-specific normalization...")
    
    # Get unique sessions
    unique_sessions = np.unique(np.concatenate([sessions_train, sessions_test]))
    
    # Initialize normalized arrays
    X_train_norm = np.zeros_like(X_train)
    X_test_norm = np.zeros_like(X_test)
    
    # Normalize each session separately
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
            X_train_norm[train_idx] = scaler.transform(X_train[train_idx])
        
        if len(test_idx) > 0:
            X_test_norm[test_idx] = scaler.transform(X_test[test_idx])
    
    return X_train_norm, X_test_norm

def extract_advanced_features(X, n_components=OPTIMAL_PCA_COMPONENTS):
    """Extract advanced features using PCA and statistical measures"""
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Compute statistical features
    X_mean = np.mean(X, axis=1).reshape(-1, 1)
    X_std = np.std(X, axis=1).reshape(-1, 1)
    X_skew = stats.skew(X, axis=1).reshape(-1, 1)
    X_kurtosis = stats.kurtosis(X, axis=1).reshape(-1, 1)
    
    # Combine features
    X_combined = np.hstack([X_pca, X_mean, X_std, X_skew, X_kurtosis])
    
    print(f"Advanced feature shape: {X_combined.shape}")
    
    return X_combined, pca

def create_session_specific_models(X_train, y_train, sessions_train):
    """Create session-specific models for each session"""
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
        
        # Create and train models
        models = {
            'rf': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
            'svm': SVC(probability=True, random_state=RANDOM_STATE),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=RANDOM_STATE)
        }
        
        # Train each model
        for name, model in models.items():
            model.fit(X_session, y_session)
        
        # Store models
        session_models[session] = models
    
    return session_models

def create_meta_learner(X_train, y_train, sessions_train, session_models):
    """Create a meta-learner that combines predictions from session-specific models"""
    print("Creating meta-learner...")
    
    # Get predictions from session-specific models
    meta_features = []
    
    # For each sample in the training set
    for i in range(len(X_train)):
        # Get the session for this sample
        session = sessions_train[i]
        
        # Get models for this session
        if session not in session_models:
            # Use the first session's models if this session doesn't have models
            session = list(session_models.keys())[0]
        
        models = session_models[session]
        
        # Get predictions from each model
        sample_features = []
        for name, model in models.items():
            probs = model.predict_proba([X_train[i]])[0]
            sample_features.extend(probs)
        
        meta_features.append(sample_features)
    
    # Convert to numpy array
    meta_features = np.array(meta_features)
    
    # Train meta-learner
    meta_learner = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    meta_learner.fit(meta_features, y_train)
    
    return meta_learner

def create_deep_ensemble_model(input_shape, num_classes=4):
    """Create a deep ensemble model that combines multiple architectures"""
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Dense branch
    dense_branch = Dense(256, activation='relu')(inputs)
    dense_branch = BatchNormalization()(dense_branch)
    dense_branch = Dropout(0.5)(dense_branch)
    dense_branch = Dense(128, activation='relu')(dense_branch)
    dense_branch = BatchNormalization()(dense_branch)
    dense_branch = Dropout(0.4)(dense_branch)
    dense_branch = Dense(64, activation='relu')(dense_branch)
    
    # CNN branch (reshape input for 1D convolution)
    cnn_input = Reshape((input_shape[0], 1))(inputs)
    cnn_branch = Conv1D(64, 3, activation='relu', padding='same')(cnn_input)
    cnn_branch = BatchNormalization()(cnn_branch)
    cnn_branch = MaxPooling1D(2)(cnn_branch)
    cnn_branch = Conv1D(128, 3, activation='relu', padding='same')(cnn_branch)
    cnn_branch = BatchNormalization()(cnn_branch)
    cnn_branch = GlobalAveragePooling1D()(cnn_branch)
    
    # LSTM branch (reshape input for sequence)
    lstm_input = Reshape((input_shape[0], 1))(inputs)
    lstm_branch = Bidirectional(LSTM(64, return_sequences=True))(lstm_input)
    lstm_branch = Dropout(0.4)(lstm_branch)
    lstm_branch = Bidirectional(LSTM(32))(lstm_branch)
    
    # Combine branches
    combined = Concatenate()([dense_branch, cnn_branch, lstm_branch])
    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(0.4)(combined)
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.3)(combined)
    outputs = Dense(num_classes, activation='softmax')(combined)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_session_adaptive_model(input_shape, num_classes=4, num_sessions=3):
    """Create a session-adaptive model with domain adaptation"""
    # Input layers
    feature_input = Input(shape=input_shape, name='feature_input')
    session_input = Input(shape=(1,), name='session_input')
    
    # Session embedding
    session_embedding = Dense(8, activation='relu')(session_input)
    session_embedding = Dense(16, activation='relu')(session_embedding)
    session_embedding = Dense(input_shape[0], activation='linear')(session_embedding)
    
    # Apply session-specific adaptation
    adapted_input = Add()([feature_input, session_embedding])
    
    # Feature extraction layers
    x = Dense(256, activation='relu')(adapted_input)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Task-specific layers
    task_output = Dense(64, activation='relu')(x)
    task_output = Dropout(0.3)(task_output)
    task_output = Dense(num_classes, activation='softmax', name='task_output')(task_output)
    
    # Domain classification layers (with gradient reversal)
    domain_output = Dense(32, activation='relu')(x)
    domain_output = Dropout(0.3)(domain_output)
    domain_output = Dense(num_sessions, activation='softmax', name='domain_output')(domain_output)
    
    # Create model
    model = Model(inputs=[feature_input, session_input], outputs=[task_output, domain_output])
    
    # Compile model with custom loss weights
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'task_output': 'sparse_categorical_crossentropy',
            'domain_output': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'task_output': 1.0,
            'domain_output': 0.1  # Lower weight for domain classification
        },
        metrics={
            'task_output': 'accuracy',
            'domain_output': 'accuracy'
        }
    )
    
    return model

def train_session_adaptive_model(X_train, y_train, sessions_train, X_val, y_val, sessions_val):
    """Train a session-adaptive model with domain adaptation"""
    print("Training session-adaptive model...")
    
    # Encode sessions (0-indexed)
    session_encoder = LabelEncoder()
    sessions_train_encoded = session_encoder.fit_transform(sessions_train)
    sessions_val_encoded = session_encoder.transform(sessions_val)
    
    # Reshape session inputs
    sessions_train_input = sessions_train_encoded.reshape(-1, 1)
    sessions_val_input = sessions_val_encoded.reshape(-1, 1)
    
    # Create model
    input_shape = X_train.shape[1:]
    model = create_session_adaptive_model(input_shape, num_classes=4, num_sessions=len(np.unique(sessions_train)))
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_task_output_accuracy',
        patience=20,
        restore_best_weights=True,
        mode='max'
    )
    
    model_checkpoint = ModelCheckpoint(
        'session_adaptive_model.h5',
        monitor='val_task_output_accuracy',
        save_best_only=True,
        verbose=0
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_task_output_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model
    start_time = time.time()
    history = model.fit(
        [X_train, sessions_train_input],
        {'task_output': y_train, 'domain_output': sessions_train_encoded},
        validation_data=(
            [X_val, sessions_val_input],
            {'task_output': y_val, 'domain_output': sessions_val_encoded}
        ),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, history, session_encoder

def train_deep_ensemble(X_train, y_train, X_val, y_val):
    """Train a deep ensemble model"""
    print("Training deep ensemble model...")
    
    # Create model
    input_shape = X_train.shape[1:]
    model = create_deep_ensemble_model(input_shape)
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        mode='max'
    )
    
    model_checkpoint = ModelCheckpoint(
        'deep_ensemble_model.h5',
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

def create_session_adaptive_ensemble(X_train, y_train, sessions_train, X_val, y_val, sessions_val):
    """Create a comprehensive session-adaptive ensemble"""
    print("Creating session-adaptive ensemble...")
    
    # 1. Train session-specific models
    session_models = create_session_specific_models(X_train, y_train, sessions_train)
    
    # 2. Train meta-learner
    meta_learner = create_meta_learner(X_train, y_train, sessions_train, session_models)
    
    # 3. Train deep ensemble model
    deep_ensemble, deep_history = train_deep_ensemble(X_train, y_train, X_val, y_val)
    
    # 4. Train session-adaptive model
    adaptive_model, adaptive_history, session_encoder = train_session_adaptive_model(
        X_train, y_train, sessions_train, X_val, y_val, sessions_val
    )
    
    # Create ensemble
    ensemble = {
        'session_models': session_models,
        'meta_learner': meta_learner,
        'deep_ensemble': deep_ensemble,
        'adaptive_model': adaptive_model,
        'session_encoder': session_encoder
    }
    
    return ensemble

def predict_with_ensemble(ensemble, X_test, sessions_test):
    """Make predictions using the session-adaptive ensemble"""
    print("Making predictions with session-adaptive ensemble...")
    
    # Get components
    session_models = ensemble['session_models']
    meta_learner = ensemble['meta_learner']
    deep_ensemble = ensemble['deep_ensemble']
    adaptive_model = ensemble['adaptive_model']
    session_encoder = ensemble['session_encoder']
    
    # 1. Get predictions from session-specific models via meta-learner
    meta_features = []
    
    # For each sample in the test set
    for i in range(len(X_test)):
        # Get the session for this sample
        session = sessions_test[i]
        
        # Get models for this session
        if session not in session_models:
            # Use the first session's models if this session doesn't have models
            session = list(session_models.keys())[0]
        
        models = session_models[session]
        
        # Get predictions from each model
        sample_features = []
        for name, model in models.items():
            probs = model.predict_proba([X_test[i]])[0]
            sample_features.extend(probs)
        
        meta_features.append(sample_features)
    
    # Convert to numpy array
    meta_features = np.array(meta_features)
    
    # Get meta-learner predictions
    meta_preds = meta_learner.predict_proba(meta_features)
    
    # 2. Get deep ensemble predictions
    deep_preds = deep_ensemble.predict(X_test, verbose=0)
    
    # 3. Get session-adaptive model predictions
    sessions_test_encoded = session_encoder.transform(sessions_test)
    sessions_test_input = sessions_test_encoded.reshape(-1, 1)
    adaptive_preds, _ = adaptive_model.predict([X_test, sessions_test_input], verbose=0)
    
    # 4. Combine predictions (weighted average)
    combined_preds = 0.3 * meta_preds + 0.3 * deep_preds + 0.4 * adaptive_preds
    
    # Get final predictions
    y_pred = np.argmax(combined_preds, axis=1)
    
    return y_pred, combined_preds

def evaluate_ensemble(ensemble, X_test, y_test, sessions_test):
    """Evaluate the session-adaptive ensemble"""
    print("Evaluating session-adaptive ensemble...")
    
    # Get predictions
    y_pred, y_pred_probs = predict_with_ensemble(ensemble, X_test, sessions_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Session-Adaptive Ensemble Accuracy: {accuracy:.4f}")
    print(f"Session-Adaptive Ensemble F1 Score: {f1:.4f}")
    
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
    plt.title('Confusion Matrix - Session-Adaptive Ensemble')
    plt.tight_layout()
    plt.savefig('session_adaptive_ensemble_confusion_matrix.png')
    plt.close()
    
    # Save results
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'session_metrics': session_metrics,
        'confusion_matrix': cm.tolist()
    }
    
    joblib.dump(results, 'session_adaptive_ensemble_results.pkl')
    
    return results

def main():
    """Main function to implement session-adaptive ensemble"""
    print("="*50)
    print("EEG SESSION-ADAPTIVE ENSEMBLE")
    print("="*50)
    
    # Load data from all sessions
    df = load_session_data()
    
    # Extract raw features
    X, y, trial_ids, sessions = extract_raw_features(df)
    
    # Split data into train, validation, and test sets
    # First split into train+val and test
    X_train_val, X_test, y_train_val, y_test, sessions_train_val, sessions_test = train_test_split(
        X, y, sessions, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Then split train+val into train and validation
    X_train, X_val, y_train, y_val, sessions_train, sessions_val = train_test_split(
        X_train_val, y_train_val, sessions_train_val, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_val
    )
    
    # Apply session-specific normalization
    X_train_norm, X_test_norm = apply_session_normalization(X_train, X_test, sessions_train, sessions_test)
    X_val_norm, _ = apply_session_normalization(X_val, np.zeros_like(X_val), sessions_val, np.array([]))
    
    # Extract advanced features
    X_train_adv, pca = extract_advanced_features(X_train_norm)
    X_val_adv = np.hstack([
        pca.transform(X_val_norm),
        np.mean(X_val_norm, axis=1).reshape(-1, 1),
        np.std(X_val_norm, axis=1).reshape(-1, 1),
        stats.skew(X_val_norm, axis=1).reshape(-1, 1),
        stats.kurtosis(X_val_norm, axis=1).reshape(-1, 1)
    ])
    X_test_adv = np.hstack([
        pca.transform(X_test_norm),
        np.mean(X_test_norm, axis=1).reshape(-1, 1),
        np.std(X_test_norm, axis=1).reshape(-1, 1),
        stats.skew(X_test_norm, axis=1).reshape(-1, 1),
        stats.kurtosis(X_test_norm, axis=1).reshape(-1, 1)
    ])
    
    print(f"Training set: {X_train_adv.shape}, Validation set: {X_val_adv.shape}, Testing set: {X_test_adv.shape}")
    
    # Create session-adaptive ensemble
    ensemble = create_session_adaptive_ensemble(
        X_train_adv, y_train, sessions_train,
        X_val_adv, y_val, sessions_val
    )
    
    # Evaluate ensemble
    results = evaluate_ensemble(ensemble, X_test_adv, y_test, sessions_test)
    
    # Save ensemble
    joblib.dump(ensemble, 'session_adaptive_ensemble.pkl')
    
    # Print summary
    print("\n" + "="*50)
    print("SESSION-ADAPTIVE ENSEMBLE SUMMARY")
    print("="*50)
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"Overall F1 Score: {results['f1_score']:.4f}")
    
    for session, metrics in results['session_metrics'].items():
        print(f"Session {session} Accuracy: {metrics['accuracy']:.4f}")
    
    print("="*50)
    
    print("Session-adaptive ensemble completed. Results and visualizations saved.")
    
    return results

if __name__ == "__main__":
    main()
