"""
EEG Model Comparison on Combined Dataset

This script compares the optimized stacking ensemble and Deep PCA model
on the combined dataset using the optimal number of PCA components.

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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1_l2
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
        
        # Add session identifier to trial_id
        df['trial_id'] = f'sess{session_id}_' + df['trial_id'].astype(str)
        df['session'] = session_id  # Add session column for tracking
        
        return df
    except Exception as e:
        print(f"Error loading Session {session_id}: {e}")
        return None

def extract_features(df, n_components=22):
    """Extract features using important channels and apply PCA"""
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
        trial_data = trial_data[trial_data['channel'].isin(IMPORTANT_CHANNELS)]
        
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
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_trials)
    
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Explained variance with {n_components} components: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"PCA feature shape: {X_pca.shape}")
    print(f"Class distribution: {np.bincount(y_trials)}")
    
    return X_pca, y_trials, trial_ids, sessions, pca, scaler

def create_optimized_stacking_ensemble():
    """Create the optimized stacking ensemble model"""
    # Define base models
    base_models = [
        ('svm', SVC(probability=True, C=10, gamma='scale', kernel='rbf', random_state=RANDOM_STATE)),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, random_state=RANDOM_STATE)),
        ('gb', GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), alpha=0.001, max_iter=1000, random_state=RANDOM_STATE))
    ]
    
    # Define meta-learner
    meta_learner = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    
    # Create stacking ensemble
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    
    return stacking

def create_deep_pca_model(input_shape, num_classes=4):
    """Create the Deep PCA model"""
    model = Sequential([
        # Input layer
        Dense(512, activation='relu', input_shape=(input_shape,), 
              kernel_regularizer=l1_l2(l1=5e-5, l2=5e-4)),
        BatchNormalization(),
        Dropout(0.5),
        
        # Hidden layers
        Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=5e-5, l2=5e-4)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=5e-5, l2=5e-4)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=5e-5, l2=5e-4)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_evaluate_models(X, y, n_components):
    """Train and evaluate both models on the combined dataset"""
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
    
    # 1. Train and evaluate optimized stacking ensemble
    print("\nTraining Optimized Stacking Ensemble...")
    stacking_model = create_optimized_stacking_ensemble()
    
    start_time = time.time()
    stacking_model.fit(X_train, y_train)
    stacking_train_time = time.time() - start_time
    
    start_time = time.time()
    stacking_pred = stacking_model.predict(X_test)
    stacking_inference_time = time.time() - start_time
    
    stacking_accuracy = accuracy_score(y_test, stacking_pred)
    stacking_f1 = f1_score(y_test, stacking_pred, average='weighted')
    
    print(f"Stacking Ensemble Accuracy: {stacking_accuracy:.4f}")
    print(f"Stacking Ensemble F1 Score: {stacking_f1:.4f}")
    print(f"Training Time: {stacking_train_time:.4f}s, Inference Time: {stacking_inference_time:.4f}s")
    
    # 2. Train and evaluate Deep PCA model
    print("\nTraining Deep PCA Model...")
    deep_pca_model = create_deep_pca_model(X_train.shape[1])
    
    # Convert labels to numpy array
    y_train_np = np.array(y_train)
    y_test_np = np.array(y_test)
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    start_time = time.time()
    history = deep_pca_model.fit(
        X_train, y_train_np,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    deep_pca_train_time = time.time() - start_time
    
    start_time = time.time()
    deep_pca_pred_probs = deep_pca_model.predict(X_test)
    deep_pca_inference_time = time.time() - start_time
    
    deep_pca_pred = np.argmax(deep_pca_pred_probs, axis=1)
    deep_pca_accuracy = accuracy_score(y_test_np, deep_pca_pred)
    deep_pca_f1 = f1_score(y_test_np, deep_pca_pred, average='weighted')
    
    print(f"Deep PCA Model Accuracy: {deep_pca_accuracy:.4f}")
    print(f"Deep PCA Model F1 Score: {deep_pca_f1:.4f}")
    print(f"Training Time: {deep_pca_train_time:.4f}s, Inference Time: {deep_pca_inference_time:.4f}s")
    
    # Save confusion matrices
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, stacking_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - Stacking Ensemble with {n_components} PCA Components')
    plt.tight_layout()
    plt.savefig(f'stacking_cm_{n_components}_components.png')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test_np, deep_pca_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - Deep PCA Model with {n_components} PCA Components')
    plt.tight_layout()
    plt.savefig(f'deep_pca_cm_{n_components}_components.png')
    plt.close()
    
    # Save classification reports
    with open(f'stacking_report_{n_components}_components.txt', 'w') as f:
        f.write(f"Classification Report - Stacking Ensemble with {n_components} PCA Components\n\n")
        f.write(classification_report(y_test, stacking_pred))
    
    with open(f'deep_pca_report_{n_components}_components.txt', 'w') as f:
        f.write(f"Classification Report - Deep PCA Model with {n_components} PCA Components\n\n")
        f.write(classification_report(y_test_np, deep_pca_pred))
    
    # Plot training history for Deep PCA model
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'deep_pca_training_history_{n_components}.png')
    plt.close()
    
    # Compare models
    models = ['Stacking Ensemble', 'Deep PCA Model']
    accuracies = [stacking_accuracy, deep_pca_accuracy]
    f1_scores = [stacking_f1, deep_pca_f1]
    train_times = [stacking_train_time, deep_pca_train_time]
    inference_times = [stacking_inference_time, deep_pca_inference_time]
    
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    bars = plt.bar(models, accuracies, color=['blue', 'green'])
    plt.axhline(y=0.25, color='r', linestyle='--', label='Random Baseline')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=12)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1.0)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    bars = plt.bar(models, f1_scores, color=['blue', 'green'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=12)
    plt.ylabel('F1 Score')
    plt.title('Model F1 Score Comparison')
    plt.ylim(0, 1.0)
    
    plt.subplot(2, 2, 3)
    bars = plt.bar(models, train_times, color=['blue', 'green'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=12)
    plt.ylabel('Time (seconds)')
    plt.title('Training Time Comparison')
    
    plt.subplot(2, 2, 4)
    bars = plt.bar(models, inference_times, color=['blue', 'green'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}s', ha='center', va='bottom', fontsize=12)
    plt.ylabel('Time (seconds)')
    plt.title('Inference Time Comparison')
    
    plt.tight_layout()
    plt.savefig(f'model_comparison_{n_components}_components.png')
    plt.close()
    
    return {
        'stacking': {
            'accuracy': stacking_accuracy,
            'f1_score': stacking_f1,
            'train_time': stacking_train_time,
            'inference_time': stacking_inference_time
        },
        'deep_pca': {
            'accuracy': deep_pca_accuracy,
            'f1_score': deep_pca_f1,
            'train_time': deep_pca_train_time,
            'inference_time': deep_pca_inference_time
        }
    }

def main():
    """Main function to compare models on combined dataset"""
    print("="*50)
    print("EEG MODEL COMPARISON ON COMBINED DATASET")
    print("="*50)
    
    # Load data from all sessions
    all_dfs = []
    for session_id in SESSION_FILES.keys():
        df = load_session_data(session_id)
        if df is not None:
            all_dfs.append(df)
    
    # Combine all session data
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Number of unique trials: {combined_df['trial_id'].nunique()}")
    print(f"Number of unique labels: {combined_df['label'].nunique()}")
    
    # Define optimal PCA components based on our optimization results
    optimal_components = 25  # From the PCA component optimization
    
    # Extract features with optimal PCA components
    X, y, trial_ids, sessions, pca, scaler = extract_features(combined_df, n_components=optimal_components)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X, y, optimal_components)
    
    # Save results
    joblib.dump(results, f'model_comparison_results_{optimal_components}_components.pkl')
    
    # Print summary
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    print(f"PCA Components: {optimal_components}")
    print(f"Stacking Ensemble Accuracy: {results['stacking']['accuracy']:.4f}")
    print(f"Deep PCA Model Accuracy: {results['deep_pca']['accuracy']:.4f}")
    print(f"Best Model: {max(results, key=lambda x: results[x]['accuracy'])}")
    print("="*50)
    
    print("Model comparison completed. Results and visualizations saved.")
    
    return results

if __name__ == "__main__":
    main()
