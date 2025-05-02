"""
EEG Domain Adaptation

This script implements domain adaptation techniques to improve EEG classification
across different sessions by reducing domain shift between sessions.

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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
OPTIMAL_PCA_COMPONENTS = 25  # Based on previous optimization

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

def extract_features(df, n_components=OPTIMAL_PCA_COMPONENTS):
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
    
    return X_pca, y_trials, sessions, pca, scaler

def create_domain_adaptation_model(input_dim, num_classes=4):
    """
    Create a domain adaptation model with feature extractor, 
    label predictor, and domain classifier
    """
    # Feature extractor
    feature_input = Input(shape=(input_dim,))
    feature_layer = Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(feature_input)
    feature_layer = Dropout(0.5)(feature_layer)
    feature_layer = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(feature_layer)
    feature_layer = Dropout(0.4)(feature_layer)
    feature_layer = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(feature_layer)
    feature_output = Dropout(0.3)(feature_layer)
    
    feature_extractor = Model(inputs=feature_input, outputs=feature_output, name='feature_extractor')
    
    # Label predictor (task classifier)
    label_input = Input(shape=(64,))
    label_layer = Dense(32, activation='relu')(label_input)
    label_output = Dense(num_classes, activation='softmax')(label_layer)
    
    label_predictor = Model(inputs=label_input, outputs=label_output, name='label_predictor')
    
    # Domain classifier with gradient reversal layer
    domain_input = Input(shape=(64,))
    # Lambda layer for gradient reversal
    reverse_gradient = Lambda(lambda x: x * (-1), output_shape=lambda x: x)(domain_input)
    domain_layer = Dense(32, activation='relu')(reverse_gradient)
    domain_output = Dense(1, activation='sigmoid')(domain_layer)
    
    domain_classifier = Model(inputs=domain_input, outputs=domain_output, name='domain_classifier')
    
    # Combined model
    combined_input = Input(shape=(input_dim,))
    features = feature_extractor(combined_input)
    label_pred = label_predictor(features)
    domain_pred = domain_classifier(features)
    
    combined_model = Model(
        inputs=combined_input,
        outputs=[label_pred, domain_pred],
        name='domain_adaptation_model'
    )
    
    # Compile model
    combined_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'label_predictor': 'sparse_categorical_crossentropy',
            'domain_classifier': 'binary_crossentropy'
        },
        loss_weights={
            'label_predictor': 1.0,
            'domain_classifier': 0.1  # Weight for domain classification loss
        },
        metrics={
            'label_predictor': 'accuracy',
            'domain_classifier': 'accuracy'
        }
    )
    
    return combined_model, feature_extractor, label_predictor, domain_classifier

def train_domain_adaptation_model(source_data, target_data):
    """Train domain adaptation model using source and target domain data"""
    # Unpack data
    X_source, y_source, _, _, _ = source_data
    X_target, y_target, _, _, _ = target_data
    
    # Create domain labels (0 for source, 1 for target)
    domain_source = np.zeros(len(X_source))
    domain_target = np.ones(len(X_target))
    
    # Split source data into train and validation
    X_source_train, X_source_val, y_source_train, y_source_val, domain_source_train, domain_source_val = \
        train_test_split(X_source, y_source, domain_source, test_size=0.2, random_state=RANDOM_STATE, stratify=y_source)
    
    # Split target data into train and test
    X_target_train, X_target_test, y_target_train, y_target_test, domain_target_train, domain_target_test = \
        train_test_split(X_target, y_target, domain_target, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_target)
    
    # Combine source and target training data
    X_train = np.vstack((X_source_train, X_target_train))
    y_train = np.hstack((y_source_train, y_target_train))
    domain_train = np.hstack((domain_source_train, domain_target_train))
    
    # Combine source and target validation data
    X_val = np.vstack((X_source_val, X_target_train[:len(X_source_val)]))  # Use some target train data for validation
    y_val = np.hstack((y_source_val, y_target_train[:len(y_source_val)]))
    domain_val = np.hstack((domain_source_val, domain_target_train[:len(domain_source_val)]))
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Target test data shape: {X_target_test.shape}")
    
    # Create model
    input_dim = X_train.shape[1]
    model, feature_extractor, label_predictor, domain_classifier = create_domain_adaptation_model(input_dim)
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_label_predictor_accuracy',
        patience=20,
        restore_best_weights=True,
        mode='max'  # Explicitly set mode to maximize accuracy
    )
    
    model_checkpoint = ModelCheckpoint(
        'domain_adaptation_model.h5',
        monitor='val_label_predictor_accuracy',
        save_best_only=True,
        verbose=0
    )
    
    # Train model
    print("\nTraining domain adaptation model...")
    start_time = time.time()
    history = model.fit(
        X_train,
        {'label_predictor': y_train, 'domain_classifier': domain_train},
        validation_data=(
            X_val,
            {'label_predictor': y_val, 'domain_classifier': domain_val}
        ),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint],
        verbose=0
    )
    training_time = time.time() - start_time
    
    # Evaluate on target test data
    print("\nEvaluating on target test data...")
    start_time = time.time()
    y_pred_probs, _ = model.predict(X_target_test, verbose=0)
    inference_time = time.time() - start_time
    
    y_pred = np.argmax(y_pred_probs, axis=1)
    accuracy = accuracy_score(y_target_test, y_pred)
    f1 = f1_score(y_target_test, y_pred, average='weighted')
    
    print(f"Domain Adaptation Model Accuracy: {accuracy:.4f}")
    print(f"Domain Adaptation Model F1 Score: {f1:.4f}")
    print(f"Training Time: {training_time:.4f}s, Inference Time: {inference_time:.4f}s")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['label_predictor_loss'], label='Training Loss')
    plt.plot(history.history['val_label_predictor_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Label Predictor Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['label_predictor_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_label_predictor_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Label Predictor Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['domain_classifier_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_domain_classifier_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Domain Classifier Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('domain_adaptation_training_history.png')
    plt.close()
    
    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_target_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - Domain Adaptation Model')
    plt.tight_layout()
    plt.savefig('domain_adaptation_confusion_matrix.png')
    plt.close()
    
    return model, accuracy, f1, training_time, inference_time, history

def train_baseline_model(target_data):
    """Train a baseline model directly on target data for comparison"""
    # Unpack data
    X_target, y_target, _, _, _ = target_data
    
    # Split target data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_target, y_target, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_target
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Create model
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],), 
              kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Dropout(0.3),
        Dense(4, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        'baseline_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    )
    
    # Train model
    print("\nTraining baseline model...")
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint],
        verbose=0
    )
    training_time = time.time() - start_time
    
    # Evaluate on test data
    start_time = time.time()
    y_pred_probs = model.predict(X_test, verbose=0)
    inference_time = time.time() - start_time
    
    y_pred = np.argmax(y_pred_probs, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Baseline Model Accuracy: {accuracy:.4f}")
    print(f"Baseline Model F1 Score: {f1:.4f}")
    print(f"Training Time: {training_time:.4f}s, Inference Time: {inference_time:.4f}s")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Baseline Model - Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Baseline Model - Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('baseline_training_history.png')
    plt.close()
    
    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - Baseline Model')
    plt.tight_layout()
    plt.savefig('baseline_confusion_matrix.png')
    plt.close()
    
    return model, accuracy, f1, training_time, inference_time, history

def compare_models(baseline_results, domain_adaptation_results):
    """Compare baseline and domain adaptation models"""
    print("\nComparing model performance...")
    
    # Extract results
    baseline_accuracy = baseline_results['accuracy']
    baseline_f1 = baseline_results['f1']
    baseline_train_time = baseline_results['training_time']
    baseline_inference_time = baseline_results['inference_time']
    
    da_accuracy = domain_adaptation_results['accuracy']
    da_f1 = domain_adaptation_results['f1']
    da_train_time = domain_adaptation_results['training_time']
    da_inference_time = domain_adaptation_results['inference_time']
    
    # Create comparison plots
    plt.figure(figsize=(14, 10))
    
    # Accuracy comparison
    plt.subplot(2, 2, 1)
    models = ['Baseline', 'Domain Adaptation']
    accuracies = [baseline_accuracy, da_accuracy]
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
    
    # F1 score comparison
    plt.subplot(2, 2, 2)
    f1_scores = [baseline_f1, da_f1]
    bars = plt.bar(models, f1_scores, color=['blue', 'green'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=12)
    plt.ylabel('F1 Score')
    plt.title('Model F1 Score Comparison')
    plt.ylim(0, 1.0)
    
    # Training time comparison
    plt.subplot(2, 2, 3)
    train_times = [baseline_train_time, da_train_time]
    bars = plt.bar(models, train_times, color=['blue', 'green'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=12)
    plt.ylabel('Time (seconds)')
    plt.title('Training Time Comparison')
    
    # Inference time comparison
    plt.subplot(2, 2, 4)
    inference_times = [baseline_inference_time, da_inference_time]
    bars = plt.bar(models, inference_times, color=['blue', 'green'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}s', ha='center', va='bottom', fontsize=12)
    plt.ylabel('Time (seconds)')
    plt.title('Inference Time Comparison')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    # Calculate improvement
    accuracy_improvement = da_accuracy - baseline_accuracy
    f1_improvement = da_f1 - baseline_f1
    
    print(f"Accuracy improvement with domain adaptation: {accuracy_improvement:.4f} ({accuracy_improvement*100:.1f}%)")
    print(f"F1 score improvement with domain adaptation: {f1_improvement:.4f} ({f1_improvement*100:.1f}%)")
    
    return accuracy_improvement, f1_improvement

def main():
    """Main function to implement domain adaptation"""
    print("="*50)
    print("EEG DOMAIN ADAPTATION")
    print("="*50)
    
    # Load data from all sessions
    session_dfs = {}
    for session_id in SESSION_FILES.keys():
        df = load_session_data(session_id)
        if df is not None:
            session_dfs[session_id] = df
    
    # Extract features for each session
    session_data = {}
    for session_id, df in session_dfs.items():
        print(f"\nProcessing Session {session_id}...")
        X, y, sessions, pca, scaler = extract_features(df)
        session_data[session_id] = (X, y, sessions, pca, scaler)
    
    # Use Session 1 as source domain (best performing session)
    # Use Session 3 as target domain (to be improved)
    source_data = session_data[1]
    target_data = session_data[3]
    
    # Train baseline model on target data only
    baseline_model, baseline_accuracy, baseline_f1, baseline_train_time, baseline_inference_time, baseline_history = \
        train_baseline_model(target_data)
    
    # Train domain adaptation model
    da_model, da_accuracy, da_f1, da_train_time, da_inference_time, da_history = \
        train_domain_adaptation_model(source_data, target_data)
    
    # Compare models
    baseline_results = {
        'accuracy': baseline_accuracy,
        'f1': baseline_f1,
        'training_time': baseline_train_time,
        'inference_time': baseline_inference_time,
        'history': baseline_history
    }
    
    domain_adaptation_results = {
        'accuracy': da_accuracy,
        'f1': da_f1,
        'training_time': da_train_time,
        'inference_time': da_inference_time,
        'history': da_history
    }
    
    accuracy_improvement, f1_improvement = compare_models(baseline_results, domain_adaptation_results)
    
    # Save results
    results = {
        'baseline_model': baseline_results,
        'domain_adaptation_model': domain_adaptation_results,
        'accuracy_improvement': accuracy_improvement,
        'f1_improvement': f1_improvement
    }
    
    joblib.dump(results, 'domain_adaptation_results.pkl')
    
    # Print summary
    print("\n" + "="*50)
    print("DOMAIN ADAPTATION SUMMARY")
    print("="*50)
    print(f"Baseline Model Accuracy: {baseline_accuracy:.4f}")
    print(f"Domain Adaptation Model Accuracy: {da_accuracy:.4f}")
    print(f"Accuracy Improvement: {accuracy_improvement:.4f} ({accuracy_improvement*100:.1f}%)")
    print("="*50)
    
    print("Domain adaptation completed. Results and visualizations saved.")
    
    return results

if __name__ == "__main__":
    main()
