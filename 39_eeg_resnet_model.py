"""
EEG ResNet Model

This script implements a ResNet-based deep learning model for EEG classification.
It uses residual connections to enable deeper network architectures and better
feature learning for EEG data.

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
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Activation, Add
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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

def extract_features(df, n_components=OPTIMAL_PCA_COMPONENTS, reshape_for_cnn=False):
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
    
    # Reshape for CNN if requested
    if reshape_for_cnn:
        # Reshape to (samples, timesteps, features) format for CNN
        # We'll treat PCA components as time steps for 1D convolution
        X_pca = X_pca.reshape(X_pca.shape[0], X_pca.shape[1], 1)
        print(f"Reshaped for CNN: {X_pca.shape}")
    
    return X_pca, y_trials, sessions, pca, scaler

def residual_block(x, filters, kernel_size=3, strides=1, activation='relu'):
    """Create a residual block for ResNet"""
    # Shortcut connection
    shortcut = x
    
    # First convolution layer
    x = Conv1D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    # Second convolution layer
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    # If dimensions don't match, adjust shortcut
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Add shortcut to output
    x = Add()([x, shortcut])
    x = Activation(activation)(x)
    
    return x

def create_resnet_model(input_shape, num_classes=4):
    """Create a ResNet model for EEG classification"""
    inputs = Input(shape=input_shape)
    
    # Initial convolution
    x = Conv1D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    
    # Residual blocks
    # First block group - 64 filters
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    # Second block group - 128 filters
    x = residual_block(x, 128, strides=2)
    x = residual_block(x, 128)
    
    # Third block group - 256 filters
    x = residual_block(x, 256, strides=2)
    x = residual_block(x, 256)
    
    # Global pooling and classification
    x = GlobalAveragePooling1D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_resnet_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train ResNet model for EEG classification"""
    # Create model
    input_shape = X_train.shape[1:]
    model = create_resnet_model(input_shape)
    
    # Print model summary
    model.summary()
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        mode='max'
    )
    
    model_checkpoint = ModelCheckpoint(
        'resnet_model.h5',
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
    print("\nTraining ResNet model...")
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
    
    # Evaluate on test data
    print("\nEvaluating on test data...")
    start_time = time.time()
    y_pred_probs = model.predict(X_test, verbose=0)
    inference_time = time.time() - start_time
    
    y_pred = np.argmax(y_pred_probs, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"ResNet Model Accuracy: {accuracy:.4f}")
    print(f"ResNet Model F1 Score: {f1:.4f}")
    print(f"Training Time: {training_time:.4f}s, Inference Time: {inference_time:.4f}s")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('ResNet Model - Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('ResNet Model - Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('resnet_training_history.png')
    plt.close()
    
    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - ResNet Model')
    plt.tight_layout()
    plt.savefig('resnet_confusion_matrix.png')
    plt.close()
    
    # Save results
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'training_time': training_time,
        'inference_time': inference_time,
        'history': history.history,
        'confusion_matrix': cm.tolist()
    }
    
    joblib.dump(results, 'resnet_results.pkl')
    
    return model, results

def create_deeper_resnet_model(input_shape, num_classes=4):
    """Create a deeper ResNet model for EEG classification (ResNet-34 style)"""
    inputs = Input(shape=input_shape)
    
    # Initial convolution
    x = Conv1D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    
    # Residual blocks
    # First block group - 64 filters (3 blocks)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    # Second block group - 128 filters (4 blocks)
    x = residual_block(x, 128, strides=2)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    
    # Third block group - 256 filters (6 blocks)
    x = residual_block(x, 256, strides=2)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    
    # Fourth block group - 512 filters (3 blocks)
    x = residual_block(x, 512, strides=2)
    x = residual_block(x, 512)
    x = residual_block(x, 512)
    
    # Global pooling and classification
    x = GlobalAveragePooling1D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Lower learning rate for deeper network
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_deeper_resnet_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train deeper ResNet model for EEG classification"""
    # Create model
    input_shape = X_train.shape[1:]
    model = create_deeper_resnet_model(input_shape)
    
    # Print model summary
    model.summary()
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=25,  # More patience for deeper network
        restore_best_weights=True,
        mode='max'
    )
    
    model_checkpoint = ModelCheckpoint(
        'deeper_resnet_model.h5',
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
    print("\nTraining Deeper ResNet model...")
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,  # More epochs for deeper network
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )
    training_time = time.time() - start_time
    
    # Evaluate on test data
    print("\nEvaluating on test data...")
    start_time = time.time()
    y_pred_probs = model.predict(X_test, verbose=0)
    inference_time = time.time() - start_time
    
    y_pred = np.argmax(y_pred_probs, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Deeper ResNet Model Accuracy: {accuracy:.4f}")
    print(f"Deeper ResNet Model F1 Score: {f1:.4f}")
    print(f"Training Time: {training_time:.4f}s, Inference Time: {inference_time:.4f}s")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Deeper ResNet Model - Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Deeper ResNet Model - Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('deeper_resnet_training_history.png')
    plt.close()
    
    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - Deeper ResNet Model')
    plt.tight_layout()
    plt.savefig('deeper_resnet_confusion_matrix.png')
    plt.close()
    
    # Save results
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'training_time': training_time,
        'inference_time': inference_time,
        'history': history.history,
        'confusion_matrix': cm.tolist()
    }
    
    joblib.dump(results, 'deeper_resnet_results.pkl')
    
    return model, results

def compare_models(resnet_results, deeper_resnet_results):
    """Compare standard ResNet and deeper ResNet models"""
    print("\nComparing model performance...")
    
    # Extract results
    resnet_accuracy = resnet_results['accuracy']
    resnet_f1 = resnet_results['f1_score']
    resnet_train_time = resnet_results['training_time']
    resnet_inference_time = resnet_results['inference_time']
    
    deeper_accuracy = deeper_resnet_results['accuracy']
    deeper_f1 = deeper_resnet_results['f1_score']
    deeper_train_time = deeper_resnet_results['training_time']
    deeper_inference_time = deeper_resnet_results['inference_time']
    
    # Create comparison plots
    plt.figure(figsize=(14, 10))
    
    # Accuracy comparison
    plt.subplot(2, 2, 1)
    models = ['ResNet', 'Deeper ResNet']
    accuracies = [resnet_accuracy, deeper_accuracy]
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
    f1_scores = [resnet_f1, deeper_f1]
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
    train_times = [resnet_train_time, deeper_train_time]
    bars = plt.bar(models, train_times, color=['blue', 'green'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=12)
    plt.ylabel('Time (seconds)')
    plt.title('Training Time Comparison')
    
    # Inference time comparison
    plt.subplot(2, 2, 4)
    inference_times = [resnet_inference_time, deeper_inference_time]
    bars = plt.bar(models, inference_times, color=['blue', 'green'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}s', ha='center', va='bottom', fontsize=12)
    plt.ylabel('Time (seconds)')
    plt.title('Inference Time Comparison')
    
    plt.tight_layout()
    plt.savefig('resnet_model_comparison.png')
    plt.close()
    
    # Calculate improvement
    accuracy_improvement = deeper_accuracy - resnet_accuracy
    f1_improvement = deeper_f1 - resnet_f1
    
    print(f"Accuracy improvement with deeper ResNet: {accuracy_improvement:.4f} ({accuracy_improvement*100:.1f}%)")
    print(f"F1 score improvement with deeper ResNet: {f1_improvement:.4f} ({f1_improvement*100:.1f}%)")
    
    # Save comparison results
    comparison_results = {
        'resnet': {
            'accuracy': resnet_accuracy,
            'f1_score': resnet_f1,
            'training_time': resnet_train_time,
            'inference_time': resnet_inference_time
        },
        'deeper_resnet': {
            'accuracy': deeper_accuracy,
            'f1_score': deeper_f1,
            'training_time': deeper_train_time,
            'inference_time': deeper_inference_time
        },
        'accuracy_improvement': accuracy_improvement,
        'f1_improvement': f1_improvement
    }
    
    joblib.dump(comparison_results, 'resnet_comparison_results.pkl')
    
    return comparison_results

def main():
    """Main function to implement ResNet models for EEG classification"""
    print("="*50)
    print("EEG RESNET MODEL")
    print("="*50)
    
    # Load data from all sessions
    df = load_session_data()
    
    # Extract features with reshaping for CNN
    X, y, sessions, pca, scaler = extract_features(df, reshape_for_cnn=True)
    
    # Split data into train, validation, and test sets
    # First split into train+val and test
    X_train_val, X_test, y_train_val, y_test, sessions_train_val, sessions_test = train_test_split(
        X, y, sessions, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Then split train+val into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_val
    )
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Testing set: {X_test.shape}")
    
    # Train standard ResNet model
    resnet_model, resnet_results = train_resnet_model(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Train deeper ResNet model
    deeper_resnet_model, deeper_resnet_results = train_deeper_resnet_model(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Compare models
    comparison_results = compare_models(resnet_results, deeper_resnet_results)
    
    # Print summary
    print("\n" + "="*50)
    print("RESNET MODEL SUMMARY")
    print("="*50)
    print(f"Standard ResNet Accuracy: {resnet_results['accuracy']:.4f}")
    print(f"Deeper ResNet Accuracy: {deeper_resnet_results['accuracy']:.4f}")
    print(f"Accuracy Improvement: {comparison_results['accuracy_improvement']:.4f} ({comparison_results['accuracy_improvement']*100:.1f}%)")
    print("="*50)
    
    print("ResNet models training completed. Results and visualizations saved.")
    
    return comparison_results

if __name__ == "__main__":
    main()
