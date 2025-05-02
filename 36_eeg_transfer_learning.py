"""
EEG Transfer Learning

This script implements transfer learning for EEG classification by:
1. Loading a pre-trained EEGNet model
2. Fine-tuning it on the combined EEG dataset
3. Comparing performance with and without transfer learning

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
from tensorflow.keras.layers import Dense, Dropout, Conv2D, AveragePooling2D, SeparableConv2D
from tensorflow.keras.layers import BatchNormalization, Activation, Flatten, Input, DepthwiseConv2D
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
N_CLASSES = 4
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

def extract_features(df, apply_pca=True, n_components=OPTIMAL_PCA_COMPONENTS):
    """Extract features using important channels and optionally apply PCA"""
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
    
    if apply_pca:
        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=n_components)
        X_features = pca.fit_transform(X_scaled)
        print(f"Explained variance with {n_components} components: {pca.explained_variance_ratio_.sum():.4f}")
        print(f"PCA feature shape: {X_features.shape}")
    else:
        X_features = X_scaled
        pca = None
        print(f"Standardized feature shape: {X_features.shape}")
    
    print(f"Class distribution: {np.bincount(y_trials)}")
    
    return X_features, y_trials, trial_ids, sessions, pca, scaler

def create_eegnet_model(nb_classes=N_CLASSES, Chans=10, Samples=49, 
                      dropoutRate=0.5, kernLength=32, F1=8, 
                      D=2, F2=16, norm_rate=0.25):
    """
    Create EEGNet model architecture
    Based on the paper: EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs
    """
    # Reshape input for EEGNet
    input_shape = (Chans, Samples, 1)
    
    # Start the model
    input_layer = Input(shape=input_shape)
    
    # Block 1: Temporal Convolution
    block1 = Conv2D(F1, (1, kernLength), padding='same', input_shape=input_shape, use_bias=False)(input_layer)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D, 
                            depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)
    
    # Block 2: Separable Convolution
    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropoutRate)(block2)
    
    # Classification block
    flatten = Flatten(name='flatten')(block2)
    dense = Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)
    
    return Model(inputs=input_layer, outputs=softmax)

def reshape_for_eegnet(X, y, n_channels=10, n_features_per_channel=49):
    """Reshape data for EEGNet input"""
    # Reshape X from (n_samples, n_features) to (n_samples, n_channels, n_features_per_channel, 1)
    X_reshaped = X.reshape(-1, n_channels, n_features_per_channel, 1)
    return X_reshaped, y

def train_base_model(X_train, y_train, X_test, y_test):
    """Train a base EEGNet model without transfer learning"""
    print("\nTraining base EEGNet model (no transfer learning)...")
    
    # Reshape data for EEGNet
    X_train_reshaped, y_train = reshape_for_eegnet(X_train, y_train)
    X_test_reshaped, y_test = reshape_for_eegnet(X_test, y_test)
    
    # Create model
    model = create_eegnet_model()
    
    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        'base_eegnet_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    )
    
    # Train model
    start_time = time.time()
    history = model.fit(
        X_train_reshaped, y_train,
        batch_size=32,
        epochs=100,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint],
        verbose=0
    )
    training_time = time.time() - start_time
    
    # Evaluate model
    start_time = time.time()
    y_pred_probs = model.predict(X_test_reshaped, verbose=0)
    inference_time = time.time() - start_time
    
    y_pred = np.argmax(y_pred_probs, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Base EEGNet Accuracy: {accuracy:.4f}")
    print(f"Base EEGNet F1 Score: {f1:.4f}")
    print(f"Training Time: {training_time:.4f}s, Inference Time: {inference_time:.4f}s")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Base EEGNet - Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Base EEGNet - Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('base_eegnet_training_history.png')
    plt.close()
    
    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - Base EEGNet Model')
    plt.tight_layout()
    plt.savefig('base_eegnet_confusion_matrix.png')
    plt.close()
    
    return model, accuracy, f1, training_time, inference_time, history

def pretrain_on_session(session_id):
    """Pretrain EEGNet model on a specific session"""
    print(f"\nPretraining EEGNet model on Session {session_id}...")
    
    # Load session data
    df = load_session_data(session_id)
    
    # Extract features
    X, y, _, _, _, _ = extract_features(df, apply_pca=False)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Reshape data for EEGNet
    X_train_reshaped, y_train = reshape_for_eegnet(X_train, y_train)
    X_test_reshaped, y_test = reshape_for_eegnet(X_test, y_test)
    
    # Create model
    model = create_eegnet_model()
    
    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        f'pretrained_eegnet_session{session_id}.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    )
    
    # Train model
    history = model.fit(
        X_train_reshaped, y_train,
        batch_size=32,
        epochs=100,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint],
        verbose=0
    )
    
    # Evaluate model
    y_pred_probs = model.predict(X_test_reshaped, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Pretrained Model Accuracy on Session {session_id}: {accuracy:.4f}")
    
    return model, accuracy, history

def fine_tune_model(pretrained_model, X_train, y_train, X_test, y_test, freeze_layers=True):
    """Fine-tune a pretrained EEGNet model on the combined dataset"""
    print("\nFine-tuning pretrained EEGNet model...")
    
    # Reshape data for EEGNet
    X_train_reshaped, y_train = reshape_for_eegnet(X_train, y_train)
    X_test_reshaped, y_test = reshape_for_eegnet(X_test, y_test)
    
    # Create a new model with the same architecture
    fine_tuned_model = create_eegnet_model()
    
    # Copy weights from pretrained model
    fine_tuned_model.set_weights(pretrained_model.get_weights())
    
    # Freeze layers if specified
    if freeze_layers:
        # Freeze all layers except the last two (dense and softmax)
        for layer in fine_tuned_model.layers[:-2]:
            layer.trainable = False
        
        print("Freezing early layers, only fine-tuning classification layers")
    else:
        print("Fine-tuning all layers")
    
    # Compile model with a lower learning rate for fine-tuning
    fine_tuned_model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
        metrics=['accuracy']
    )
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=30,  # More patience for fine-tuning
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        'fine_tuned_eegnet_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    )
    
    # Train model
    start_time = time.time()
    history = fine_tuned_model.fit(
        X_train_reshaped, y_train,
        batch_size=32,
        epochs=150,  # More epochs for fine-tuning
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint],
        verbose=0
    )
    training_time = time.time() - start_time
    
    # Evaluate model
    start_time = time.time()
    y_pred_probs = fine_tuned_model.predict(X_test_reshaped, verbose=0)
    inference_time = time.time() - start_time
    
    y_pred = np.argmax(y_pred_probs, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Fine-tuned EEGNet Accuracy: {accuracy:.4f}")
    print(f"Fine-tuned EEGNet F1 Score: {f1:.4f}")
    print(f"Training Time: {training_time:.4f}s, Inference Time: {inference_time:.4f}s")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Fine-tuned EEGNet - Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Fine-tuned EEGNet - Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('fine_tuned_eegnet_training_history.png')
    plt.close()
    
    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - Fine-tuned EEGNet Model')
    plt.tight_layout()
    plt.savefig('fine_tuned_eegnet_confusion_matrix.png')
    plt.close()
    
    return fine_tuned_model, accuracy, f1, training_time, inference_time, history

def compare_models(base_results, fine_tuned_results):
    """Compare base and fine-tuned models"""
    print("\nComparing model performance...")
    
    # Extract results
    base_accuracy = base_results['accuracy']
    base_f1 = base_results['f1']
    base_train_time = base_results['training_time']
    base_inference_time = base_results['inference_time']
    
    fine_tuned_accuracy = fine_tuned_results['accuracy']
    fine_tuned_f1 = fine_tuned_results['f1']
    fine_tuned_train_time = fine_tuned_results['training_time']
    fine_tuned_inference_time = fine_tuned_results['inference_time']
    
    # Create comparison plots
    plt.figure(figsize=(14, 10))
    
    # Accuracy comparison
    plt.subplot(2, 2, 1)
    models = ['Base EEGNet', 'Fine-tuned EEGNet']
    accuracies = [base_accuracy, fine_tuned_accuracy]
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
    f1_scores = [base_f1, fine_tuned_f1]
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
    train_times = [base_train_time, fine_tuned_train_time]
    bars = plt.bar(models, train_times, color=['blue', 'green'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=12)
    plt.ylabel('Time (seconds)')
    plt.title('Training Time Comparison')
    
    # Inference time comparison
    plt.subplot(2, 2, 4)
    inference_times = [base_inference_time, fine_tuned_inference_time]
    bars = plt.bar(models, inference_times, color=['blue', 'green'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}s', ha='center', va='bottom', fontsize=12)
    plt.ylabel('Time (seconds)')
    plt.title('Inference Time Comparison')
    
    plt.tight_layout()
    plt.savefig('eegnet_model_comparison.png')
    plt.close()
    
    # Calculate improvement
    accuracy_improvement = fine_tuned_accuracy - base_accuracy
    f1_improvement = fine_tuned_f1 - base_f1
    
    print(f"Accuracy improvement with transfer learning: {accuracy_improvement:.4f} ({accuracy_improvement*100:.1f}%)")
    print(f"F1 score improvement with transfer learning: {f1_improvement:.4f} ({f1_improvement*100:.1f}%)")
    
    return accuracy_improvement, f1_improvement

def main():
    """Main function to implement transfer learning"""
    print("="*50)
    print("EEG TRANSFER LEARNING")
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
    
    # Extract features without PCA for EEGNet
    X, y, trial_ids, sessions, _, _ = extract_features(combined_df, apply_pca=False)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
    
    # Train base EEGNet model (no transfer learning)
    base_model, base_accuracy, base_f1, base_train_time, base_inference_time, base_history = \
        train_base_model(X_train, y_train, X_test, y_test)
    
    # Pretrain on Session 1 (which had the best performance in previous analyses)
    pretrained_model, pretrain_accuracy, pretrain_history = pretrain_on_session(1)
    
    # Fine-tune pretrained model on combined dataset
    fine_tuned_model, fine_tuned_accuracy, fine_tuned_f1, fine_tuned_train_time, fine_tuned_inference_time, fine_tuned_history = \
        fine_tune_model(pretrained_model, X_train, y_train, X_test, y_test, freeze_layers=True)
    
    # Compare models
    base_results = {
        'accuracy': base_accuracy,
        'f1': base_f1,
        'training_time': base_train_time,
        'inference_time': base_inference_time,
        'history': base_history
    }
    
    fine_tuned_results = {
        'accuracy': fine_tuned_accuracy,
        'f1': fine_tuned_f1,
        'training_time': fine_tuned_train_time,
        'inference_time': fine_tuned_inference_time,
        'history': fine_tuned_history
    }
    
    accuracy_improvement, f1_improvement = compare_models(base_results, fine_tuned_results)
    
    # Save results
    results = {
        'base_model': base_results,
        'fine_tuned_model': fine_tuned_results,
        'pretrain_accuracy': pretrain_accuracy,
        'accuracy_improvement': accuracy_improvement,
        'f1_improvement': f1_improvement
    }
    
    joblib.dump(results, 'transfer_learning_results.pkl')
    
    # Print summary
    print("\n" + "="*50)
    print("TRANSFER LEARNING SUMMARY")
    print("="*50)
    print(f"Base EEGNet Accuracy: {base_accuracy:.4f}")
    print(f"Pretrained Model Accuracy (Session 1): {pretrain_accuracy:.4f}")
    print(f"Fine-tuned EEGNet Accuracy: {fine_tuned_accuracy:.4f}")
    print(f"Accuracy Improvement: {accuracy_improvement:.4f} ({accuracy_improvement*100:.1f}%)")
    print("="*50)
    
    print("Transfer learning completed. Results and visualizations saved.")
    
    return results

if __name__ == "__main__":
    main()
