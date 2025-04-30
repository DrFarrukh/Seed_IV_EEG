import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, LSTM, GRU, SimpleRNN
from tensorflow.keras.layers import Bidirectional, Reshape, Flatten, TimeDistributed, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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
print(f"Number of unique channels: {df['channel'].nunique()}")
print(f"Number of unique epochs: {df['epoch'].nunique()}")

# Identify feature columns (excluding metadata)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
metadata_cols = ['epoch', 'channel', 'label', 'trial_id']
feature_cols = [col for col in numerical_cols if col not in metadata_cols]
print(f"Number of features: {len(feature_cols)}")

# Define important channels based on previous analysis
important_channels = [61, 58, 2, 31, 23, 4, 10, 15, 30, 40]
print(f"Using top {len(important_channels)} important channels")

# Extract features for PCA approach
def extract_pca_features(df, n_components=20):
    """
    Extract PCA features from important channels
    """
    # Get unique trials
    unique_trials = sorted(df['trial_id'].unique())
    n_trials = len(unique_trials)
    
    print(f"Extracting PCA features for {n_trials} trials")
    
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
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Explained variance with {n_components} components: {pca.explained_variance_ratio_.sum():.4f}")
    
    return X_pca, y_trials

# Extract features for recurrent approach (preserving temporal structure)
def extract_temporal_features(df):
    """
    Extract features preserving temporal structure for recurrent models
    """
    # Get unique trials and epochs
    unique_trials = sorted(df['trial_id'].unique())
    unique_epochs = sorted(df['epoch'].unique())
    n_trials = len(unique_trials)
    n_epochs = len(unique_epochs)
    
    print(f"Extracting temporal features for {n_trials} trials across {n_epochs} epochs")
    
    # Initialize arrays
    X_temporal = np.zeros((n_trials, n_epochs, len(important_channels) * 5))  # Using 5 key features per channel
    y_temporal = np.zeros(n_trials)
    
    # Extract features for each trial
    for i, trial in enumerate(unique_trials):
        # Get data for this trial
        trial_data = df[df['trial_id'] == trial]
        
        # Skip if trial doesn't have data
        if len(trial_data) == 0:
            continue
        
        # Get the label (should be the same for all rows in the trial)
        y_temporal[i] = trial_data['label'].iloc[0]
        
        # Filter for important channels only
        trial_data = trial_data[trial_data['channel'].isin(important_channels)]
        
        # Extract features for each epoch
        for j, epoch in enumerate(unique_epochs):
            epoch_data = trial_data[trial_data['epoch'] == epoch]
            
            # Skip if epoch doesn't have data
            if len(epoch_data) == 0:
                continue
            
            # Extract key features for each channel
            for k, channel in enumerate(important_channels):
                channel_data = epoch_data[epoch_data['channel'] == channel]
                
                # Skip if channel doesn't have data
                if len(channel_data) == 0:
                    continue
                
                # Extract 5 key features for this channel
                # 1. Mean of all features
                mean_features = channel_data[feature_cols].mean().mean()
                
                # 2-4. Power in key frequency bands (if available)
                delta = channel_data['delta_power'].mean() if 'delta_power' in feature_cols else 0
                theta = channel_data['theta_power'].mean() if 'theta_power' in feature_cols else 0
                alpha = channel_data['alpha_power'].mean() if 'alpha_power' in feature_cols else 0
                beta = channel_data['beta_power'].mean() if 'beta_power' in feature_cols else 0
                
                # 5. Standard deviation of features
                std_features = channel_data[feature_cols].std().mean()
                
                # Store features
                feature_idx = k * 5
                X_temporal[i, j, feature_idx] = mean_features
                X_temporal[i, j, feature_idx + 1] = delta
                X_temporal[i, j, feature_idx + 2] = theta
                X_temporal[i, j, feature_idx + 3] = alpha
                X_temporal[i, j, feature_idx + 4] = std_features
    
    # Standardize features
    # Reshape to 2D for standardization
    orig_shape = X_temporal.shape
    X_reshaped = X_temporal.reshape(-1, X_temporal.shape[2])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    # Reshape back to 3D
    X_temporal_scaled = X_scaled.reshape(orig_shape)
    
    return X_temporal_scaled, y_temporal.astype(int)

# Extract features for channel-sequence approach (preserving channel structure)
def extract_channel_sequence_features(df):
    """
    Extract features preserving channel structure for recurrent models
    treating channels as a sequence
    """
    # Get unique trials
    unique_trials = sorted(df['trial_id'].unique())
    n_trials = len(unique_trials)
    
    print(f"Extracting channel sequence features for {n_trials} trials")
    
    # Initialize arrays
    X_channel_seq = np.zeros((n_trials, len(important_channels), len(feature_cols)))
    y_channel_seq = np.zeros(n_trials)
    
    # Extract features for each trial
    for i, trial in enumerate(unique_trials):
        # Get data for this trial
        trial_data = df[df['trial_id'] == trial]
        
        # Skip if trial doesn't have data
        if len(trial_data) == 0:
            continue
        
        # Get the label (should be the same for all rows in the trial)
        y_channel_seq[i] = trial_data['label'].iloc[0]
        
        # Extract features for each channel
        for j, channel in enumerate(important_channels):
            channel_data = trial_data[trial_data['channel'] == channel]
            
            # Skip if channel doesn't have data
            if len(channel_data) == 0:
                continue
            
            # Average features across epochs for this channel
            X_channel_seq[i, j, :] = channel_data[feature_cols].mean().values
    
    # Standardize features
    # Reshape to 2D for standardization
    orig_shape = X_channel_seq.shape
    X_reshaped = X_channel_seq.reshape(-1, X_channel_seq.shape[2])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    # Reshape back to 3D
    X_channel_seq_scaled = X_scaled.reshape(orig_shape)
    
    return X_channel_seq_scaled, y_channel_seq.astype(int)

# Extract all feature sets
print("\nExtracting features for different models...")
X_pca, y_pca = extract_pca_features(df, n_components=20)
X_temporal, y_temporal = extract_temporal_features(df)
X_channel_seq, y_channel_seq = extract_channel_sequence_features(df)

print(f"PCA feature shape: {X_pca.shape}")
print(f"Temporal feature shape: {X_temporal.shape}")
print(f"Channel sequence feature shape: {X_channel_seq.shape}")

# Split data for all feature sets
print("\nSplitting data into training and testing sets...")
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y_pca, test_size=0.3, random_state=42, stratify=y_pca
)

X_train_temporal, X_test_temporal, y_train_temporal, y_test_temporal = train_test_split(
    X_temporal, y_temporal, test_size=0.3, random_state=42, stratify=y_temporal
)

X_train_channel, X_test_channel, y_train_channel, y_test_channel = train_test_split(
    X_channel_seq, y_channel_seq, test_size=0.3, random_state=42, stratify=y_channel_seq
)

# Convert to categorical for deep learning
y_train_cat = to_categorical(y_train_pca, num_classes=4)
y_test_cat = to_categorical(y_test_pca, num_classes=4)

# Create LSTM model for temporal features
def create_lstm_model(input_shape):
    """
    Create an LSTM model for temporal features
    """
    model = Sequential([
        # LSTM layers
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        
        # Dense layers
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Output layer
        Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create GRU model for channel sequence features
def create_gru_model(input_shape):
    """
    Create a GRU model for channel sequence features
    """
    model = Sequential([
        # GRU layers
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        
        GRU(32, return_sequences=False),
        Dropout(0.3),
        
        # Dense layers
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Output layer
        Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create RNN model for PCA features (reshaped to sequence)
def create_rnn_model(input_shape):
    """
    Create a simple RNN model for PCA features reshaped as a sequence
    """
    model = Sequential([
        # Reshape layer to convert PCA features to sequence
        Reshape((input_shape[0] // 4, 4), input_shape=(input_shape[0],)),
        
        # RNN layer
        SimpleRNN(32, return_sequences=False),
        Dropout(0.3),
        
        # Dense layers
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Output layer
        Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create traditional models
def create_traditional_models():
    """
    Create traditional machine learning models
    """
    # SVM
    svm = SVC(C=10, gamma=0.01, kernel='rbf', probability=True, random_state=42)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
    
    return svm, rf, gb

# Define callbacks for deep learning models
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.0001
)

# Train LSTM model
print("\nTraining LSTM model on temporal features...")
lstm_model = create_lstm_model((X_train_temporal.shape[1], X_train_temporal.shape[2]))
lstm_history = lstm_model.fit(
    X_train_temporal, to_categorical(y_train_temporal, num_classes=4),
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Train GRU model
print("\nTraining GRU model on channel sequence features...")
gru_model = create_gru_model((X_train_channel.shape[1], X_train_channel.shape[2]))
gru_history = gru_model.fit(
    X_train_channel, to_categorical(y_train_channel, num_classes=4),
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Train RNN model
print("\nTraining RNN model on PCA features...")
rnn_model = create_rnn_model((X_train_pca.shape[1],))
rnn_history = rnn_model.fit(
    X_train_pca, to_categorical(y_train_pca, num_classes=4),
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Train traditional models
print("\nTraining traditional models...")
svm, rf, gb = create_traditional_models()
svm.fit(X_train_pca, y_train_pca)
rf.fit(X_train_pca, y_train_pca)
gb.fit(X_train_pca, y_train_pca)

# Evaluate all models
print("\nEvaluating individual models...")

# Evaluate LSTM
lstm_preds = lstm_model.predict(X_test_temporal)
lstm_classes = np.argmax(lstm_preds, axis=1)
lstm_accuracy = accuracy_score(y_test_temporal, lstm_classes)
print(f"LSTM Model Accuracy: {lstm_accuracy:.4f}")

# Evaluate GRU
gru_preds = gru_model.predict(X_test_channel)
gru_classes = np.argmax(gru_preds, axis=1)
gru_accuracy = accuracy_score(y_test_channel, gru_classes)
print(f"GRU Model Accuracy: {gru_accuracy:.4f}")

# Evaluate RNN
rnn_preds = rnn_model.predict(X_test_pca)
rnn_classes = np.argmax(rnn_preds, axis=1)
rnn_accuracy = accuracy_score(y_test_pca, rnn_classes)
print(f"RNN Model Accuracy: {rnn_accuracy:.4f}")

# Evaluate traditional models
svm_preds = svm.predict(X_test_pca)
rf_preds = rf.predict(X_test_pca)
gb_preds = gb.predict(X_test_pca)

svm_accuracy = accuracy_score(y_test_pca, svm_preds)
rf_accuracy = accuracy_score(y_test_pca, rf_preds)
gb_accuracy = accuracy_score(y_test_pca, gb_preds)

print(f"SVM Accuracy: {svm_accuracy:.4f}")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")

# Create recurrent stacking ensemble
print("\nCreating recurrent stacking ensemble...")

# Function to create stacked predictions
def create_stacked_predictions(X_test_pca, X_test_temporal, X_test_channel):
    """
    Create stacked predictions from all models
    """
    # Get predictions from all models
    lstm_preds = lstm_model.predict(X_test_temporal)
    gru_preds = gru_model.predict(X_test_channel)
    rnn_preds = rnn_model.predict(X_test_pca)
    
    svm_preds_proba = svm.predict_proba(X_test_pca)
    rf_preds_proba = rf.predict_proba(X_test_pca)
    gb_preds_proba = gb.predict_proba(X_test_pca)
    
    # Stack all predictions
    stacked_preds = np.column_stack((
        lstm_preds, gru_preds, rnn_preds, 
        svm_preds_proba, rf_preds_proba, gb_preds_proba
    ))
    
    return stacked_preds

# Create a meta-learner model
def create_meta_learner(input_shape):
    """
    Create a meta-learner model that combines predictions from all models
    """
    model = Sequential([
        # Input layer
        Dense(64, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layer
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Output layer
        Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Generate stacked predictions for training
print("Generating stacked predictions for training...")
# We need to use cross-validation to avoid data leakage
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize arrays for stacked predictions
X_stacked_train = np.zeros((X_train_pca.shape[0], 24))  # 4 classes * 6 models
y_stacked_train = y_train_pca.copy()

# Generate out-of-fold predictions
for train_idx, val_idx in kf.split(X_train_pca, y_train_pca):
    # Split data
    X_train_pca_fold, X_val_pca_fold = X_train_pca[train_idx], X_train_pca[val_idx]
    X_train_temporal_fold, X_val_temporal_fold = X_train_temporal[train_idx], X_train_temporal[val_idx]
    X_train_channel_fold, X_val_channel_fold = X_train_channel[train_idx], X_train_channel[val_idx]
    y_train_fold, y_val_fold = y_train_pca[train_idx], y_train_pca[val_idx]
    
    # Train models on fold
    # LSTM
    lstm_fold = create_lstm_model((X_train_temporal_fold.shape[1], X_train_temporal_fold.shape[2]))
    lstm_fold.fit(
        X_train_temporal_fold, to_categorical(y_train_fold, num_classes=4),
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    # GRU
    gru_fold = create_gru_model((X_train_channel_fold.shape[1], X_train_channel_fold.shape[2]))
    gru_fold.fit(
        X_train_channel_fold, to_categorical(y_train_fold, num_classes=4),
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    # RNN
    rnn_fold = create_rnn_model((X_train_pca_fold.shape[1],))
    rnn_fold.fit(
        X_train_pca_fold, to_categorical(y_train_fold, num_classes=4),
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    # Traditional models
    svm_fold = SVC(C=10, gamma=0.01, kernel='rbf', probability=True, random_state=42)
    rf_fold = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
    gb_fold = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
    
    svm_fold.fit(X_train_pca_fold, y_train_fold)
    rf_fold.fit(X_train_pca_fold, y_train_fold)
    gb_fold.fit(X_train_pca_fold, y_train_fold)
    
    # Generate predictions on validation fold
    lstm_val_preds = lstm_fold.predict(X_val_temporal_fold)
    gru_val_preds = gru_fold.predict(X_val_channel_fold)
    rnn_val_preds = rnn_fold.predict(X_val_pca_fold)
    
    svm_val_preds = svm_fold.predict_proba(X_val_pca_fold)
    rf_val_preds = rf_fold.predict_proba(X_val_pca_fold)
    gb_val_preds = gb_fold.predict_proba(X_val_pca_fold)
    
    # Stack predictions
    stacked_val_preds = np.column_stack((
        lstm_val_preds, gru_val_preds, rnn_val_preds,
        svm_val_preds, rf_val_preds, gb_val_preds
    ))
    
    # Store in stacked predictions array
    X_stacked_train[val_idx] = stacked_val_preds

# Generate stacked predictions for testing
print("Generating stacked predictions for testing...")
X_stacked_test = create_stacked_predictions(X_test_pca, X_test_temporal, X_test_channel)

# Train meta-learner
print("Training meta-learner...")
meta_learner = create_meta_learner(X_stacked_train.shape[1])
meta_history = meta_learner.fit(
    X_stacked_train, to_categorical(y_stacked_train, num_classes=4),
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate meta-learner
meta_preds = meta_learner.predict(X_stacked_test)
meta_classes = np.argmax(meta_preds, axis=1)
meta_accuracy = accuracy_score(y_test_pca, meta_classes)
print(f"Meta-Learner Accuracy: {meta_accuracy:.4f}")

# Print classification report
print("\nClassification Report (Meta-Learner):")
print(classification_report(y_test_pca, meta_classes))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_pca, meta_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Recurrent Stacking Ensemble')
plt.tight_layout()
plt.savefig('recurrent_stacking_confusion_matrix.png')
plt.close()

# Plot learning curves for recurrent models
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(lstm_history.history['accuracy'], label='Train')
plt.plot(lstm_history.history['val_accuracy'], label='Validation')
plt.title('LSTM Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(gru_history.history['accuracy'], label='Train')
plt.plot(gru_history.history['val_accuracy'], label='Validation')
plt.title('GRU Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(rnn_history.history['accuracy'], label='Train')
plt.plot(rnn_history.history['val_accuracy'], label='Validation')
plt.title('RNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('recurrent_learning_curves.png')
plt.close()

# Plot meta-learner learning curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(meta_history.history['accuracy'], label='Train')
plt.plot(meta_history.history['val_accuracy'], label='Validation')
plt.title('Meta-Learner Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(meta_history.history['loss'], label='Train')
plt.plot(meta_history.history['val_loss'], label='Validation')
plt.title('Meta-Learner Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('meta_learner_curves.png')
plt.close()

# Compare with previous approaches
print("\nComparing with previous approaches:")
print(f"PCA + Ensemble (20 components): 68.06%")
print(f"Optimized Stacking Ensemble: 72.22%")
print(f"Deep PCA + Weighted Ensemble: 72.22%")
print(f"LSTM Model: {lstm_accuracy:.4f}")
print(f"GRU Model: {gru_accuracy:.4f}")
print(f"RNN Model: {rnn_accuracy:.4f}")
print(f"Recurrent Stacking Ensemble: {meta_accuracy:.4f}")

# Analyze per-class performance
class_report = classification_report(y_test_pca, meta_classes, output_dict=True)
class_f1 = [class_report[str(i)]['f1-score'] for i in range(4)]

plt.figure(figsize=(10, 6))
plt.bar(['Class 0', 'Class 1', 'Class 2', 'Class 3'], class_f1, color='skyblue')
plt.axhline(y=0.5, color='r', linestyle='--', label='Baseline')
plt.xlabel('Class')
plt.ylabel('F1-Score')
plt.title('Per-Class Performance - Recurrent Stacking Ensemble')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig('recurrent_stacking_per_class_performance.png')
plt.close()

print("\nRecurrent stacking ensemble analysis completed. Visualization images saved.")
