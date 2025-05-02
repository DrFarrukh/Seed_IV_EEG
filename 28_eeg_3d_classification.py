import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Reshape, Conv1D, MaxPooling1D
from tensorflow.keras.layers import LSTM, Bidirectional, TimeDistributed, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

# Function to reshape data into 3D format (trials × channels × features)
def reshape_to_3d(df, feature_cols, max_channels=None, max_epochs=None):
    """
    Reshape data into 3D format for deep learning models
    Returns: X (trials × channels × features), y (trial labels)
    """
    # Get unique trials, channels, and epochs
    unique_trials = sorted(df['trial_id'].unique())
    unique_channels = sorted(df['channel'].unique())
    unique_epochs = sorted(df['epoch'].unique())
    
    # Limit channels and epochs if specified
    if max_channels is not None:
        unique_channels = unique_channels[:max_channels]
    if max_epochs is not None:
        unique_epochs = unique_epochs[:max_epochs]
    
    n_trials = len(unique_trials)
    n_channels = len(unique_channels)
    n_epochs = len(unique_epochs)
    n_features = len(feature_cols)
    
    print(f"Reshaping to 3D: {n_trials} trials × {n_channels} channels × {n_epochs} epochs × {n_features} features")
    
    # Initialize arrays
    X = np.zeros((n_trials, n_channels, n_epochs, n_features))
    y = np.zeros(n_trials)
    
    # Fill arrays
    for i, trial in enumerate(unique_trials):
        trial_data = df[df['trial_id'] == trial]
        
        # Get label (should be consistent within trial)
        if len(trial_data) > 0:
            y[i] = trial_data['label'].iloc[0]
        
        # Fill feature data
        for j, channel in enumerate(unique_channels):
            channel_data = trial_data[trial_data['channel'] == channel]
            
            for k, epoch in enumerate(unique_epochs):
                epoch_data = channel_data[channel_data['epoch'] == epoch]
                
                if len(epoch_data) > 0:
                    X[i, j, k, :] = epoch_data[feature_cols].values[0]
    
    return X, y.astype(int)

# Function to reshape data into channel-first format (trials × channels × features)
def reshape_to_channel_first(df, feature_cols):
    """
    Reshape data into channel-first format, averaging across epochs
    Returns: X (trials × channels × features), y (trial labels)
    """
    # Get unique trials and channels
    unique_trials = sorted(df['trial_id'].unique())
    unique_channels = sorted(df['channel'].unique())
    
    n_trials = len(unique_trials)
    n_channels = len(unique_channels)
    n_features = len(feature_cols)
    
    print(f"Reshaping to channel-first: {n_trials} trials × {n_channels} channels × {n_features} features")
    
    # Initialize arrays
    X = np.zeros((n_trials, n_channels, n_features))
    y = np.zeros(n_trials)
    
    # Fill arrays
    for i, trial in enumerate(unique_trials):
        trial_data = df[df['trial_id'] == trial]
        
        # Get label (should be consistent within trial)
        if len(trial_data) > 0:
            y[i] = trial_data['label'].iloc[0]
        
        # Fill feature data
        for j, channel in enumerate(unique_channels):
            channel_data = trial_data[trial_data['channel'] == channel]
            
            if len(channel_data) > 0:
                # Average across epochs
                X[i, j, :] = channel_data[feature_cols].mean().values
    
    return X, y.astype(int)

# Function to reshape data into epoch-first format (trials × epochs × features)
def reshape_to_epoch_first(df, feature_cols):
    """
    Reshape data into epoch-first format, averaging across channels
    Returns: X (trials × epochs × features), y (trial labels)
    """
    # Get unique trials and epochs
    unique_trials = sorted(df['trial_id'].unique())
    unique_epochs = sorted(df['epoch'].unique())
    
    n_trials = len(unique_trials)
    n_epochs = len(unique_epochs)
    n_features = len(feature_cols)
    
    print(f"Reshaping to epoch-first: {n_trials} trials × {n_epochs} epochs × {n_features} features")
    
    # Initialize arrays
    X = np.zeros((n_trials, n_epochs, n_features))
    y = np.zeros(n_trials)
    
    # Fill arrays
    for i, trial in enumerate(unique_trials):
        trial_data = df[df['trial_id'] == trial]
        
        # Get label (should be consistent within trial)
        if len(trial_data) > 0:
            y[i] = trial_data['label'].iloc[0]
        
        # Fill feature data
        for k, epoch in enumerate(unique_epochs):
            epoch_data = trial_data[trial_data['epoch'] == epoch]
            
            if len(epoch_data) > 0:
                # Average across channels
                X[i, k, :] = epoch_data[feature_cols].mean().values
    
    return X, y.astype(int)

# Try different 3D representations
print("\nCreating different 3D representations of the data...")

# 1. Channel-first representation (averaging across epochs)
X_channel, y_channel = reshape_to_channel_first(df, feature_cols)

# 2. Epoch-first representation (averaging across channels)
X_epoch, y_epoch = reshape_to_epoch_first(df, feature_cols)

# 3. Full 3D representation (limited to first 20 epochs for manageability)
X_3d, y_3d = reshape_to_3d(df, feature_cols, max_epochs=20)

# Standardize features
print("\nStandardizing features...")
# For channel-first data
for i in range(X_channel.shape[0]):
    for j in range(X_channel.shape[1]):
        X_channel[i, j, :] = (X_channel[i, j, :] - np.mean(X_channel[i, j, :])) / (np.std(X_channel[i, j, :]) + 1e-10)

# For epoch-first data
for i in range(X_epoch.shape[0]):
    for j in range(X_epoch.shape[1]):
        X_epoch[i, j, :] = (X_epoch[i, j, :] - np.mean(X_epoch[i, j, :])) / (np.std(X_epoch[i, j, :]) + 1e-10)

# For full 3D data
for i in range(X_3d.shape[0]):
    for j in range(X_3d.shape[1]):
        for k in range(X_3d.shape[2]):
            X_3d[i, j, k, :] = (X_3d[i, j, k, :] - np.mean(X_3d[i, j, k, :])) / (np.std(X_3d[i, j, k, :]) + 1e-10)

# Convert labels to categorical
n_classes = len(np.unique(y_channel))
y_channel_cat = to_categorical(y_channel, n_classes)
y_epoch_cat = to_categorical(y_epoch, n_classes)
y_3d_cat = to_categorical(y_3d, n_classes)

# Split data into training and testing sets
print("\nSplitting data into training and testing sets...")
# Channel-first data
X_channel_train, X_channel_test, y_channel_train, y_channel_test = train_test_split(
    X_channel, y_channel_cat, test_size=0.3, random_state=42, stratify=y_channel
)

# Epoch-first data
X_epoch_train, X_epoch_test, y_epoch_train, y_epoch_test = train_test_split(
    X_epoch, y_epoch_cat, test_size=0.3, random_state=42, stratify=y_epoch
)

# Full 3D data
X_3d_train, X_3d_test, y_3d_train, y_3d_test = train_test_split(
    X_3d, y_3d_cat, test_size=0.3, random_state=42, stratify=y_3d
)

# Define models for different data representations
print("\nDefining models for different data representations...")

# 1. CNN for channel-first data
def create_channel_cnn(input_shape, n_classes):
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        
        Conv1D(64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(n_classes, activation='softmax')
    ])
    return model

# 2. LSTM for epoch-first data
def create_epoch_lstm(input_shape, n_classes):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.25),
        Bidirectional(LSTM(32)),
        Dropout(0.25),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(n_classes, activation='softmax')
    ])
    return model

# 3. EEGNet-inspired model for full 3D data
def create_eegnet(input_shape, n_classes):
    # Reshape to (channels, epochs, features)
    input_layer = Input(shape=input_shape)
    reshaped = Reshape((input_shape[0], input_shape[1] * input_shape[2]))(input_layer)
    
    # First convolutional block
    conv1 = Conv1D(16, kernel_size=3, padding='same', activation='relu')(reshaped)
    batch1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=2)(batch1)
    drop1 = Dropout(0.25)(pool1)
    
    # Second convolutional block
    conv2 = Conv1D(32, kernel_size=3, padding='same', activation='relu')(drop1)
    batch2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=2)(batch2)
    drop2 = Dropout(0.25)(pool2)
    
    # Flatten and dense layers
    flat = Flatten()(drop2)
    dense1 = Dense(64, activation='relu')(flat)
    batch3 = BatchNormalization()(dense1)
    drop3 = Dropout(0.5)(batch3)
    output = Dense(n_classes, activation='softmax')(drop3)
    
    model = Model(inputs=input_layer, outputs=output)
    return model

# Compile and train models
print("\nCompiling and training models...")

# Callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0001)
]

# 1. Channel-CNN model
print("\nTraining Channel-CNN model...")
channel_input_shape = (X_channel_train.shape[1], X_channel_train.shape[2])
channel_model = create_channel_cnn(channel_input_shape, n_classes)
channel_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

channel_history = channel_model.fit(
    X_channel_train, y_channel_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# 2. Epoch-LSTM model
print("\nTraining Epoch-LSTM model...")
epoch_input_shape = (X_epoch_train.shape[1], X_epoch_train.shape[2])
epoch_model = create_epoch_lstm(epoch_input_shape, n_classes)
epoch_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epoch_history = epoch_model.fit(
    X_epoch_train, y_epoch_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# 3. EEGNet-inspired model
print("\nTraining EEGNet-inspired model...")
eegnet_input_shape = (X_3d_train.shape[1], X_3d_train.shape[2], X_3d_train.shape[3])
eegnet_model = create_eegnet(eegnet_input_shape, n_classes)
eegnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

eegnet_history = eegnet_model.fit(
    X_3d_train, y_3d_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Evaluate models
print("\nEvaluating models...")

# 1. Channel-CNN model
channel_loss, channel_acc = channel_model.evaluate(X_channel_test, y_channel_test)
print(f"Channel-CNN Test Accuracy: {channel_acc:.4f}")

# 2. Epoch-LSTM model
epoch_loss, epoch_acc = epoch_model.evaluate(X_epoch_test, y_epoch_test)
print(f"Epoch-LSTM Test Accuracy: {epoch_acc:.4f}")

# 3. EEGNet-inspired model
eegnet_loss, eegnet_acc = eegnet_model.evaluate(X_3d_test, y_3d_test)
print(f"EEGNet-inspired Test Accuracy: {eegnet_acc:.4f}")

# Get predictions
channel_pred = np.argmax(channel_model.predict(X_channel_test), axis=1)
epoch_pred = np.argmax(epoch_model.predict(X_epoch_test), axis=1)
eegnet_pred = np.argmax(eegnet_model.predict(X_3d_test), axis=1)

# Get true labels
y_channel_true = np.argmax(y_channel_test, axis=1)
y_epoch_true = np.argmax(y_epoch_test, axis=1)
y_3d_true = np.argmax(y_3d_test, axis=1)

# Print classification reports
print("\nChannel-CNN Classification Report:")
print(classification_report(y_channel_true, channel_pred))

print("\nEpoch-LSTM Classification Report:")
print(classification_report(y_epoch_true, epoch_pred))

print("\nEEGNet-inspired Classification Report:")
print(classification_report(y_3d_true, eegnet_pred))

# Plot confusion matrices
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
cm_channel = confusion_matrix(y_channel_true, channel_pred)
sns.heatmap(cm_channel, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Channel-CNN Confusion Matrix')

plt.subplot(1, 3, 2)
cm_epoch = confusion_matrix(y_epoch_true, epoch_pred)
sns.heatmap(cm_epoch, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Epoch-LSTM Confusion Matrix')

plt.subplot(1, 3, 3)
cm_eegnet = confusion_matrix(y_3d_true, eegnet_pred)
sns.heatmap(cm_eegnet, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('EEGNet-inspired Confusion Matrix')

plt.tight_layout()
plt.savefig('3d_models_confusion_matrices.png')
plt.close()

# Plot training history
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(channel_history.history['accuracy'], label='Train')
plt.plot(channel_history.history['val_accuracy'], label='Validation')
plt.title('Channel-CNN Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(epoch_history.history['accuracy'], label='Train')
plt.plot(epoch_history.history['val_accuracy'], label='Validation')
plt.title('Epoch-LSTM Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(eegnet_history.history['accuracy'], label='Train')
plt.plot(eegnet_history.history['val_accuracy'], label='Validation')
plt.title('EEGNet-inspired Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('3d_models_training_history.png')
plt.close()

# Compare with previous best model
print("\nComparing 3D models with previous best model (PCA + Ensemble):")
print(f"Previous best: 67.59% (PCA + Ensemble)")
print(f"Channel-CNN: {channel_acc:.4f}")
print(f"Epoch-LSTM: {epoch_acc:.4f}")
print(f"EEGNet-inspired: {eegnet_acc:.4f}")

print("\n3D classification models completed. Visualization images saved.")
