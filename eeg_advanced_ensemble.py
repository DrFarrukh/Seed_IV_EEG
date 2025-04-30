import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Loading EEG dataset...")
file_path = "combined_reduced_epochs (1).csv"
df = pd.read_csv(file_path)

# Identify feature columns (excluding metadata)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
metadata_cols = ['epoch', 'channel', 'label', 'trial_id']
feature_cols = [col for col in numerical_cols if col not in metadata_cols]

# Define important channels based on previous analysis
important_channels = [61, 58, 2, 31, 23, 4, 10, 15, 30, 40]

# Class-specific optimal channels
class_specific_channels = {
    0: [61, 2, 31, 23],    # Class 0 optimal channels
    1: [58, 10, 15, 40],   # Class 1 optimal channels
    2: [61, 58, 2, 30],    # Class 2 optimal channels
    3: [31, 23, 4, 15]     # Class 3 optimal channels
}

# Extract features with channel structure preserved
def extract_channel_features(df):
    """Extract features preserving channel structure"""
    unique_trials = sorted(df['trial_id'].unique())
    n_trials = len(unique_trials)
    
    # Initialize arrays
    X_channel = np.zeros((n_trials, len(important_channels), len(feature_cols)))
    y_channel = np.zeros(n_trials)
    
    # Extract features for each trial
    for i, trial in enumerate(unique_trials):
        trial_data = df[df['trial_id'] == trial]
        
        # Skip if trial doesn't have data
        if len(trial_data) == 0:
            continue
        
        # Get the label
        y_channel[i] = trial_data['label'].iloc[0]
        
        # Extract features for each important channel
        for j, channel in enumerate(important_channels):
            channel_data = trial_data[trial_data['channel'] == channel]
            
            # Skip if channel doesn't have data
            if len(channel_data) == 0:
                continue
            
            # Average features across epochs for this channel
            X_channel[i, j, :] = channel_data[feature_cols].mean().values
    
    return X_channel, y_channel.astype(int)

# Extract features for PCA
def extract_pca_features(df, n_components=20):
    """Extract PCA features"""
    unique_trials = sorted(df['trial_id'].unique())
    n_trials = len(unique_trials)
    
    # Initialize arrays
    X_trials = []
    y_trials = []
    
    # Extract features for each trial
    for trial in unique_trials:
        trial_data = df[df['trial_id'] == trial]
        
        # Skip if trial doesn't have data
        if len(trial_data) == 0:
            continue
        
        # Get the label
        label = trial_data['label'].iloc[0]
        
        # Extract features by averaging across epochs for each channel
        channel_features = trial_data.groupby('channel')[feature_cols].mean()
        # Flatten the channel features into a single vector
        features = channel_features.values.flatten()
        
        X_trials.append(features)
        y_trials.append(label)
    
    # Convert to numpy arrays
    X_trials = np.array(X_trials)
    y_trials = np.array(y_trials)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_trials)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, y_trials

# Data augmentation for EEG
def augment_eeg_data(X_channel, y_channel, augmentation_factor=2):
    """
    Augment EEG data using:
    1. Gaussian noise addition
    2. Channel swapping
    3. Time warping (simulated by feature perturbation)
    """
    n_samples, n_channels, n_features = X_channel.shape
    X_augmented = np.zeros((n_samples * augmentation_factor, n_channels, n_features))
    y_augmented = np.zeros(n_samples * augmentation_factor)
    
    # Copy original data
    X_augmented[:n_samples] = X_channel
    y_augmented[:n_samples] = y_channel
    
    # Generate augmented samples
    for i in range(1, augmentation_factor):
        start_idx = i * n_samples
        end_idx = (i + 1) * n_samples
        
        # Copy original data
        X_augmented[start_idx:end_idx] = X_channel.copy()
        y_augmented[start_idx:end_idx] = y_channel.copy()
        
        # Apply augmentation techniques
        for j in range(start_idx, end_idx):
            # 1. Add Gaussian noise (small amount to preserve signal)
            noise_level = 0.05 * np.random.rand()  # Random noise level up to 5%
            X_augmented[j] += np.random.normal(0, noise_level, X_augmented[j].shape)
            
            # 2. Swap random channels (within same hemisphere)
            if np.random.rand() > 0.5:
                ch1, ch2 = np.random.choice(range(n_channels), 2, replace=False)
                X_augmented[j, [ch1, ch2]] = X_augmented[j, [ch2, ch1]]
            
            # 3. Feature perturbation (simulating time warping)
            scaling_factor = 0.95 + 0.1 * np.random.rand()  # Random scaling between 0.95 and 1.05
            X_augmented[j] *= scaling_factor
    
    return X_augmented, y_augmented

# Extract and prepare data
print("Extracting features...")
X_channel, y_channel = extract_channel_features(df)
X_pca, y_pca = extract_pca_features(df, n_components=20)

# Augment data
print("Augmenting data...")
X_augmented, y_augmented = augment_eeg_data(X_channel, y_channel, augmentation_factor=3)

print(f"Original data shape: {X_channel.shape}")
print(f"Augmented data shape: {X_augmented.shape}")

# Split data
X_train_channel, X_test_channel, y_train_channel, y_test_channel = train_test_split(
    X_channel, y_channel, test_size=0.3, random_state=42, stratify=y_channel
)

X_train_aug, X_test_aug, y_train_aug, y_test_aug = train_test_split(
    X_augmented, y_augmented, test_size=0.3, random_state=42, stratify=y_augmented
)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y_pca, test_size=0.3, random_state=42, stratify=y_pca
)

# Convert to categorical for deep learning
y_train_cat = to_categorical(y_train_aug, num_classes=4)
y_test_cat = to_categorical(y_test_channel, num_classes=4)

# Create CNN model for channel data
def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(256, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create class-specific models
def create_class_specific_models():
    models = []
    
    # Class 0 model (SVM)
    models.append(SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42))
    
    # Class 1 model (Random Forest)
    models.append(RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1))
    
    # Class 2 model (Gradient Boosting)
    models.append(GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42))
    
    # Class 3 model (Random Forest with different params)
    models.append(RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1))
    
    return models

# Create and train CNN model
print("Training CNN model...")
# Reshape for CNN (samples, timesteps, features)
X_train_cnn = X_train_aug.reshape(X_train_aug.shape[0], X_train_aug.shape[1], X_train_aug.shape[2])
X_test_cnn = X_test_channel.reshape(X_test_channel.shape[0], X_test_channel.shape[1], X_test_channel.shape[2])

cnn_model = create_cnn_model((X_train_cnn.shape[1], X_train_cnn.shape[2]))

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Train CNN model
cnn_history = cnn_model.fit(
    X_train_cnn, y_train_cat,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Train class-specific models
print("Training class-specific models...")
class_models = create_class_specific_models()

# Train each model
for i, model in enumerate(class_models):
    # Create binary target (1 for current class, 0 for others)
    binary_y_train = (y_train_pca == i).astype(int)
    model.fit(X_train_pca, binary_y_train)

# Advanced Ensemble Classifier
class AdvancedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, class_models, cnn_model):
        self.class_models = class_models
        self.cnn_model = cnn_model
        
    def fit(self, X, y):
        # Already fitted in main code
        return self
    
    def predict(self, X_pca, X_cnn):
        # Get CNN predictions
        cnn_preds = self.cnn_model.predict(X_cnn)
        cnn_classes = np.argmax(cnn_preds, axis=1)
        
        # Get class-specific model predictions
        class_preds = np.zeros((X_pca.shape[0], len(self.class_models)))
        
        for i, model in enumerate(self.class_models):
            # Predict probability of being in this class
            if hasattr(model, "predict_proba"):
                class_preds[:, i] = model.predict_proba(X_pca)[:, 1]
            else:
                # For models without predict_proba, use decision_function
                class_preds[:, i] = model.decision_function(X_pca)
        
        # Normalize class_preds
        class_preds = (class_preds - class_preds.min(axis=1, keepdims=True)) / \
                     (class_preds.max(axis=1, keepdims=True) - class_preds.min(axis=1, keepdims=True) + 1e-10)
        
        # Combine predictions (weighted average)
        cnn_weight = 0.6  # Give more weight to CNN
        class_weight = 0.4
        
        # Convert CNN predictions to one-hot
        cnn_one_hot = np.zeros_like(class_preds)
        for i, cls in enumerate(cnn_classes):
            cnn_one_hot[i, cls] = 1
        
        # Weighted combination
        combined_preds = (cnn_weight * cnn_one_hot) + (class_weight * class_preds)
        
        # Return class with highest probability
        return np.argmax(combined_preds, axis=1)

# Create and evaluate advanced ensemble
print("Evaluating advanced ensemble...")
advanced_ensemble = AdvancedEnsembleClassifier(class_models, cnn_model)
y_pred = advanced_ensemble.predict(X_test_pca, X_test_cnn)
accuracy = accuracy_score(y_test_channel, y_pred)

print(f"Advanced Ensemble Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_channel, y_pred))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_channel, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Advanced Ensemble')
plt.tight_layout()
plt.savefig('advanced_ensemble_confusion_matrix.png')
plt.close()

# Compare with previous approaches
print("\nComparing with previous approaches:")
print(f"PCA + Ensemble (20 components): 68.06%")
print(f"Specialized Ensemble (50 features): 66.67%")
print(f"Super Learner: 62.96%")
print(f"Advanced Ensemble: {accuracy:.4f}")

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(cnn_history.history['accuracy'], label='Train')
plt.plot(cnn_history.history['val_accuracy'], label='Validation')
plt.title('CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('cnn_learning_curve.png')
plt.close()

print("\nAdvanced ensemble analysis completed. Visualization images saved.")
