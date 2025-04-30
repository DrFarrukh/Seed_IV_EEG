import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Concatenate
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

# Function to extract trial features with important channels only
def extract_balanced_features(df):
    """
    Extract features using only important channels and apply PCA
    to reduce dimensionality while preserving variance
    """
    # Get unique trials
    unique_trials = sorted(df['trial_id'].unique())
    n_trials = len(unique_trials)
    
    print(f"Extracting balanced features for {n_trials} trials")
    
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
    pca = PCA(n_components=30)  # Balanced number of components
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Explained variance with 30 components: {pca.explained_variance_ratio_.sum():.4f}")
    
    return X_pca, y_trials, X_scaled

# Function to reshape data into 3D format for deep learning
def reshape_to_3d(df, feature_cols):
    """
    Reshape the dataset into a 3D format (trials × channels × features)
    for use with deep learning models
    """
    # Get unique trials, channels, and epochs
    unique_trials = sorted(df['trial_id'].unique())
    unique_channels = sorted(df['channel'].unique())
    
    n_trials = len(unique_trials)
    n_channels = len(unique_channels)
    n_features = len(feature_cols)
    
    print(f"Reshaping to 3D: {n_trials} trials × {n_channels} channels × {n_features} features")
    
    # Initialize arrays
    X_3d = np.zeros((n_trials, n_channels, n_features))
    y_3d = np.zeros(n_trials)
    
    # Fill arrays
    for i, trial in enumerate(unique_trials):
        trial_data = df[df['trial_id'] == trial]
        
        # Skip if trial doesn't have data
        if len(trial_data) == 0:
            continue
        
        # Get the label (should be the same for all rows in the trial)
        y_3d[i] = trial_data['label'].iloc[0]
        
        # Extract features for each channel
        for j, channel in enumerate(unique_channels):
            channel_data = trial_data[trial_data['channel'] == channel]
            
            # Skip if channel doesn't have data
            if len(channel_data) == 0:
                continue
            
            # Average features across epochs for this channel
            X_3d[i, j, :] = channel_data[feature_cols].mean().values
    
    return X_3d, y_3d.astype(int)

# Extract features using both approaches
print("\nExtracting features for super learner...")
X_balanced, y_balanced, X_scaled_full = extract_balanced_features(df)
X_3d, y_3d = reshape_to_3d(df, feature_cols)

print(f"Balanced feature shape: {X_balanced.shape}")
print(f"3D feature shape: {X_3d.shape}")
print(f"Number of trials: {len(y_balanced)}")
print(f"Class distribution: {np.bincount(y_balanced)}")

# Split data into training and testing sets
print("\nSplitting data into training and testing sets...")
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(
    X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
)

X_train_3d, X_test_3d, y_train_3d, y_test_3d = train_test_split(
    X_3d, y_3d, test_size=0.3, random_state=42, stratify=y_3d
)

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_scaled_full, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
)

print(f"Training set (balanced): {X_train_balanced.shape}, Testing set: {X_test_balanced.shape}")
print(f"Training set (3D): {X_train_3d.shape}, Testing set: {X_test_3d.shape}")

# Define base models for meta-learning
print("\nDefining base models for meta-learning...")
base_models = [
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))
]

# Create meta-learner (stacking classifier)
meta_model = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    stack_method='predict_proba'
)

# Train meta-learner
print("\nTraining meta-learner...")
stacking_clf.fit(X_train_balanced, y_train_balanced)
meta_preds = stacking_clf.predict_proba(X_test_balanced)

# Define deep learning model for 3D data
def create_deep_3d_model(input_shape, num_classes):
    """
    Create a deep learning model for 3D EEG data
    """
    inputs = Input(shape=input_shape)
    
    # Flatten the 3D input
    x = tf.keras.layers.Reshape((input_shape[0] * input_shape[1],))(inputs)
    
    # First dense block
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Second dense block
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Third dense block
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Define deep learning model for PCA features
def create_deep_pca_model(input_shape, num_classes):
    """
    Create a deep learning model for PCA-reduced features
    """
    inputs = Input(shape=(input_shape,))
    
    # First dense block
    x = Dense(64, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second dense block
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Define callbacks for training
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

# Train deep learning models
print("\nTraining deep learning models...")
# Convert labels to one-hot encoding
y_train_categorical = to_categorical(y_train_balanced, num_classes=4)
y_test_categorical = to_categorical(y_test_balanced, num_classes=4)

y_train_3d_categorical = to_categorical(y_train_3d, num_classes=4)
y_test_3d_categorical = to_categorical(y_test_3d, num_classes=4)

# Create and train 3D model
deep_3d_model = create_deep_3d_model((X_train_3d.shape[1], X_train_3d.shape[2]), 4)
deep_3d_history = deep_3d_model.fit(
    X_train_3d,
    y_train_3d_categorical,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Create and train PCA model
deep_pca_model = create_deep_pca_model(X_train_balanced.shape[1], 4)
deep_pca_history = deep_pca_model.fit(
    X_train_balanced,
    y_train_categorical,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Get predictions from deep learning models
deep_3d_preds = deep_3d_model.predict(X_test_3d)
deep_pca_preds = deep_pca_model.predict(X_test_balanced)

# Create super learner model
def create_super_learner_model(meta_shape, deep_3d_shape, deep_pca_shape, num_classes):
    """
    Create a super learner model that combines predictions from
    meta-learning and deep learning models
    """
    # Input for meta-learner predictions
    meta_input = Input(shape=(meta_shape,))
    
    # Input for deep 3D model predictions
    deep_3d_input = Input(shape=(deep_3d_shape,))
    
    # Input for deep PCA model predictions
    deep_pca_input = Input(shape=(deep_pca_shape,))
    
    # Concatenate all predictions
    combined = Concatenate()([meta_input, deep_3d_input, deep_pca_input])
    
    # Dense layer to learn optimal weighting
    x = Dense(32, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=[meta_input, deep_3d_input, deep_pca_input], outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Generate cross-validation predictions for training the super learner
print("\nGenerating cross-validation predictions for super learner...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Get meta-learner CV predictions
meta_train_preds = np.zeros((X_train_balanced.shape[0], 4))
for model_name, model in base_models:
    cv_preds = cross_val_predict(model, X_train_balanced, y_train_balanced, cv=cv, method='predict_proba')
    meta_train_preds += cv_preds / len(base_models)

# Create and train super learner
print("\nTraining super learner model...")
super_learner = create_super_learner_model(4, 4, 4, 4)

# Use validation data from deep learning models as training data for super learner
val_split_idx = int(0.8 * X_train_balanced.shape[0])
X_val_balanced = X_train_balanced[val_split_idx:]
y_val_categorical = y_train_categorical[val_split_idx:]

# Get predictions on validation data
meta_val_preds = stacking_clf.predict_proba(X_val_balanced)
deep_3d_val_preds = deep_3d_model.predict(X_train_3d[val_split_idx:])
deep_pca_val_preds = deep_pca_model.predict(X_val_balanced)

# Train super learner
super_learner_history = super_learner.fit(
    [meta_val_preds, deep_3d_val_preds, deep_pca_val_preds],
    y_val_categorical,
    epochs=50,
    batch_size=16,
    callbacks=[early_stopping],
    verbose=1
)

# Make final predictions with super learner
print("\nMaking final predictions with super learner...")
super_learner_preds = super_learner.predict([meta_preds, deep_3d_preds, deep_pca_preds])
super_learner_classes = np.argmax(super_learner_preds, axis=1)

# Evaluate individual models
meta_classes = np.argmax(meta_preds, axis=1)
deep_3d_classes = np.argmax(deep_3d_preds, axis=1)
deep_pca_classes = np.argmax(deep_pca_preds, axis=1)

meta_accuracy = accuracy_score(y_test_balanced, meta_classes)
deep_3d_accuracy = accuracy_score(y_test_balanced, deep_3d_classes)
deep_pca_accuracy = accuracy_score(y_test_balanced, deep_pca_classes)
super_learner_accuracy = accuracy_score(y_test_balanced, super_learner_classes)

print(f"Meta-learner accuracy: {meta_accuracy:.4f}")
print(f"Deep 3D model accuracy: {deep_3d_accuracy:.4f}")
print(f"Deep PCA model accuracy: {deep_pca_accuracy:.4f}")
print(f"Super learner accuracy: {super_learner_accuracy:.4f}")

# Print classification report for super learner
print("\nClassification Report (Super Learner):")
print(classification_report(y_test_balanced, super_learner_classes))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_balanced, super_learner_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Super Learner')
plt.tight_layout()
plt.savefig('super_learner_confusion_matrix.png')
plt.close()

# Compare with previous approaches
print("\nComparing with previous approaches:")
print(f"PCA + Ensemble (20 components): 68.06%")
print(f"Specialized Ensemble (50 features): 66.67%")
print(f"Balanced Final Model (30 features): 66.67%")
print(f"Meta-learner: {meta_accuracy:.4f}")
print(f"Deep 3D model: {deep_3d_accuracy:.4f}")
print(f"Deep PCA model: {deep_pca_accuracy:.4f}")
print(f"Super learner: {super_learner_accuracy:.4f}")

# Plot learning curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(deep_3d_history.history['accuracy'], label='Train')
plt.plot(deep_3d_history.history['val_accuracy'], label='Validation')
plt.title('Deep 3D Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(deep_pca_history.history['accuracy'], label='Train')
plt.plot(deep_pca_history.history['val_accuracy'], label='Validation')
plt.title('Deep PCA Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('deep_learning_curves.png')
plt.close()

# Analyze per-class performance
class_report = classification_report(y_test_balanced, super_learner_classes, output_dict=True)
class_f1 = [class_report[str(i)]['f1-score'] for i in range(4)]

plt.figure(figsize=(10, 6))
plt.bar(['Class 0', 'Class 1', 'Class 2', 'Class 3'], class_f1, color='skyblue')
plt.axhline(y=0.5, color='r', linestyle='--', label='Baseline')
plt.xlabel('Class')
plt.ylabel('F1-Score')
plt.title('Per-Class Performance - Super Learner')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig('super_learner_per_class_performance.png')
plt.close()

print("\nSuper learner analysis completed. Visualization images saved.")
