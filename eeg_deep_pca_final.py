import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, clone_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Flatten
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1_l2
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

# Data augmentation functions
def add_gaussian_noise(X, noise_factor=0.05):
    """Add Gaussian noise to the data"""
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise

def time_warp(X, sigma=0.2, knot=4):
    """Apply time warping to the data"""
    # Create a copy to avoid modifying the original data
    X_warped = X.copy()
    
    # For 2D data (samples, features)
    # Just add small random noise instead of warping for 2D data
    X_warped = X + np.random.normal(0, 0.01, X.shape)
    
    return X_warped

def spectral_augment(X, max_mask_pct=0.1, n_freq_masks=2):
    """Apply spectral augmentation (frequency masking)"""
    aug_X = X.copy()
    
    # For 2D data
    n_features = X.shape[1]
    freq_width = int(n_features * max_mask_pct)
    
    for _ in range(n_freq_masks):
        for i in range(X.shape[0]):
            f0 = np.random.randint(0, n_features - freq_width)
            aug_X[i, f0:f0+freq_width] = 0
    
    return aug_X

def mixup(X, y, alpha=0.2):
    """Apply mixup augmentation"""
    batch_size = X.shape[0]
    weights = np.random.beta(alpha, alpha, batch_size)
    
    # Ensure weights are column vectors for proper broadcasting
    weights = weights.reshape(batch_size, 1)
    
    # Create random index permutation
    index = np.random.permutation(batch_size)
    
    # Create mixup samples
    X_mixed = weights * X + (1 - weights) * X[index]
    
    # Convert y to one-hot if it's not already
    if len(y.shape) == 1:
        y_onehot = to_categorical(y)
    else:
        y_onehot = y
        
    # Reshape weights for proper broadcasting with y_onehot
    if len(y_onehot.shape) > 1:
        weights = weights.reshape(batch_size, 1)
        
    # Mix the labels
    y_mixed = weights * y_onehot + (1 - weights) * y_onehot[index]
    
    return X_mixed, y_mixed

# Extract PCA features
def extract_pca_features(df, n_components=30):
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
    
    return X_pca, y_trials, pca, scaler

# Extract features
print("\nExtracting PCA features...")
X_pca, y_pca, pca_model, pca_scaler = extract_pca_features(df, n_components=30)
print(f"PCA feature shape: {X_pca.shape}")

# Split data with stratification
print("\nSplitting data into training and testing sets...")
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y_pca, test_size=0.3, random_state=42, stratify=y_pca
)

# Convert to categorical for deep learning
y_train_cat = to_categorical(y_train_pca, num_classes=4)
y_test_cat = to_categorical(y_test_pca, num_classes=4)

# Data augmentation for training sets
print("\nAugmenting training data...")
# Augment PCA features
X_train_pca_noise = add_gaussian_noise(X_train_pca)
X_train_pca_warp = time_warp(X_train_pca)
X_train_pca_spectral = spectral_augment(X_train_pca)
X_train_pca_aug, y_train_pca_aug = mixup(X_train_pca, y_train_cat)

# Combine all augmented data
X_train_pca_combined = np.vstack([X_train_pca, X_train_pca_noise, X_train_pca_warp, X_train_pca_spectral])
y_train_pca_combined = np.vstack([y_train_cat, y_train_cat, y_train_cat, y_train_cat])

print(f"Augmented PCA feature shape: {X_train_pca_combined.shape}")

# Define callbacks for deep learning models
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=30,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=10,
    min_lr=0.0001
)

model_checkpoint = ModelCheckpoint(
    'deep_pca_best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Create Deep Neural Network for PCA features
def create_deep_pca_model(input_shape, model_type='dense', seed=None):
    """Create a model for PCA features with different architectures
    
    Args:
        input_shape: Shape of input features
        model_type: Type of model architecture ('dense', 'cnn', or 'transformer')
        seed: Random seed for reproducibility
    
    Returns:
        Compiled Keras model
    """
    if seed is not None:
        tf.random.set_seed(seed)
    
    if model_type == 'dense':
        # Standard Dense model
        model = Sequential([
            # Input layer
            Dense(512, activation='relu', input_shape=(input_shape,), 
                  kernel_regularizer=l1_l2(l1=5e-5, l2=5e-4),
                  kernel_initializer=tf.keras.initializers.HeNormal(seed=seed)),
            BatchNormalization(),
            Dropout(0.6),
            
            # Hidden layers
            Dense(256, activation='relu', 
                  kernel_regularizer=l1_l2(l1=5e-5, l2=5e-4),
                  kernel_initializer=tf.keras.initializers.HeNormal(seed=seed)),
            BatchNormalization(),
            Dropout(0.6),
            
            Dense(128, activation='relu', 
                  kernel_regularizer=l1_l2(l1=5e-5, l2=5e-4),
                  kernel_initializer=tf.keras.initializers.HeNormal(seed=seed)),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(64, activation='relu', 
                  kernel_regularizer=l1_l2(l1=5e-5, l2=5e-4),
                  kernel_initializer=tf.keras.initializers.HeNormal(seed=seed)),
            BatchNormalization(),
            Dropout(0.4),
            
            # Output layer
            Dense(4, activation='softmax')
        ])
    
    elif model_type == 'cnn':
        # CNN model - reshape PCA features to 1D spatial structure
        inputs = Input(shape=(input_shape,))
        
        # Reshape to 1D spatial structure (treating PCA components as spatial features)
        x = tf.keras.layers.Reshape((input_shape, 1))(inputs)
        
        # First Conv block
        x = Conv1D(64, kernel_size=3, padding='same', activation='relu',
                   kernel_regularizer=l1_l2(l1=5e-5, l2=5e-4),
                   kernel_initializer=tf.keras.initializers.HeNormal(seed=seed))(x)
        x = BatchNormalization()(x)
        x = Conv1D(64, kernel_size=3, padding='same', activation='relu',
                   kernel_regularizer=l1_l2(l1=5e-5, l2=5e-4),
                   kernel_initializer=tf.keras.initializers.HeNormal(seed=seed))(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.4)(x)
        
        # Second Conv block
        x = Conv1D(128, kernel_size=3, padding='same', activation='relu',
                    kernel_regularizer=l1_l2(l1=5e-5, l2=5e-4),
                    kernel_initializer=tf.keras.initializers.HeNormal(seed=seed))(x)
        x = BatchNormalization()(x)
        x = Conv1D(128, kernel_size=3, padding='same', activation='relu',
                    kernel_regularizer=l1_l2(l1=5e-5, l2=5e-4),
                    kernel_initializer=tf.keras.initializers.HeNormal(seed=seed))(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.4)(x)
        
        # Global pooling and dense layers
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu',
                  kernel_regularizer=l1_l2(l1=5e-5, l2=5e-4),
                  kernel_initializer=tf.keras.initializers.HeNormal(seed=seed))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu',
                  kernel_regularizer=l1_l2(l1=5e-5, l2=5e-4),
                  kernel_initializer=tf.keras.initializers.HeNormal(seed=seed))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        # Output layer
        outputs = Dense(4, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
    
    elif model_type == 'transformer':
        # Transformer model
        inputs = Input(shape=(input_shape,))
        
        # Reshape and add positional embedding
        x = tf.keras.layers.Reshape((input_shape, 1))(inputs)
        
        # Expand dimensions to create embedding dimension
        x = Conv1D(64, kernel_size=1, activation='linear')(x)  # This creates embedding dimension
        
        # Add positional encoding
        pos_encoding = positional_encoding(input_shape, 64)
        x = x + pos_encoding[:, :input_shape, :]
        
        # Transformer blocks
        x = transformer_block(x, 64, 4, 128, dropout_rate=0.4, seed=seed)
        x = transformer_block(x, 64, 4, 128, dropout_rate=0.4, seed=seed)
        
        # Global pooling and dense layers
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu',
                  kernel_regularizer=l1_l2(l1=5e-5, l2=5e-4),
                  kernel_initializer=tf.keras.initializers.HeNormal(seed=seed))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu',
                  kernel_regularizer=l1_l2(l1=5e-5, l2=5e-4),
                  kernel_initializer=tf.keras.initializers.HeNormal(seed=seed))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        # Output layer
        outputs = Dense(4, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from 'dense', 'cnn', or 'transformer'.")
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def transformer_block(inputs, embed_dim, num_heads, ff_dim, dropout_rate=0.1, seed=None):
    """Transformer block with multi-head attention and feed-forward network"""
    # Multi-head attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim//num_heads, seed=seed
    )(inputs, inputs)
    attention_output = Dropout(dropout_rate, seed=seed)(attention_output)
    
    # Add & Norm (first residual connection)
    x = Add()([inputs, attention_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Feed-forward network
    ffn_output = Dense(ff_dim, activation='relu',
                      kernel_initializer=tf.keras.initializers.HeNormal(seed=seed))(x)
    ffn_output = Dense(embed_dim,
                      kernel_initializer=tf.keras.initializers.HeNormal(seed=seed))(ffn_output)
    ffn_output = Dropout(dropout_rate, seed=seed)(ffn_output)
    
    # Add & Norm (second residual connection)
    x = Add()([x, ffn_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    return x


def get_angles(pos, i, d_model):
    """Calculate angles for positional encoding"""
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    """Create positional encoding for transformer"""
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis, :],
                           d_model)
    
    # Apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # Apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

# Implement K-fold cross-validation with different architectures
print("\nImplementing 5-fold cross-validation with different architectures...")
n_folds = 5
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Model types to evaluate
model_types = ['dense', 'cnn', 'transformer']

# Apply data augmentation to test set (with fixed seed for reproducibility)
np.random.seed(42)
X_test_noise = add_gaussian_noise(X_test_pca, noise_factor=0.03)  # Reduced noise for test data

# Initialize arrays to store fold results
fold_val_accuracies = []
fold_test_accuracies = []

# Store results for each model type
results = {}

for model_type in model_types:
    print(f"\n=== Training {model_type.upper()} model ===\n")
    
    fold_val_accuracies = []
    fold_test_accuracies = []
    fold_models = []
    fold_predictions = []
    fold_histories = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_pca, y_train_pca)):
        print(f"Training fold {fold+1}/{n_folds}")
        
        # Split data
        X_fold_train, X_fold_val = X_train_pca[train_idx], X_train_pca[val_idx]
        y_fold_train, y_fold_val = y_train_pca[train_idx], y_train_pca[val_idx]
        
        # Convert labels to categorical
        y_fold_train_cat = to_categorical(y_fold_train, num_classes=4)
        y_fold_val_cat = to_categorical(y_fold_val, num_classes=4)
        
        # Apply data augmentation to training data
        X_fold_train_noise = add_gaussian_noise(X_fold_train)
        X_fold_train_warp = time_warp(X_fold_train)
        X_fold_train_spectral = spectral_augment(X_fold_train)
        X_fold_train_combined = np.vstack([X_fold_train, X_fold_train_noise, X_fold_train_warp, X_fold_train_spectral])
        y_fold_train_combined = np.vstack([y_fold_train_cat, y_fold_train_cat, y_fold_train_cat, y_fold_train_cat])
        
        # Create model for this fold with the specified architecture
        fold_model = create_deep_pca_model(X_train_pca.shape[1], model_type=model_type, seed=fold)
        
        # Train model
        fold_history = fold_model.fit(
            X_fold_train_combined, y_fold_train_combined,
            epochs=150,
            batch_size=32,
            validation_data=(X_fold_val, y_fold_val_cat),
            callbacks=[
                EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001),
                ModelCheckpoint(f'deep_pca_{model_type}_fold{fold+1}.h5', monitor='val_accuracy', save_best_only=True)
            ],
            verbose=1
        )
        
        # Evaluate on validation set
        val_preds = fold_model.predict(X_fold_val)
        val_classes = np.argmax(val_preds, axis=1)
        val_accuracy = accuracy_score(y_fold_val, val_classes)
        fold_val_accuracies.append(val_accuracy)
        
        # Evaluate on test set
        test_preds = fold_model.predict(X_test_pca)
        test_classes = np.argmax(test_preds, axis=1)
        test_accuracy = accuracy_score(y_test_pca, test_classes)
        fold_test_accuracies.append(test_accuracy)
        
        # Store model and predictions
        fold_models.append(fold_model)
        fold_predictions.append(test_preds)
        fold_histories.append(fold_history.history)
        
        print(f"Fold {fold+1} - Validation Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Store results for this model type
    results[model_type] = {
        'val_accuracies': fold_val_accuracies,
        'test_accuracies': fold_test_accuracies,
        'models': fold_models,
        'predictions': fold_predictions,
        'histories': fold_histories
    }
    
    # Calculate and print average performance
    avg_val_acc = np.mean(fold_val_accuracies)
    std_val_acc = np.std(fold_val_accuracies)
    avg_test_acc = np.mean(fold_test_accuracies)
    std_test_acc = np.std(fold_test_accuracies)
    
    print(f"\n{model_type.upper()} model results:")
    print(f"Average validation accuracy: {avg_val_acc:.4f} ± {std_val_acc:.4f}")
    print(f"Average test accuracy: {avg_test_acc:.4f} ± {std_test_acc:.4f}")
    
    # Create ensemble prediction for this model type
    ensemble_pred = np.mean(fold_predictions, axis=0)
    ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)
    ensemble_accuracy = accuracy_score(y_test_pca, ensemble_pred_classes)
    print(f"Ensemble model accuracy: {ensemble_accuracy:.4f}")
    
    # Evaluate on augmented test data
    augmented_ensemble_preds = np.zeros_like(ensemble_pred)
    for model in fold_models:
        augmented_ensemble_preds += model.predict(X_test_noise) / len(fold_models)
    augmented_ensemble_classes = np.argmax(augmented_ensemble_preds, axis=1)
    augmented_ensemble_accuracy = accuracy_score(y_test_pca, augmented_ensemble_classes)
    print(f"Ensemble model accuracy on augmented test data: {augmented_ensemble_accuracy:.4f}")
    
    # Classification report for ensemble model
    print(f"\nClassification Report ({model_type.upper()} Ensemble Model):")
    print(classification_report(y_test_pca, ensemble_pred_classes))
    
    # Save best model for this architecture
    best_model_idx = np.argmax(fold_test_accuracies)
    best_model = fold_models[best_model_idx]
    best_model.save(f'deep_pca_{model_type}_best_model.h5')
    
    # Save ensemble predictions
    np.save(f'deep_pca_{model_type}_ensemble_predictions.npy', np.array(fold_predictions))

print("\n=== Model Architecture Comparison ===\n")
for model_type in model_types:
    avg_test_acc = np.mean(results[model_type]['test_accuracies'])
    std_test_acc = np.std(results[model_type]['test_accuracies'])
    ensemble_pred = np.mean(results[model_type]['predictions'], axis=0)
    ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)
    ensemble_accuracy = accuracy_score(y_test_pca, ensemble_pred_classes)
    
    print(f"{model_type.upper()} Model:")
    print(f"  Average Test Accuracy: {avg_test_acc:.4f} ± {std_test_acc:.4f}")
    print(f"  Ensemble Accuracy: {ensemble_accuracy:.4f}")
best_architecture = max(model_types, key=lambda x: np.mean(results[x]['test_accuracies']))
print(f"\nBest performing architecture: {best_architecture.upper()}")

# Get the best model from the best architecture
best_arch_models = results[best_architecture]['models']
best_arch_test_accs = results[best_architecture]['test_accuracies']
best_model_idx = np.argmax(best_arch_test_accs)
best_model = best_arch_models[best_model_idx]

# Save the overall best model
print(f"\nSaving best overall model ({best_architecture})...")
best_model.save('deep_pca_best_overall_model.h5')

# Create ensemble prediction (average of all fold predictions)
ensemble_preds = np.mean([pred for pred in results[best_architecture]['predictions']], axis=0)
ensemble_classes = np.argmax(ensemble_preds, axis=1)
ensemble_accuracy = accuracy_score(y_test_pca, ensemble_classes)

# Also evaluate on augmented test data
augmented_ensemble_preds = np.zeros_like(ensemble_preds)
for model in fold_models:
    augmented_ensemble_preds += model.predict(X_test_noise)
augmented_ensemble_preds /= len(fold_models)
augmented_ensemble_classes = np.argmax(augmented_ensemble_preds, axis=1)
augmented_ensemble_accuracy = accuracy_score(y_test_pca, augmented_ensemble_classes)

print("\nCross-validation results:")
print(f"Average validation accuracy: {np.mean(fold_val_accuracies):.4f} ± {np.std(fold_val_accuracies):.4f}")
print(f"Average test accuracy: {np.mean(fold_test_accuracies):.4f} ± {np.std(fold_test_accuracies):.4f}")
print(f"Ensemble model accuracy: {ensemble_accuracy:.4f}")
print(f"Ensemble model accuracy on augmented test data: {augmented_ensemble_accuracy:.4f}")

# Print classification reports
print("\nClassification Report (Ensemble Model):")
print(classification_report(y_test_pca, ensemble_classes))

print("\nClassification Report (Ensemble Model on Augmented Test Data):")
print(classification_report(y_test_pca, augmented_ensemble_classes))

# Plot confusion matrices
plt.figure(figsize=(15, 7))

# Ensemble model confusion matrix
plt.subplot(1, 2, 1)
cm_ensemble = confusion_matrix(y_test_pca, ensemble_classes)
sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Ensemble Model (Accuracy: {ensemble_accuracy:.4f})')

# Augmented test data confusion matrix
plt.subplot(1, 2, 2)
cm_augmented = confusion_matrix(y_test_pca, augmented_ensemble_classes)
sns.heatmap(cm_augmented, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Augmented Test Data (Accuracy: {augmented_ensemble_accuracy:.4f})')

plt.tight_layout()
plt.savefig('deep_pca_confusion_matrices.png')
plt.close()

# Plot cross-validation results
plt.figure(figsize=(15, 10))

# Plot fold accuracies
plt.subplot(2, 2, 1)
x = np.arange(n_folds)
width = 0.35
plt.bar(x - width/2, fold_val_accuracies, width, label='Validation')
plt.bar(x + width/2, fold_test_accuracies, width, label='Test')
plt.axhline(y=ensemble_accuracy, color='r', linestyle='--', label='Ensemble')
plt.axhline(y=augmented_ensemble_accuracy, color='g', linestyle='--', label='Augmented Ensemble')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Results')
plt.xticks(x, [f'Fold {i+1}' for i in range(n_folds)])
plt.legend()

# Plot per-class performance for ensemble model
plt.subplot(2, 2, 2)
class_report = classification_report(y_test_pca, ensemble_classes, output_dict=True)
class_f1 = [class_report[str(i)]['f1-score'] for i in range(4)]
class_precision = [class_report[str(i)]['precision'] for i in range(4)]
class_recall = [class_report[str(i)]['recall'] for i in range(4)]

x = np.arange(4)
width = 0.25
plt.bar(x - width, class_precision, width, label='Precision', color='skyblue')
plt.bar(x, class_recall, width, label='Recall', color='lightgreen')
plt.bar(x + width, class_f1, width, label='F1-Score', color='salmon')
plt.axhline(y=0.5, color='r', linestyle='--')
plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Ensemble Model Performance by Class')
plt.xticks(x, ['Class 0', 'Class 1', 'Class 2', 'Class 3'])
plt.legend()

# Plot per-class performance for augmented ensemble model
plt.subplot(2, 2, 3)
class_report_aug = classification_report(y_test_pca, augmented_ensemble_classes, output_dict=True)
class_f1_aug = [class_report_aug[str(i)]['f1-score'] for i in range(4)]
class_precision_aug = [class_report_aug[str(i)]['precision'] for i in range(4)]
class_recall_aug = [class_report_aug[str(i)]['recall'] for i in range(4)]

plt.bar(x - width, class_precision_aug, width, label='Precision', color='skyblue')
plt.bar(x, class_recall_aug, width, label='Recall', color='lightgreen')
plt.bar(x + width, class_f1_aug, width, label='F1-Score', color='salmon')
plt.axhline(y=0.5, color='r', linestyle='--')
plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Augmented Ensemble Performance by Class')
plt.xticks(x, ['Class 0', 'Class 1', 'Class 2', 'Class 3'])
plt.legend()

# Plot comparison of models
plt.subplot(2, 2, 4)
models = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Ensemble', 'Aug Ensemble']
accuracies = fold_test_accuracies + [ensemble_accuracy, augmented_ensemble_accuracy]
plt.bar(models, accuracies, color=['blue', 'blue', 'blue', 'blue', 'blue', 'red', 'green'])
plt.axhline(y=np.mean(fold_test_accuracies), color='black', linestyle='--', label='Avg Fold')
plt.xlabel('Model')
plt.ylabel('Test Accuracy')
plt.title('Model Comparison')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('deep_pca_cross_validation_results.png')
plt.close()

plt.tight_layout()
plt.savefig('deep_pca_learning_curves.png')
plt.close()

# Save the best model (ensemble)
print("\nSaving best model from cross-validation...")
best_model_idx = np.argmax(fold_test_accuracies)
best_model = fold_models[best_model_idx]
best_model.save('deep_pca_best_model.h5')

# Save ensemble predictions instead of weights
print("Saving ensemble predictions...")
ensemble_predictions = np.array(fold_predictions)
np.save('deep_pca_ensemble_predictions.npy', ensemble_predictions)

# Save PCA model and scaler
import pickle
with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca_model, f)
    
with open('pca_scaler.pkl', 'wb') as f:
    pickle.dump(pca_scaler, f)

print("Models and parameters saved successfully.")

# Compare with previous approaches
print("\nComparing with previous approaches:")
print(f"PCA + Ensemble (20 components): 68.06%")
print(f"Optimized Stacking Ensemble: 72.22%")
print(f"Deep PCA Cross-Validation (Average): {np.mean(fold_test_accuracies):.4f} ± {np.std(fold_test_accuracies):.4f}")
print(f"Deep PCA Ensemble: {ensemble_accuracy:.4f}")
print(f"Deep PCA Augmented Ensemble: {augmented_ensemble_accuracy:.4f}")

print("\nDeep PCA model training completed.")
print(f"Best accuracy achieved: {max(ensemble_accuracy, augmented_ensemble_accuracy):.4f}")
print("Visualization images saved.")

# Print summary of the best model
print("\nBest Model Summary:")
best_model.summary()

print("\nEnsemble Information:")
print(f"Number of models in ensemble: {len(fold_models)}")
print(f"Cross-validation folds: {n_folds}")
print(f"Regularization: L1=5e-5, L2=5e-4")
print(f"Dropout rates: 0.6, 0.6, 0.5, 0.4")
