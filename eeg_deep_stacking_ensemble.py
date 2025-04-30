import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
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

# Identify feature columns (excluding metadata)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
metadata_cols = ['epoch', 'channel', 'label', 'trial_id']
feature_cols = [col for col in numerical_cols if col not in metadata_cols]
print(f"Number of features: {len(feature_cols)}")

# Define important channels based on previous analysis
important_channels = [61, 58, 2, 31, 23, 4, 10, 15, 30, 40]
print(f"Using top {len(important_channels)} important channels")

# Extract optimized features
def extract_optimized_features(df):
    """
    Extract features using important channels and apply PCA
    with optimal number of components
    """
    # Get unique trials
    unique_trials = sorted(df['trial_id'].unique())
    n_trials = len(unique_trials)
    
    print(f"Extracting optimized features for {n_trials} trials")
    
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
    pca = PCA(n_components=20)  # Optimal number based on previous analysis
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Explained variance with 20 components: {pca.explained_variance_ratio_.sum():.4f}")
    
    return X_pca, y_trials, X_scaled

# Extract optimized features
print("\nExtracting optimized features...")
X_optimized, y_optimized, X_scaled = extract_optimized_features(df)

print(f"Optimized feature shape: {X_optimized.shape}")
print(f"Number of trials: {len(y_optimized)}")
print(f"Class distribution: {np.bincount(y_optimized)}")

# Split data into training and testing sets
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_optimized, y_optimized, test_size=0.3, random_state=42, stratify=y_optimized
)

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_scaled, y_optimized, test_size=0.3, random_state=42, stratify=y_optimized
)

print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

# Convert to categorical for deep learning
y_train_cat = to_categorical(y_train, num_classes=4)
y_test_cat = to_categorical(y_test, num_classes=4)

# Create Deep PCA model
def create_deep_pca_model(input_shape):
    """
    Create a deep neural network for PCA-reduced features
    """
    model = Sequential([
        # Input layer
        Dense(128, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layers
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
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

# Define base models with hyperparameter tuning
print("\nDefining and tuning base models...")

# SVM with grid search
svm_params = {
    'C': [1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf', 'poly']
}
svm_grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    svm_params,
    cv=5,
    n_jobs=-1,
    verbose=0
)
svm_grid.fit(X_train, y_train)
print(f"Best SVM parameters: {svm_grid.best_params_}")
best_svm = svm_grid.best_estimator_

# Random Forest with grid search
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_params,
    cv=5,
    n_jobs=-1,
    verbose=0
)
rf_grid.fit(X_train, y_train)
print(f"Best RF parameters: {rf_grid.best_params_}")
best_rf = rf_grid.best_estimator_

# Gradient Boosting with grid search
gb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}
gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_params,
    cv=5,
    n_jobs=-1,
    verbose=0
)
gb_grid.fit(X_train, y_train)
print(f"Best GB parameters: {gb_grid.best_params_}")
best_gb = gb_grid.best_estimator_

# MLP with grid search
mlp_params = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}
mlp_grid = GridSearchCV(
    MLPClassifier(max_iter=1000, random_state=42),
    mlp_params,
    cv=5,
    n_jobs=-1,
    verbose=0
)
mlp_grid.fit(X_train, y_train)
print(f"Best MLP parameters: {mlp_grid.best_params_}")
best_mlp = mlp_grid.best_estimator_

# Train Deep PCA model
print("\nTraining Deep PCA model...")
deep_pca_model = create_deep_pca_model(X_train.shape[1])

# Define callbacks
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

# Train the model
history = deep_pca_model.fit(
    X_train, y_train_cat,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate Deep PCA model
deep_pca_preds_proba = deep_pca_model.predict(X_test)
deep_pca_preds = np.argmax(deep_pca_preds_proba, axis=1)
deep_pca_accuracy = accuracy_score(y_test, deep_pca_preds)
print(f"Deep PCA Model Accuracy: {deep_pca_accuracy:.4f}")

# Create a custom Deep PCA Classifier that works with scikit-learn
class DeepPCAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model=None):
        self.model = model
        
    def fit(self, X, y):
        # Already fitted in main code
        return self
    
    def predict(self, X):
        preds = self.model.predict(X)
        return np.argmax(preds, axis=1)
    
    def predict_proba(self, X):
        return self.model.predict(X)

# Create deep PCA classifier instance
deep_pca_clf = DeepPCAClassifier(model=deep_pca_model)

# Create combined stacking ensemble with traditional models and Deep PCA
print("\nCreating combined stacking ensemble...")
combined_stacking = StackingClassifier(
    estimators=[
        ('svm', best_svm),
        ('rf', best_rf),
        ('gb', best_gb),
        ('mlp', best_mlp),
        ('deep_pca', deep_pca_clf)
    ],
    final_estimator=RandomForestClassifier(n_estimators=200, random_state=42),
    cv=5
)

# Train combined stacking ensemble
print("Training combined stacking ensemble...")
combined_stacking.fit(X_train, y_train)

# Evaluate combined stacking ensemble
combined_preds = combined_stacking.predict(X_test)
combined_accuracy = accuracy_score(y_test, combined_preds)
print(f"Combined Stacking Ensemble Accuracy: {combined_accuracy:.4f}")

# Print classification report
print("\nClassification Report (Combined Stacking Ensemble):")
print(classification_report(y_test, combined_preds))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, combined_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Combined Stacking Ensemble')
plt.tight_layout()
plt.savefig('combined_stacking_confusion_matrix.png')
plt.close()

# Plot learning curves for Deep PCA model
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Deep PCA Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Deep PCA Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('deep_pca_learning_curves.png')
plt.close()

# Compare with previous approaches
print("\nComparing with previous approaches:")
print(f"PCA + Ensemble (20 components): 68.06%")
print(f"Optimized Stacking Ensemble: 72.22%")
print(f"Deep PCA Model: {deep_pca_accuracy:.4f}")
print(f"Combined Stacking Ensemble: {combined_accuracy:.4f}")

# Analyze per-class performance
class_report = classification_report(y_test, combined_preds, output_dict=True)
class_f1 = [class_report[str(i)]['f1-score'] for i in range(4)]

plt.figure(figsize=(10, 6))
plt.bar(['Class 0', 'Class 1', 'Class 2', 'Class 3'], class_f1, color='skyblue')
plt.axhline(y=0.5, color='r', linestyle='--', label='Baseline')
plt.xlabel('Class')
plt.ylabel('F1-Score')
plt.title('Per-Class Performance - Combined Stacking Ensemble')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig('combined_stacking_per_class_performance.png')
plt.close()

print("\nCombined stacking ensemble analysis completed. Visualization images saved.")
