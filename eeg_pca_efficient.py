import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

print("Loading EEG dataset...")
file_path = "combined_reduced_epochs (1).csv"

# Load a sample of the data for faster processing
# Adjust sample_size based on your system's memory capacity
sample_size = 50000  # Using a sample for faster processing
df = pd.read_csv(file_path)
df_sample = df.sample(sample_size, random_state=42)

print(f"Full dataset shape: {df.shape}")
print(f"Sample dataset shape: {df_sample.shape}")

# Identify feature columns (excluding metadata)
numerical_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
metadata_cols = ['epoch', 'channel', 'label', 'trial_id']
feature_cols = [col for col in numerical_cols if col not in metadata_cols]
print(f"Number of features: {len(feature_cols)}")

# Handle potential outliers and infinite values
df_clean = df_sample.replace([np.inf, -np.inf], np.nan)
df_clean = df_clean.dropna(subset=feature_cols)
print(f"Shape after cleaning: {df_clean.shape}")

# Prepare data for PCA
X = df_clean[feature_cols].values
y = df_clean['label'].values

# Standardize the features
print("\nStandardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
print("\nApplying PCA...")
# First, determine optimal number of components
pca_full = PCA()
pca_full.fit(X_scaled)

# Plot explained variance ratio
plt.figure(figsize=(12, 6))
explained_variance = np.cumsum(pca_full.explained_variance_ratio_)
plt.plot(explained_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Components')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
plt.axhline(y=0.99, color='g', linestyle='--', label='99% Explained Variance')
plt.legend()
plt.grid(True)
plt.savefig('pca_explained_variance.png')
plt.close()

# Find number of components for 95% and 99% variance explained
n_components_95 = np.argmax(explained_variance >= 0.95) + 1
n_components_99 = np.argmax(explained_variance >= 0.99) + 1

print(f"Number of components for 95% variance: {n_components_95}")
print(f"Number of components for 99% variance: {n_components_99}")

# Apply PCA with the number of components for 95% variance
pca_95 = PCA(n_components=n_components_95)
X_pca_95 = pca_95.fit_transform(X_scaled)

# Apply PCA with just 2 components for visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# Apply PCA with 3 components for 3D visualization
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

# Create DataFrames for visualization
pca_df_2d = pd.DataFrame(data=X_pca_2d, columns=['PC1', 'PC2'])
pca_df_2d['label'] = y

pca_df_3d = pd.DataFrame(data=X_pca_3d, columns=['PC1', 'PC2', 'PC3'])
pca_df_3d['label'] = y

# Visualize 2D PCA
plt.figure(figsize=(12, 10))
sns.scatterplot(x='PC1', y='PC2', hue='label', data=pca_df_2d, palette='viridis', alpha=0.6, s=30)
plt.title('PCA: First Two Principal Components')
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance explained)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance explained)')
plt.legend(title='Class')
plt.tight_layout()
plt.savefig('pca_2d_visualization.png')
plt.close()

# Visualize 3D PCA using multiple 2D plots
fig = plt.figure(figsize=(18, 6))

# PC1 vs PC2
ax1 = fig.add_subplot(131)
sns.scatterplot(x='PC1', y='PC2', hue='label', data=pca_df_3d, palette='viridis', alpha=0.6, s=30, ax=ax1)
ax1.set_title('PC1 vs PC2')
ax1.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})')
ax1.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})')

# PC1 vs PC3
ax2 = fig.add_subplot(132)
sns.scatterplot(x='PC1', y='PC3', hue='label', data=pca_df_3d, palette='viridis', alpha=0.6, s=30, ax=ax2)
ax2.set_title('PC1 vs PC3')
ax2.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})')
ax2.set_ylabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})')

# PC2 vs PC3
ax3 = fig.add_subplot(133)
sns.scatterplot(x='PC2', y='PC3', hue='label', data=pca_df_3d, palette='viridis', alpha=0.6, s=30, ax=ax3)
ax3.set_title('PC2 vs PC3')
ax3.set_xlabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})')
ax3.set_ylabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})')

plt.suptitle('PCA: First Three Principal Components', fontsize=16)
plt.tight_layout()
plt.savefig('pca_3d_visualization.png')
plt.close()

# Analyze feature contributions to principal components
print("\nAnalyzing feature contributions to principal components...")
components_df = pd.DataFrame(pca_2d.components_.T, columns=['PC1', 'PC2'], index=feature_cols)

# Plot feature loadings for PC1 and PC2
plt.figure(figsize=(14, 10))
loading_scores = pd.DataFrame(
    data=pca_2d.components_.T * np.sqrt(pca_2d.explained_variance_), 
    columns=['PC1', 'PC2'],
    index=feature_cols
)

# Sort by absolute contribution to PC1
loading_scores_sorted = loading_scores.abs().sort_values(by='PC1', ascending=False)
top_features = loading_scores_sorted.index[:15]  # Top 15 features

# Plot the loadings of top features
loading_scores.loc[top_features].plot(kind='barh', figsize=(14, 10))
plt.title('Feature Contributions to Principal Components')
plt.xlabel('Loading Score')
plt.tight_layout()
plt.savefig('pca_feature_contributions.png')
plt.close()

# Evaluate classification performance using PCA features
print("\nEvaluating classification performance with PCA features...")
# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_pca_95, y, test_size=0.3, random_state=42, stratify=y
)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nClassification Accuracy with {n_components_95} PCA components: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('pca_confusion_matrix.png')
plt.close()

# Compare with original features
print("\nComparing with original features...")
# Split the data
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Train a Random Forest classifier
clf_orig = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf_orig.fit(X_train_orig, y_train_orig)

# Make predictions
y_pred_orig = clf_orig.predict(X_test_orig)

# Evaluate the model
accuracy_orig = accuracy_score(y_test_orig, y_pred_orig)
print(f"Classification Accuracy with original features: {accuracy_orig:.4f}")

print("\nPCA analysis completed. Visualization images saved.")
