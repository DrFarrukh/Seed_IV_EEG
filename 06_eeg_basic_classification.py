import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Load the data
print("Loading EEG dataset...")
file_path = "combined_reduced_epochs (1).csv"

# Read the first few rows to understand the structure
df_sample = pd.read_csv(file_path, nrows=5)
print(f"Columns in the dataset: {len(df_sample.columns)}")

# Load the full dataset
df = pd.read_csv(file_path)

# Sample the data for visualization (to avoid memory issues)
sample_size = min(50000, len(df))
df_viz = df.sample(sample_size, random_state=42)

# Display basic information
print("\n--- Basic Dataset Information ---")
print(f"Dataset shape: {df.shape}")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")

# Check for missing values
print("\n--- Missing Values ---")
missing_values = df.isnull().sum()
print(f"Total missing values: {missing_values.sum()}")
if missing_values.sum() > 0:
    print(missing_values[missing_values > 0])

# Display class distribution
print("\n--- Class Distribution ---")
class_counts = df['label'].value_counts().sort_index()
print(class_counts)

# Display basic statistics for numerical features
print("\n--- Feature Statistics ---")
# Select only numerical columns for statistics (excluding metadata columns)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
metadata_cols = ['epoch', 'channel', 'label', 'trial_id']
feature_cols = [col for col in numerical_cols if col not in metadata_cols]

# Display statistics for the first 5 features
print(df[feature_cols[:5]].describe())

# Visualize class distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='label', data=df)
plt.title('Class Distribution')
plt.xlabel('Class Label')
plt.ylabel('Count')
plt.savefig('class_distribution.png')
plt.close()

# Visualize distribution of a few key features using the sampled data
plt.figure(figsize=(15, 10))
for i, feature in enumerate(feature_cols[:6]):
    plt.subplot(2, 3, i+1)
    # Remove extreme outliers for visualization
    feature_data = df_viz[feature]
    q1, q3 = np.percentile(feature_data, [1, 99])
    feature_filtered = feature_data[(feature_data >= q1) & (feature_data <= q3)]
    sns.histplot(feature_filtered, kde=True, bins=30)
    plt.title(f'Distribution of {feature} (1-99 percentile)')
    plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.close()

# Correlation analysis
print("\n--- Correlation Analysis ---")
# Select a subset of features to avoid overwhelming correlation matrix
selected_features = feature_cols[:10] + ['label']
correlation = df[selected_features].corr()

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Selected Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Explore feature differences across classes using the sampled data
plt.figure(figsize=(15, 10))
for i, feature in enumerate(feature_cols[:6]):
    plt.subplot(2, 3, i+1)
    # Remove extreme outliers for visualization
    feature_data = df_viz.copy()
    q1, q3 = np.percentile(feature_data[feature], [1, 99])
    feature_data = feature_data[(feature_data[feature] >= q1) & (feature_data[feature] <= q3)]
    sns.boxplot(x='label', y=feature, data=feature_data)
    plt.title(f'{feature} by Class (1-99 percentile)')
    plt.tight_layout()
plt.savefig('features_by_class.png')
plt.close()

# PCA for dimensionality reduction and visualization
print("\n--- PCA Analysis ---")
# Use the already sampled data for PCA

# Handle potential NaN or infinite values
df_pca = df_viz.copy()
df_pca = df_pca.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)

# Limit to a reasonable number for PCA
sample_size_pca = min(10000, len(df_pca))
df_pca = df_pca.sample(sample_size_pca, random_state=42)

# Standardize the features
X = df_pca[feature_cols].values
y = df_pca['label'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for the PCA results
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['label'] = y

# Visualize PCA results
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='label', data=pca_df, palette='viridis', alpha=0.7)
plt.title('PCA: First Two Principal Components')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance explained)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance explained)')
plt.legend(title='Class')
plt.tight_layout()
plt.savefig('pca_visualization.png')
plt.close()

# Explore channel and epoch distribution
print("\n--- Channel and Epoch Analysis ---")
print("Channel distribution:")
print(df['channel'].value_counts().sort_index())

print("\nEpoch distribution:")
print(df['epoch'].value_counts().sort_index())

# Visualize feature importance by class
# Calculate mean values of features for each class
class_means = df.groupby('label')[feature_cols[:10]].mean()

# Plot heatmap of feature means by class
plt.figure(figsize=(14, 8))
sns.heatmap(class_means, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title('Mean Feature Values by Class')
plt.tight_layout()
plt.savefig('feature_means_by_class.png')
plt.close()

# Additional exploratory analysis
# Analyze distribution of power bands across classes (important for EEG)
power_bands = ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']
if all(band in df.columns for band in power_bands):
    print("\n--- EEG Power Band Analysis ---")
    
    # Create a plot for power bands by class
    plt.figure(figsize=(12, 8))
    power_means = df.groupby('label')[power_bands].mean()
    power_means.plot(kind='bar', figsize=(12, 6))
    plt.title('Mean Power Band Values by Class')
    plt.ylabel('Power')
    plt.xlabel('Class')
    plt.legend(title='Frequency Band')
    plt.tight_layout()
    plt.savefig('power_bands_by_class.png')
    plt.close()

# Check for feature importance using variance
print("\n--- Feature Variance Analysis ---")
feature_variance = df[feature_cols].var().sort_values(ascending=False)
print("Top 10 features by variance:")
print(feature_variance.head(10))

# Plot top features by variance
plt.figure(figsize=(12, 6))
feature_variance.head(10).plot(kind='bar')
plt.title('Top 10 Features by Variance')
plt.ylabel('Variance')
plt.xlabel('Features')
plt.tight_layout()
plt.savefig('top_features_by_variance.png')
plt.close()

print("\nData exploration completed. Visualization images saved.")
