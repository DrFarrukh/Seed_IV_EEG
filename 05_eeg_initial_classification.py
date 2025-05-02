import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

print("Loading EEG dataset...")
file_path = "combined_reduced_epochs (1).csv"

# Load a smaller sample for faster processing
sample_size = 50000
df = pd.read_csv(file_path)
df_sample = df.sample(sample_size, random_state=42)

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

# Prepare data for modeling
X = df_clean[feature_cols].values
y = df_clean['label'].values

# Standardize the features
print("\nStandardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Feature selection using Random Forest importance
print("\nPerforming feature selection...")
# Random Forest for feature importance
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_selector.fit(X_train, y_train)

# Get feature importances
importances = rf_selector.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.title('Feature Importances')
plt.bar(range(20), importances[indices[:20]], align='center')
plt.xticks(range(20), [feature_cols[i] for i in indices[:20]], rotation=90)
plt.tight_layout()
plt.savefig('feature_importances.png')
plt.close()

# Select top features
print("\nSelecting top features...")
n_top_features = 20  # Select top 20 features
top_indices = indices[:n_top_features]
top_features = [feature_cols[i] for i in top_indices]
print(f"Top {n_top_features} features: {top_features}")

# Create dataset with only top features
X_train_top = X_train[:, top_indices]
X_test_top = X_test[:, top_indices]

# Train Random Forest with top features
print("\nTraining Random Forest with top features...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train_top, y_train)
rf_pred = rf_model.predict(X_test_top)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest accuracy: {rf_accuracy:.4f}")

# Train Neural Network with top features
print("\nTraining Neural Network with top features...")
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, alpha=0.0001, 
                         solver='adam', random_state=42)
nn_model.fit(X_train_top, y_train)
nn_pred = nn_model.predict(X_test_top)
nn_accuracy = accuracy_score(y_test, nn_pred)
print(f"Neural Network accuracy: {nn_accuracy:.4f}")

# Print classification report for the best model
best_model = "Random Forest" if rf_accuracy > nn_accuracy else "Neural Network"
best_pred = rf_pred if rf_accuracy > nn_accuracy else nn_pred
best_accuracy = max(rf_accuracy, nn_accuracy)

print(f"\nBest model: {best_model} with accuracy: {best_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, best_pred))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Confusion Matrix - {best_model}')
plt.tight_layout()
plt.savefig('best_model_confusion_matrix.png')
plt.close()

# Channel-specific analysis if channel information is available
if 'channel' in df_clean.columns:
    print("\nPerforming channel-specific analysis...")
    # Analyze performance by channel
    channel_accuracies = {}
    
    for channel in sorted(df_clean['channel'].unique())[:10]:  # Analyze first 10 channels
        channel_data = df_clean[df_clean['channel'] == channel]
        if len(channel_data) < 100:  # Skip channels with too few samples
            continue
            
        X_channel = channel_data[feature_cols].values
        y_channel = channel_data['label'].values
        
        # Standardize
        X_channel_scaled = scaler.transform(X_channel)
        
        # Use only top features
        X_channel_top = X_channel_scaled[:, top_indices]
        
        # Split data
        if len(np.unique(y_channel)) < 2:
            continue  # Skip if only one class is present
            
        X_train_ch, X_test_ch, y_train_ch, y_test_ch = train_test_split(
            X_channel_top, y_channel, test_size=0.3, random_state=42, stratify=y_channel
        )
        
        # Train model
        if best_model == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        else:
            model = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=200, random_state=42)
            
        model.fit(X_train_ch, y_train_ch)
        y_pred_ch = model.predict(X_test_ch)
        accuracy = accuracy_score(y_test_ch, y_pred_ch)
        channel_accuracies[channel] = accuracy
    
    # Plot channel accuracies
    if channel_accuracies:
        plt.figure(figsize=(12, 6))
        channels = list(channel_accuracies.keys())
        accuracies = list(channel_accuracies.values())
        plt.bar(channels, accuracies)
        plt.xlabel('Channel')
        plt.ylabel('Accuracy')
        plt.title('Classification Accuracy by Channel')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig('channel_accuracies.png')
        plt.close()
        
        print(f"Channel accuracies: {channel_accuracies}")
        best_channel = max(channel_accuracies, key=channel_accuracies.get)
        print(f"Best performing channel: {best_channel} with accuracy: {channel_accuracies[best_channel]:.4f}")

# Analyze power band features if available
power_bands = ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']
if all(band in df_clean.columns for band in power_bands):
    print("\nAnalyzing EEG power bands...")
    
    # Plot mean power band values by class
    plt.figure(figsize=(12, 8))
    power_means = df_clean.groupby('label')[power_bands].mean()
    power_means.plot(kind='bar', figsize=(12, 6))
    plt.title('Mean Power Band Values by Class')
    plt.ylabel('Power')
    plt.xlabel('Class')
    plt.legend(title='Frequency Band')
    plt.tight_layout()
    plt.savefig('power_bands_by_class.png')
    plt.close()
    
    # Check if power bands are among top features
    power_band_importance = {band: importances[feature_cols.index(band)] for band in power_bands if band in feature_cols}
    print(f"Power band importance: {power_band_importance}")

print("\nQuick classification analysis completed. Visualization images saved.")
