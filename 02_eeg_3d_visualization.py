import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

print("Loading EEG dataset...")
file_path = "combined_reduced_epochs (1).csv"
df = pd.read_csv(file_path)

print(f"Dataset shape: {df.shape}")

# Verify the structure
print("\nVerifying dataset structure...")
print(f"Number of unique channels: {df['channel'].nunique()}")
print(f"Number of unique epochs: {df['epoch'].nunique()}")
print(f"Number of unique labels: {df['label'].nunique()}")
print(f"Number of unique trial_ids: {df['trial_id'].nunique()}")

# Count samples per channel and epoch
channel_counts = df['channel'].value_counts().sort_index()
epoch_counts = df['epoch'].value_counts().sort_index()

print(f"\nSamples per channel (first 5): {channel_counts.head()}")
print(f"Samples per epoch (first 5): {epoch_counts.head()}")

# Identify feature columns (excluding metadata)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
metadata_cols = ['epoch', 'channel', 'label', 'trial_id']
feature_cols = [col for col in numerical_cols if col not in metadata_cols]
print(f"\nNumber of features: {len(feature_cols)}")

# Reconstruct 3D structure for a single trial and label
print("\nReconstructing 3D structure for a sample...")

# Get a list of unique trial_ids
unique_trials = df['trial_id'].unique()
print(f"Number of unique trials: {len(unique_trials)}")

# Select a single trial for demonstration
sample_trial = unique_trials[0]
sample_label = 0  # Choose label 0 for this example

# Filter data for the selected trial and label
sample_data = df[(df['trial_id'] == sample_trial) & (df['label'] == sample_label)]
print(f"Sample data shape: {sample_data.shape}")

# Check if we have the expected number of rows (channels × epochs)
expected_rows = df['channel'].nunique() * df['epoch'].nunique()
print(f"Expected rows for a single trial: {expected_rows}")
print(f"Actual rows in sample: {len(sample_data)}")

# Reshape data into 3D format (channels × epochs × features)
# First, create a pivot table with channels as rows and epochs as columns
# For simplicity, we'll use just one feature for demonstration
feature_to_pivot = feature_cols[0]  # Use the first feature

try:
    pivot_data = sample_data.pivot(index='channel', columns='epoch', values=feature_to_pivot)
    print(f"\nPivot table shape for feature '{feature_to_pivot}': {pivot_data.shape}")
    
    # Visualize the pivot table as a heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_data, cmap='viridis')
    plt.title(f'Channel × Epoch Heatmap for {feature_to_pivot} (Label {sample_label})')
    plt.xlabel('Epoch')
    plt.ylabel('Channel')
    plt.tight_layout()
    plt.savefig('channel_epoch_heatmap.png')
    plt.close()
except Exception as e:
    print(f"Error creating pivot table: {e}")
    # If pivot fails, it might be because we don't have all channel-epoch combinations
    # Let's check the distribution
    channel_epoch_counts = sample_data.groupby(['channel', 'epoch']).size()
    print(f"Number of channel-epoch combinations: {len(channel_epoch_counts)}")
    print(f"Expected combinations: {df['channel'].nunique() * df['epoch'].nunique()}")
    
    # Create a cross-tabulation to see which combinations exist
    channel_epoch_matrix = pd.crosstab(sample_data['channel'], sample_data['epoch'])
    print("\nChannel-Epoch availability matrix (sample):")
    print(channel_epoch_matrix.iloc[:5, :5])  # Show just a sample

# Try a different approach - create 3D features for classification
print("\nCreating 3D feature representation for classification...")

# Function to extract features for a single trial
def extract_3d_features(trial_data):
    # Group by channel and compute statistics across epochs
    channel_features = trial_data.groupby('channel')[feature_cols].mean()
    
    # Flatten the channel features into a single vector
    flattened = channel_features.values.flatten()
    
    return flattened

# Sample a subset of trials for faster processing
unique_trials = df['trial_id'].unique()
sample_size = min(1000, len(unique_trials))
sampled_trials = np.random.choice(unique_trials, sample_size, replace=False)

# Create 3D features and labels
X_3d = []
y_3d = []

for trial in sampled_trials:
    trial_data = df[df['trial_id'] == trial]
    
    # Skip if trial doesn't have data
    if len(trial_data) == 0:
        continue
    
    # Get the label (should be the same for all rows in the trial)
    label = trial_data['label'].iloc[0]
    
    # Extract 3D features
    features = extract_3d_features(trial_data)
    
    X_3d.append(features)
    y_3d.append(label)

# Convert to numpy arrays
X_3d = np.array(X_3d)
y_3d = np.array(y_3d)

print(f"3D feature shape: {X_3d.shape}")
print(f"Number of labels: {len(y_3d)}")

# Check if we have enough samples for classification
if len(np.unique(y_3d)) > 1 and len(y_3d) > 20:
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_3d, y_3d, test_size=0.3, random_state=42, stratify=y_3d
    )
    
    # Train a classifier
    print("\nTraining classifier with 3D features...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
else:
    print("\nNot enough data for classification with 3D features")

# Visualize channel patterns by class
print("\nVisualizing channel patterns by class...")

# Power bands are important for EEG analysis
power_bands = ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']
if all(band in df.columns for band in power_bands):
    plt.figure(figsize=(15, 12))
    
    for i, band in enumerate(power_bands):
        plt.subplot(len(power_bands), 1, i+1)
        
        for label in sorted(df['label'].unique()):
            # Get data for this label
            label_data = df[df['label'] == label]
            
            # Calculate mean power by channel
            channel_means = label_data.groupby('channel')[band].mean()
            
            # Plot
            plt.plot(channel_means.index, channel_means.values, 
                     marker='o', label=f'Class {label}')
        
        plt.title(f'{band} by Channel and Class')
        plt.xlabel('Channel')
        plt.ylabel('Power')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('power_by_channel_class.png')
    plt.close()

# Visualize epoch patterns by class
print("\nVisualizing epoch patterns by class...")

if all(band in df.columns for band in power_bands):
    plt.figure(figsize=(15, 12))
    
    for i, band in enumerate(power_bands):
        plt.subplot(len(power_bands), 1, i+1)
        
        for label in sorted(df['label'].unique()):
            # Get data for this label
            label_data = df[df['label'] == label]
            
            # Calculate mean power by epoch
            epoch_means = label_data.groupby('epoch')[band].mean()
            
            # Plot
            plt.plot(epoch_means.index, epoch_means.values, 
                     marker='o', label=f'Class {label}')
        
        plt.title(f'{band} by Epoch and Class')
        plt.xlabel('Epoch')
        plt.ylabel('Power')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('power_by_epoch_class.png')
    plt.close()

print("\n3D structure analysis completed. Visualization images saved.")
