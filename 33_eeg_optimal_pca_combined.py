"""
EEG Optimal PCA Analysis for Combined Sessions

This script analyzes the optimal number of PCA components for the combined dataset
from all three EEG sessions. It evaluates:
1. Explained variance ratio
2. Classification performance vs. number of components
3. Computational efficiency tradeoffs

Author: Dr. Farrukh
Date: May 2025
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Define important channels based on previous analysis
IMPORTANT_CHANNELS = [61, 58, 2, 31, 23, 4, 10, 15, 30, 40]
RANDOM_STATE = 42
TEST_SIZE = 0.3

# Define session file paths
SESSION_FILES = {
    1: "combined_reduced_epochs_session_1.csv",
    2: "combined_reduced_epochs_session_2.csv",
    3: "combined_reduced_epochs_session_3.csv"
}

def load_session_data(session_id):
    """Load data for a specific session"""
    file_path = SESSION_FILES[session_id]
    print(f"Loading Session {session_id} data from {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Session {session_id} dataset shape: {df.shape}")
        
        # Add session identifier to trial_id
        df['trial_id'] = f'sess{session_id}_' + df['trial_id'].astype(str)
        df['session'] = session_id  # Add session column for tracking
        
        return df
    except Exception as e:
        print(f"Error loading Session {session_id}: {e}")
        return None

def extract_raw_features(df):
    """Extract raw features using important channels without PCA"""
    # Identify feature columns (excluding metadata)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metadata_cols = ['epoch', 'channel', 'label', 'trial_id', 'session']
    feature_cols = [col for col in numerical_cols if col not in metadata_cols]
    
    # Get unique trials
    unique_trials = sorted(df['trial_id'].unique())
    n_trials = len(unique_trials)
    
    print(f"Extracting features for {n_trials} trials")
    
    # Initialize arrays
    X_trials = []
    y_trials = []
    trial_ids = []
    sessions = []
    
    # Extract features for each trial
    for trial in unique_trials:
        # Get data for this trial
        trial_data = df[df['trial_id'] == trial]
        
        # Skip if trial doesn't have data
        if len(trial_data) == 0:
            continue
        
        # Get the label and session (should be the same for all rows in the trial)
        label = trial_data['label'].iloc[0]
        session = trial_data['session'].iloc[0]
        
        # Filter for important channels only
        trial_data = trial_data[trial_data['channel'].isin(IMPORTANT_CHANNELS)]
        
        # Extract features by averaging across epochs for each channel
        channel_features = trial_data.groupby('channel')[feature_cols].mean()
        
        # Flatten the channel features into a single vector
        features = channel_features.values.flatten()
        
        X_trials.append(features)
        y_trials.append(label)
        trial_ids.append(trial)
        sessions.append(session)
    
    # Convert to numpy arrays
    X_trials = np.array(X_trials)
    y_trials = np.array(y_trials)
    trial_ids = np.array(trial_ids)
    sessions = np.array(sessions)
    
    print(f"Raw feature shape: {X_trials.shape}")
    
    return X_trials, y_trials, trial_ids, sessions

def analyze_pca_components(X, y, max_components=100):
    """Analyze the optimal number of PCA components"""
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA with maximum number of components
    print(f"Applying PCA with up to {max_components} components...")
    pca = PCA(n_components=max_components)
    pca.fit(X_scaled)
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find number of components for different variance thresholds
    variance_thresholds = [0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
    components_for_threshold = {}
    
    for threshold in variance_thresholds:
        n_components = np.argmax(cumulative_variance >= threshold) + 1
        components_for_threshold[threshold] = n_components
        print(f"Components needed for {threshold*100:.0f}% variance: {n_components}")
    
    # Plot explained variance
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-', marker='o', markersize=5)
    
    # Add markers for thresholds
    for threshold in variance_thresholds:
        n_comp = components_for_threshold[threshold]
        plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.3)
        plt.axvline(x=n_comp, color='g', linestyle='--', alpha=0.3)
        plt.plot(n_comp, threshold, 'ro')
        plt.annotate(f"{n_comp} components\n({threshold*100:.0f}% variance)", 
                    xy=(n_comp, threshold),
                    xytext=(n_comp+5, threshold-0.05),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance vs. Number of Components')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pca_explained_variance_combined.png')
    plt.close()
    
    # Plot individual explained variance (scree plot)
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.7)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot - Individual Component Contribution')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pca_scree_plot_combined.png')
    plt.close()
    
    return pca, components_for_threshold, cumulative_variance

def evaluate_classification_performance(X, y, pca, components_for_threshold):
    """Evaluate classification performance for different numbers of PCA components"""
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define component ranges to test
    component_ranges = sorted(list(components_for_threshold.values()) + [5, 10, 15, 20, 25, 30, 40, 50])
    component_ranges = sorted(list(set([c for c in component_ranges if c <= X.shape[1]])))
    
    # Initialize results
    accuracy_results = []
    training_times = []
    inference_times = []
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Evaluate for each number of components
    for n_components in component_ranges:
        print(f"Evaluating with {n_components} components...")
        
        # Apply PCA with n_components
        pca_n = PCA(n_components=n_components)
        X_train_pca = pca_n.fit_transform(X_train)
        X_test_pca = pca_n.transform(X_test)
        
        # Train and evaluate with Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        
        # Measure training time
        start_time = time.time()
        rf.fit(X_train_pca, y_train)
        training_time = time.time() - start_time
        
        # Measure inference time
        start_time = time.time()
        y_pred = rf.predict(X_test_pca)
        inference_time = time.time() - start_time
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store results
        accuracy_results.append(accuracy)
        training_times.append(training_time)
        inference_times.append(inference_time)
        
        print(f"  Accuracy: {accuracy:.4f}, Training time: {training_time:.4f}s, Inference time: {inference_time:.4f}s")
    
    # Plot accuracy vs. number of components
    plt.figure(figsize=(12, 6))
    plt.plot(component_ranges, accuracy_results, 'b-', marker='o')
    
    # Find best accuracy
    best_idx = np.argmax(accuracy_results)
    best_components = component_ranges[best_idx]
    best_accuracy = accuracy_results[best_idx]
    
    plt.axhline(y=best_accuracy, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=best_components, color='g', linestyle='--', alpha=0.5)
    plt.plot(best_components, best_accuracy, 'ro', markersize=10)
    plt.annotate(f"Best: {best_components} components\n({best_accuracy:.4f} accuracy)", 
                xy=(best_components, best_accuracy),
                xytext=(best_components+10, best_accuracy-0.05),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Classification Accuracy')
    plt.title('Classification Accuracy vs. Number of PCA Components')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pca_accuracy_combined.png')
    plt.close()
    
    # Plot training and inference times
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(component_ranges, training_times, 'g-', marker='o')
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time vs. Number of Components')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(component_ranges, inference_times, 'r-', marker='o')
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Inference Time (seconds)')
    plt.title('Inference Time vs. Number of Components')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('pca_computation_time_combined.png')
    plt.close()
    
    return component_ranges, accuracy_results, training_times, inference_times, best_components, best_accuracy

def cross_validate_optimal_components(X, y, optimal_components_list, n_folds=5):
    """Cross-validate classification performance for selected numbers of components"""
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    max_components = max(optimal_components_list)
    pca = PCA(n_components=max_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Initialize results
    cv_results = {}
    
    # Evaluate each number of components with cross-validation
    for n_components in optimal_components_list:
        print(f"Cross-validating with {n_components} components...")
        
        # Extract features
        X_subset = X_pca[:, :n_components]
        
        # Perform cross-validation with Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        scores = cross_val_score(rf, X_subset, y, cv=n_folds, scoring='accuracy')
        
        cv_results[n_components] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        
        print(f"  CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Plot cross-validation results
    plt.figure(figsize=(12, 6))
    
    components = list(cv_results.keys())
    means = [cv_results[c]['mean'] for c in components]
    stds = [cv_results[c]['std'] for c in components]
    
    plt.errorbar(components, means, yerr=stds, fmt='o-', capsize=5)
    
    # Find best cross-validated result
    best_idx = np.argmax(means)
    best_components = components[best_idx]
    best_cv_accuracy = means[best_idx]
    
    plt.axhline(y=best_cv_accuracy, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=best_components, color='g', linestyle='--', alpha=0.5)
    plt.plot(best_components, best_cv_accuracy, 'ro', markersize=10)
    plt.annotate(f"Best CV: {best_components} components\n({best_cv_accuracy:.4f} ± {stds[best_idx]:.4f})", 
                xy=(best_components, best_cv_accuracy),
                xytext=(best_components+5, best_cv_accuracy-0.05),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title(f'{n_folds}-Fold Cross-Validation Accuracy vs. Number of Components')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pca_cross_validation_combined.png')
    plt.close()
    
    return cv_results, best_components, best_cv_accuracy

def main():
    """Main function to analyze optimal PCA components"""
    print("="*50)
    print("EEG OPTIMAL PCA ANALYSIS FOR COMBINED SESSIONS")
    print("="*50)
    
    # Load data from all sessions
    all_dfs = []
    for session_id in SESSION_FILES.keys():
        df = load_session_data(session_id)
        if df is not None:
            all_dfs.append(df)
    
    # Combine all session data
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Number of unique trials: {combined_df['trial_id'].nunique()}")
    print(f"Number of unique labels: {combined_df['label'].nunique()}")
    
    # Extract raw features
    X, y, trial_ids, sessions = extract_raw_features(combined_df)
    
    # Analyze PCA components
    print("\nAnalyzing PCA components...")
    pca, components_for_threshold, cumulative_variance = analyze_pca_components(X, y)
    
    # Evaluate classification performance
    print("\nEvaluating classification performance...")
    component_ranges, accuracy_results, training_times, inference_times, best_components, best_accuracy = \
        evaluate_classification_performance(X, y, pca, components_for_threshold)
    
    # Cross-validate selected component numbers
    print("\nCross-validating selected component numbers...")
    # Select components to cross-validate (best from accuracy, plus components at key variance thresholds)
    optimal_components_list = [best_components]
    for threshold in [0.7, 0.8, 0.85, 0.9]:
        if components_for_threshold[threshold] not in optimal_components_list:
            optimal_components_list.append(components_for_threshold[threshold])
    
    # Also include standard values used in previous analyses
    for n in [20, 30, 40, 50]:
        if n not in optimal_components_list and n <= X.shape[1]:
            optimal_components_list.append(n)
    
    optimal_components_list.sort()
    
    cv_results, best_cv_components, best_cv_accuracy = \
        cross_validate_optimal_components(X, y, optimal_components_list)
    
    # Print summary of results
    print("\n" + "="*50)
    print("OPTIMAL PCA COMPONENTS SUMMARY")
    print("="*50)
    
    print("\nExplained Variance Thresholds:")
    for threshold, n_components in components_for_threshold.items():
        print(f"  {threshold*100:.0f}% variance: {n_components} components")
    
    print("\nClassification Performance:")
    print(f"  Best accuracy: {best_accuracy:.4f} with {best_components} components")
    
    print("\nCross-Validation Results:")
    print(f"  Best CV accuracy: {best_cv_accuracy:.4f} with {best_cv_components} components")
    
    print("\nRecommended number of PCA components:")
    # Consider both accuracy and variance explained
    if best_cv_components == best_components:
        recommended = best_components
        reason = "highest accuracy in both direct testing and cross-validation"
    else:
        # Find a balance between accuracy and variance
        for threshold in [0.8, 0.85, 0.9]:
            if components_for_threshold[threshold] in [best_components, best_cv_components]:
                recommended = components_for_threshold[threshold]
                reason = f"good balance between accuracy and variance explained ({threshold*100:.0f}%)"
                break
        else:
            # If no clear choice, prefer cross-validation result
            recommended = best_cv_components
            reason = "highest cross-validation accuracy"
    
    print(f"  {recommended} components - {reason}")
    print("="*50)
    
    # Save key results to a file
    with open('optimal_pca_combined_results.txt', 'w') as f:
        f.write("OPTIMAL PCA COMPONENTS FOR COMBINED SESSIONS\n")
        f.write("="*50 + "\n\n")
        
        f.write("Explained Variance Thresholds:\n")
        for threshold, n_components in components_for_threshold.items():
            f.write(f"  {threshold*100:.0f}% variance: {n_components} components\n")
        
        f.write("\nClassification Performance:\n")
        f.write(f"  Best accuracy: {best_accuracy:.4f} with {best_components} components\n")
        
        f.write("\nCross-Validation Results:\n")
        f.write(f"  Best CV accuracy: {best_cv_accuracy:.4f} with {best_cv_components} components\n")
        
        f.write("\nRecommended number of PCA components:\n")
        f.write(f"  {recommended} components - {reason}\n")
    
    print(f"Results saved to 'optimal_pca_combined_results.txt'")
    
    return {
        'components_for_threshold': components_for_threshold,
        'best_components': best_components,
        'best_accuracy': best_accuracy,
        'best_cv_components': best_cv_components,
        'best_cv_accuracy': best_cv_accuracy,
        'recommended': recommended
    }

if __name__ == "__main__":
    main()
