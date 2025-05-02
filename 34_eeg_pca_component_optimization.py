"""
EEG PCA Component Optimization

This script optimizes the number of PCA components for the combined dataset
by training multiple classifier types and evaluating test accuracy.

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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import joblib
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
    # Filter out metadata columns to get only feature columns
    feature_cols = [col for col in numerical_cols if col not in metadata_cols]
    print(f"Number of features: {len(feature_cols)}")
    
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

def optimize_pca_components(X, y, component_range):
    """Optimize PCA components by training multiple classifier types"""
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
    
    # Initialize results dictionary
    results = {
        'component_counts': component_range,
        'svm_accuracy': [],
        'rf_accuracy': [],
        'gb_accuracy': [],
        'mlp_accuracy': [],
        'voting_accuracy': [],
        'best_model_type': [],
        'best_accuracy': [],
        'training_time': [],
        'inference_time': []
    }
    
    # Test each component count
    for n_components in component_range:
        print(f"\nTesting with {n_components} PCA components...")
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        # Initialize models
        svm = SVC(probability=True, random_state=RANDOM_STATE)
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
        mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=RANDOM_STATE)
        
        # Train models and measure time
        start_time = time.time()
        
        svm.fit(X_train_pca, y_train)
        rf.fit(X_train_pca, y_train)
        gb.fit(X_train_pca, y_train)
        mlp.fit(X_train_pca, y_train)
        
        training_time = time.time() - start_time
        
        # Create voting ensemble
        voting = VotingClassifier(
            estimators=[
                ('svm', svm),
                ('rf', rf),
                ('gb', gb),
                ('mlp', mlp)
            ],
            voting='soft'
        )
        voting.fit(X_train_pca, y_train)
        
        # Evaluate models
        start_time = time.time()
        
        svm_pred = svm.predict(X_test_pca)
        rf_pred = rf.predict(X_test_pca)
        gb_pred = gb.predict(X_test_pca)
        mlp_pred = mlp.predict(X_test_pca)
        voting_pred = voting.predict(X_test_pca)
        
        inference_time = time.time() - start_time
        
        # Calculate accuracies
        svm_acc = accuracy_score(y_test, svm_pred)
        rf_acc = accuracy_score(y_test, rf_pred)
        gb_acc = accuracy_score(y_test, gb_pred)
        mlp_acc = accuracy_score(y_test, mlp_pred)
        voting_acc = accuracy_score(y_test, voting_pred)
        
        # Find best model
        model_accs = [svm_acc, rf_acc, gb_acc, mlp_acc, voting_acc]
        model_names = ['SVM', 'Random Forest', 'Gradient Boosting', 'Neural Network', 'Voting Ensemble']
        best_idx = np.argmax(model_accs)
        best_model = model_names[best_idx]
        best_acc = model_accs[best_idx]
        
        # Store results
        results['svm_accuracy'].append(svm_acc)
        results['rf_accuracy'].append(rf_acc)
        results['gb_accuracy'].append(gb_acc)
        results['mlp_accuracy'].append(mlp_acc)
        results['voting_accuracy'].append(voting_acc)
        results['best_model_type'].append(best_model)
        results['best_accuracy'].append(best_acc)
        results['training_time'].append(training_time)
        results['inference_time'].append(inference_time)
        
        print(f"SVM Accuracy: {svm_acc:.4f}")
        print(f"Random Forest Accuracy: {rf_acc:.4f}")
        print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")
        print(f"Neural Network Accuracy: {mlp_acc:.4f}")
        print(f"Voting Ensemble Accuracy: {voting_acc:.4f}")
        print(f"Best Model: {best_model} ({best_acc:.4f})")
        print(f"Training Time: {training_time:.4f}s, Inference Time: {inference_time:.4f}s")
        
        # If this is the best component count so far, save detailed results
        if best_acc == max(results['best_accuracy']):
            print("New best performance! Saving detailed results...")
            
            # Save confusion matrix
            plt.figure(figsize=(10, 8))
            # Map model name to variable name
            model_var_map = {
                'SVM': 'svm_pred',
                'Random Forest': 'rf_pred',
                'Gradient Boosting': 'gb_pred',
                'Neural Network': 'mlp_pred',
                'Voting Ensemble': 'voting_pred'
            }
            best_pred = locals()[model_var_map[best_model]]
            cm = confusion_matrix(y_test, best_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title(f'Confusion Matrix - {best_model} with {n_components} PCA Components')
            plt.tight_layout()
            plt.savefig(f'pca_opt_cm_{n_components}_components.png')
            plt.close()
            
            # Save classification report
            report = classification_report(y_test, best_pred, output_dict=True)
            with open(f'pca_opt_report_{n_components}_components.txt', 'w') as f:
                f.write(f"Classification Report - {best_model} with {n_components} PCA Components\n\n")
                f.write(classification_report(y_test, best_pred))
    
    # Find overall best performance
    best_idx = np.argmax(results['best_accuracy'])
    best_components = results['component_counts'][best_idx]
    best_model_type = results['best_model_type'][best_idx]
    best_accuracy = results['best_accuracy'][best_idx]
    
    print(f"\nOverall Best Performance:")
    print(f"Components: {best_components}")
    print(f"Model: {best_model_type}")
    print(f"Accuracy: {best_accuracy:.4f}")
    
    return results, best_components, best_model_type, best_accuracy

def visualize_results(results):
    """Visualize PCA optimization results"""
    component_counts = results['component_counts']
    
    # Plot accuracy by model type
    plt.figure(figsize=(14, 8))
    plt.plot(component_counts, results['svm_accuracy'], 'o-', label='SVM')
    plt.plot(component_counts, results['rf_accuracy'], 'o-', label='Random Forest')
    plt.plot(component_counts, results['gb_accuracy'], 'o-', label='Gradient Boosting')
    plt.plot(component_counts, results['mlp_accuracy'], 'o-', label='Neural Network')
    plt.plot(component_counts, results['voting_accuracy'], 'o-', label='Voting Ensemble')
    plt.plot(component_counts, results['best_accuracy'], 'ko-', linewidth=2, label='Best Model')
    
    # Find best point
    best_idx = np.argmax(results['best_accuracy'])
    best_components = component_counts[best_idx]
    best_accuracy = results['best_accuracy'][best_idx]
    
    plt.axvline(x=best_components, color='r', linestyle='--', alpha=0.5)
    plt.plot(best_components, best_accuracy, 'ro', markersize=10)
    plt.annotate(f"Best: {best_components} components\n({best_accuracy:.4f} accuracy)", 
                xy=(best_components, best_accuracy),
                xytext=(best_components+5, best_accuracy-0.05),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Test Accuracy')
    plt.title('Model Performance vs. Number of PCA Components')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pca_optimization_accuracy.png')
    plt.close()
    
    # Plot best model type by component count
    plt.figure(figsize=(14, 6))
    
    # Create a categorical color map
    model_types = ['SVM', 'Random Forest', 'Gradient Boosting', 'Neural Network', 'Voting Ensemble']
    model_colors = {'SVM': 'blue', 'Random Forest': 'green', 'Gradient Boosting': 'orange', 
                   'Neural Network': 'purple', 'Voting Ensemble': 'red'}
    
    # Create bar colors based on model type
    bar_colors = [model_colors[model] for model in results['best_model_type']]
    
    bars = plt.bar(component_counts, results['best_accuracy'], color=bar_colors)
    
    # Add value labels on top of bars
    for bar, acc in zip(bars, results['best_accuracy']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Create legend handles
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=model_colors[model], label=model) for model in model_types]
    plt.legend(handles=legend_elements)
    
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Best Accuracy')
    plt.title('Best Model Type and Accuracy by PCA Component Count')
    plt.tight_layout()
    plt.savefig('pca_optimization_best_models.png')
    plt.close()
    
    # Plot computation time
    plt.figure(figsize=(12, 6))
    plt.plot(component_counts, results['training_time'], 'bo-', label='Training Time')
    plt.plot(component_counts, results['inference_time'], 'ro-', label='Inference Time')
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Time (seconds)')
    plt.title('Computation Time vs. Number of PCA Components')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pca_optimization_time.png')
    plt.close()

def main():
    """Main function to optimize PCA components"""
    print("="*50)
    print("EEG PCA COMPONENT OPTIMIZATION")
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
    
    # Define PCA component range to test
    component_range = [5, 10, 15, 20, 22, 25, 30, 35, 40, 45, 50]
    
    # Optimize PCA components
    results, best_components, best_model_type, best_accuracy = optimize_pca_components(X, y, component_range)
    
    # Visualize results
    visualize_results(results)
    
    # Save results
    joblib.dump(results, 'pca_optimization_results.pkl')
    
    # Print summary
    print("\n" + "="*50)
    print("PCA COMPONENT OPTIMIZATION SUMMARY")
    print("="*50)
    print(f"Best Number of Components: {best_components}")
    print(f"Best Model Type: {best_model_type}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print("="*50)
    
    print("PCA component optimization completed. Results and visualizations saved.")
    
    return results, best_components, best_model_type, best_accuracy

if __name__ == "__main__":
    main()
