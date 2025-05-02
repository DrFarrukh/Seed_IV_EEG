import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Define the results from all approaches
approaches = [
    "Flattened Data", 
    "Trial-based",
    "PCA (20 components)",
    "PCA (94 components)",
    "PCA (100 components)",
    "Channel-CNN",
    "Epoch-LSTM",
    "EEGNet-inspired",
    "Hybrid Model",
    "Specialized Ensemble",
    "Balanced Final Model",
    "Super Learner",
    "Advanced Ensemble",
    "Optimized Stacking"
]

accuracies = [
    34.0,  # Flattened Data
    67.5,  # Trial-based (midpoint of 65-70%)
    68.06, # PCA (20 components)
    67.59, # PCA (94 components)
    68.52, # PCA (100 components)
    25.00, # Channel-CNN
    42.59, # Epoch-LSTM
    49.07, # EEGNet-inspired
    63.89, # Hybrid Model
    66.67, # Specialized Ensemble
    66.67, # Balanced Final Model
    62.96, # Super Learner
    30.56, # Advanced Ensemble
    72.22  # Optimized Stacking
]

feature_counts = [
    3038, # Flattened Data
    3038, # Trial-based
    20,   # PCA (20 components)
    94,   # PCA (94 components)
    100,  # PCA (100 components)
    3038, # Channel-CNN (3D structure)
    3038, # Epoch-LSTM (3D structure)
    3038, # EEGNet-inspired (3D structure)
    522,  # Hybrid Model
    50,   # Specialized Ensemble
    30,   # Balanced Final Model
    522,  # Super Learner (combined approaches)
    490,  # Advanced Ensemble
    20    # Optimized Stacking
]

# Create a DataFrame for easier manipulation
results_df = pd.DataFrame({
    'Approach': approaches,
    'Accuracy': accuracies,
    'Features': feature_counts
})

# Sort by accuracy for the bar chart
sorted_df = results_df.sort_values('Accuracy', ascending=False)

# Create a color palette that highlights the best approach
colors = ['#3498db' if approach != 'Optimized Stacking' else '#e74c3c' 
          for approach in sorted_df['Approach']]

# Plot accuracy comparison
plt.figure(figsize=(14, 8))
bars = plt.bar(sorted_df['Approach'], sorted_df['Accuracy'], color=colors)
plt.axhline(y=25.0, color='r', linestyle='--', label='Random Chance (25%)')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Approach')
plt.ylabel('Accuracy (%)')
plt.title('EEG Classification Performance Across Different Approaches')
plt.ylim(0, 100)
plt.tight_layout()
plt.legend()
plt.savefig('accuracy_comparison.png')
plt.close()

# Create a scatter plot to visualize the trade-off between accuracy and feature count
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    results_df['Features'], 
    results_df['Accuracy'], 
    c=results_df['Accuracy'], 
    cmap='viridis', 
    s=100, 
    alpha=0.7
)

# Add labels for each point
for i, approach in enumerate(results_df['Approach']):
    plt.annotate(
        approach, 
        (results_df['Features'][i], results_df['Accuracy'][i]),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=9
    )

plt.xscale('log')  # Use log scale for feature count due to wide range
plt.xlabel('Number of Features (log scale)')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. Feature Count: Finding the Optimal Balance')
plt.axhline(y=25.0, color='r', linestyle='--', label='Random Chance (25%)')
plt.colorbar(scatter, label='Accuracy (%)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_vs_features.png')
plt.close()

# Create a grouped bar chart for class-specific performance
# Data from the optimized stacking ensemble
classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
precision = [0.62, 0.77, 0.88, 0.62]
recall = [0.56, 0.89, 0.78, 0.67]
f1_score = [0.59, 0.83, 0.82, 0.64]

x = np.arange(len(classes))
width = 0.25

plt.figure(figsize=(12, 8))
plt.bar(x - width, precision, width, label='Precision', color='#3498db')
plt.bar(x, recall, width, label='Recall', color='#2ecc71')
plt.bar(x + width, f1_score, width, label='F1-Score', color='#e74c3c')

plt.axhline(y=0.5, color='k', linestyle='--', label='Baseline')
plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Class-Specific Performance Metrics (Optimized Stacking Ensemble)')
plt.xticks(x, classes)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig('class_specific_performance.png')
plt.close()

# Create a summary figure showing the progression of approaches
# Select key approaches in chronological order
key_approaches = [
    "Flattened Data", 
    "Trial-based",
    "PCA (20 components)",
    "EEGNet-inspired",
    "Specialized Ensemble",
    "Optimized Stacking"
]

key_accuracies = [acc for app, acc in zip(approaches, accuracies) if app in key_approaches]

plt.figure(figsize=(12, 6))
plt.plot(key_approaches, key_accuracies, marker='o', linestyle='-', linewidth=2, markersize=10, color='#3498db')
for i, (app, acc) in enumerate(zip(key_approaches, key_accuracies)):
    plt.annotate(f"{acc:.2f}%", (i, acc), xytext=(0, 10), textcoords='offset points', ha='center')

plt.axhline(y=25.0, color='r', linestyle='--', label='Random Chance (25%)')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Approach (Chronological Order)')
plt.ylabel('Accuracy (%)')
plt.title('Progression of EEG Classification Performance')
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('performance_progression.png')
plt.close()

print("Visualization summary completed. All images saved.")
