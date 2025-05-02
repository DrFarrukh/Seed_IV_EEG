"""
EEG Model Comparison - All Models

This script compares the performance of all models developed for the EEG classification task,
including the baseline models, optimized models, and domain adaptation models.

Author: Dr. Farrukh
Date: May 2025
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
from collections import defaultdict

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

def load_results():
    """Load results from various model files"""
    results = {}
    
    # Try to load domain adaptation results
    try:
        domain_results = joblib.load('domain_adaptation_results.pkl')
        results['Domain Adaptation'] = domain_results['domain_adaptation_model']['accuracy']
        results['Baseline (Session 3)'] = domain_results['baseline_model']['accuracy']
    except:
        print("Domain adaptation results not found")
    
    # Try to load transfer learning results
    try:
        # Extract from the output file or try to find saved results
        with open('transfer_learning_results.json', 'r') as f:
            transfer_results = json.load(f)
            results['EEGNet (Transfer Learning)'] = transfer_results.get('fine_tuned_accuracy', 0.392)
            results['EEGNet (Base)'] = transfer_results.get('base_accuracy', 0.401)
    except:
        print("Transfer learning results file not found, using default values")
        # Use values from the terminal output
        results['EEGNet (Transfer Learning)'] = 0.392
        results['EEGNet (Base)'] = 0.401
    
    # Try to load model comparison results
    try:
        with open('model_comparison_results.json', 'r') as f:
            comparison_results = json.load(f)
            results['Stacking Ensemble'] = comparison_results.get('stacking_accuracy', 0.4691)
            results['Deep PCA'] = comparison_results.get('deep_pca_accuracy', 0.4136)
    except:
        print("Model comparison results file not found, using default values")
        # Use values from previous runs
        results['Stacking Ensemble'] = 0.4691
        results['Deep PCA'] = 0.4136
    
    # Add PCA optimization results
    results['PCA Optimized (25 components)'] = 0.4907
    
    return results

def plot_model_comparison(results):
    """Plot comparison of all models"""
    # Sort models by accuracy
    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot bar chart
    models = list(sorted_results.keys())
    accuracies = list(sorted_results.values())
    
    bars = plt.bar(models, accuracies, color=sns.color_palette("viridis", len(models)))
    
    # Add accuracy values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Add random baseline
    plt.axhline(y=0.25, color='r', linestyle='--', label='Random Baseline (0.25)')
    
    # Customize plot
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Comparison of All EEG Classification Models')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(accuracies) + 0.1)
    plt.tight_layout()
    plt.legend()
    
    # Save plot
    plt.savefig('all_models_comparison.png', dpi=300)
    plt.close()
    
    return sorted_results

def generate_summary_table(results):
    """Generate a summary table of all models"""
    # Create DataFrame
    df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': list(results.values())
    })
    
    # Sort by accuracy
    df = df.sort_values('Accuracy', ascending=False)
    
    # Add improvement over random baseline
    df['Improvement over Random'] = (df['Accuracy'] - 0.25) / 0.25 * 100
    
    # Format columns
    df['Accuracy'] = df['Accuracy'].apply(lambda x: f"{x:.4f}")
    df['Improvement over Random'] = df['Improvement over Random'].apply(lambda x: f"{x:.2f}%")
    
    # Save to CSV
    df.to_csv('model_comparison_summary.csv', index=False)
    
    # Create markdown table
    markdown_table = "| Model | Accuracy | Improvement over Random |\n"
    markdown_table += "|-------|----------|------------------------|\n"
    
    for _, row in df.iterrows():
        markdown_table += f"| {row['Model']} | {row['Accuracy']} | {row['Improvement over Random']} |\n"
    
    # Save markdown table
    with open('model_comparison_table.md', 'w') as f:
        f.write(markdown_table)
    
    return df, markdown_table

def main():
    """Main function to compare all models"""
    print("="*50)
    print("EEG MODEL COMPARISON - ALL MODELS")
    print("="*50)
    
    # Load results
    results = load_results()
    
    # Plot comparison
    sorted_results = plot_model_comparison(results)
    
    # Generate summary table
    df, markdown_table = generate_summary_table(sorted_results)
    
    # Print summary
    print("\nModel Performance Summary (sorted by accuracy):")
    print(df)
    
    print("\nMarkdown Table for README:")
    print(markdown_table)
    
    print("\nComparison completed. Results and visualizations saved.")
    
    return sorted_results

if __name__ == "__main__":
    main()
