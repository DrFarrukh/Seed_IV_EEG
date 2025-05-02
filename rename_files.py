"""
File Renaming Script for EEG Project

This script renames all Python files in the EEG project with a numbered prefix
to make it easier to track their purpose and progression.

Categories:
- 01-05: Data preparation and exploration
- 06-10: Basic classification
- 11-15: PCA analysis and optimization
- 16-20: Ensemble methods
- 21-25: Advanced models
- 26-30: Combined sessions and specialized approaches
"""

import os
import shutil

# Define the mapping of old filenames to new numbered filenames
file_mapping = {
    # Data preparation and exploration (01-05)
    "eeg_pca_analysis.py": "01_eeg_data_exploration.py",
    "eeg_3d_analysis.py": "02_eeg_3d_visualization.py",
    "eeg_visualization_summary.py": "03_eeg_summary_visualization.py",
    "eeg_domain_features.py": "04_eeg_feature_engineering.py",
    "eeg_quick_classification.py": "05_eeg_initial_classification.py",
    
    # Basic classification (06-10)
    "eeg_classification.py": "06_eeg_basic_classification.py",
    "eeg_trial_classification.py": "07_eeg_trial_based_classification.py",
    "eeg_advanced_classification.py": "08_eeg_advanced_classification.py",
    "eeg_balanced_final.py": "09_eeg_balanced_model.py",
    "eeg_final_model.py": "10_eeg_baseline_model.py",
    
    # PCA analysis and optimization (11-15)
    "eeg_pca_efficient.py": "11_eeg_pca_implementation.py",
    "eeg_trial_pca_classification.py": "12_eeg_trial_pca_classification.py",
    "eeg_trial_pca_fixed.py": "13_eeg_trial_pca_improved.py",
    "eeg_optimal_pca.py": "14_eeg_optimal_pca_components.py",
    "eeg_deep_pca_final.py": "15_eeg_deep_pca_model.py",
    
    # Ensemble methods (16-20)
    "eeg_ensemble_optimal.py": "16_eeg_basic_ensemble.py",
    "eeg_specialized_ensemble.py": "17_eeg_specialized_ensemble.py",
    "eeg_super_learner.py": "18_eeg_super_learner.py",
    "eeg_advanced_ensemble.py": "19_eeg_advanced_ensemble.py",
    "eeg_advanced_ensemble_fixed.py": "20_eeg_advanced_ensemble_improved.py",
    
    # Advanced models (21-25)
    "eeg_deep_stacking_ensemble.py": "21_eeg_deep_stacking.py",
    "eeg_deep_stacking_fixed.py": "22_eeg_deep_stacking_improved.py",
    "eeg_recurrent_stacking.py": "23_eeg_recurrent_model.py",
    "eeg_hybrid_model.py": "24_eeg_hybrid_architecture.py",
    "eeg_advanced_architectures.py": "25_eeg_advanced_architectures.py",
    
    # Combined sessions and specialized approaches (26-30)
    "eeg_optimized_final.py": "26_eeg_optimized_stacking.py",
    "eeg_combined_sessions.py": "27_eeg_combined_sessions.py",
    "eeg_3d_classification.py": "28_eeg_3d_classification.py",
    "eeg_enhanced_deep_pca.py": "29_eeg_enhanced_deep_pca.py",
    "eeg_ultimate_model.py": "30_eeg_ultimate_model.py",
}

# Create a backup directory
backup_dir = "/home/qureshi/Desktop/EEG/original_files_backup"
if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)
    print(f"Created backup directory: {backup_dir}")

# Rename files and create backups
base_dir = "/home/qureshi/Desktop/EEG"
renamed_files = []
skipped_files = []

for old_name, new_name in file_mapping.items():
    old_path = os.path.join(base_dir, old_name)
    new_path = os.path.join(base_dir, new_name)
    backup_path = os.path.join(backup_dir, old_name)
    
    if os.path.exists(old_path):
        # Create a backup
        shutil.copy2(old_path, backup_path)
        
        # Rename the file
        shutil.copy2(old_path, new_path)
        renamed_files.append((old_name, new_name))
        print(f"Renamed: {old_name} -> {new_name}")
    else:
        skipped_files.append(old_name)
        print(f"Skipped: {old_name} (file not found)")

# Print summary
print("\nRenaming Summary:")
print(f"Total files renamed: {len(renamed_files)}")
print(f"Total files skipped: {len(skipped_files)}")

# Create a mapping file for reference
with open(os.path.join(base_dir, "file_name_mapping.txt"), "w") as f:
    f.write("EEG Project File Name Mapping\n")
    f.write("===========================\n\n")
    f.write("This file provides a reference for the renamed Python files in the EEG project.\n\n")
    
    f.write("Categories:\n")
    f.write("- 01-05: Data preparation and exploration\n")
    f.write("- 06-10: Basic classification\n")
    f.write("- 11-15: PCA analysis and optimization\n")
    f.write("- 16-20: Ensemble methods\n")
    f.write("- 21-25: Advanced models\n")
    f.write("- 26-30: Combined sessions and specialized approaches\n\n")
    
    f.write("File Mapping:\n")
    f.write("------------\n\n")
    
    for old_name, new_name in sorted(file_mapping.items(), key=lambda x: x[1]):
        f.write(f"{new_name} <- {old_name}\n")

print("\nCreated file_name_mapping.txt for reference")
print("\nNote: Original files have been backed up in the 'original_files_backup' directory")
print("The original files are still in place - this script created copies with the new names")
print("To complete the renaming process, you can delete the original files after verifying the new ones")
