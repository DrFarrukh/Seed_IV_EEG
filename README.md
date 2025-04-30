# EEG Classification Project

This project focuses on classifying EEG (electroencephalogram) data using machine learning techniques. The dataset contains EEG recordings with 4 distinct classes.

## Dataset Structure

- **Original format**: 3D structure (62 channels × 64 epochs × 49 features)
- **Current format**: 2D structure (312,480 rows × 54 columns)
- **Features**: 49 EEG features including statistical measures, frequency bands, entropy metrics, etc.
- **Metadata**: epoch, channel, label, trial_id
- **Classes**: 4 balanced classes (0, 1, 2, 3)
- **Trials**: 360 unique trials (90 per class)

## Project Progress

### 1. Data Exploration
- Analyzed dataset structure and feature distributions
- Identified key features and their statistical properties
- Visualized class distributions and feature relationships

### 2. Initial Classification Approaches
- Applied standard classification to flattened data
- Achieved ~33-35% accuracy (slightly better than random chance)

### 3. 3D Structure Analysis
- Reconstructed the original 3D structure (channel × epoch × features)
- Created trial-based features by preserving channel information
- Improved classification accuracy to ~65-70%

### 4. PCA Dimensionality Reduction
- Reduced feature space from 3,038 to 94 components (95% variance preserved)
- Tested different numbers of components (5-100)
- Found optimal performance with 20 PCA components (87.32% variance preserved)
- Achieved 68.06% cross-validation accuracy with 20 components

### 5. Optimal PCA Analysis
- Systematically tested different component counts (5, 10, 15, 20, 25, 30, 40, 50, 75, 100)
- Identified sweet spot at 20 components (152x dimensionality reduction)
- Balanced performance vs. complexity with 87.32% variance preserved

### 6. Current Work
- Fine-tuning classifiers with optimal 20 PCA components
- Implementing ensemble methods for improved classification
- Creating specialized EEG features with optimal dimensionality

### 7. Future Plans
- Apply deep learning approaches (EEGNet, RNNs)
- Explore time-frequency representations
- Implement advanced hyperparameter tuning

## Files Description

- `eeg_classification.py`: Initial data exploration script
- `eeg_pca_analysis.py`: PCA analysis on flattened data
- `eeg_pca_efficient.py`: Optimized PCA implementation
- `eeg_3d_analysis.py`: Analysis of 3D structure and trial-based classification
- `eeg_trial_classification.py`: Classification based on trial features
- `eeg_trial_pca_fixed.py`: PCA applied to trial-based features
- `eeg_optimal_pca.py`: Analysis to find optimal number of PCA components
- `eeg_ensemble_optimal.py`: Ensemble methods with optimal PCA components (current work)

## Results Summary

| Approach | Accuracy | Features |
|----------|----------|----------|
| Flattened Data | ~33-35% | Original features |
| Trial-based | ~65-70% | Channel-preserved features |
| PCA (94 components) | 67.59% | 95% variance preserved |
| PCA (20 components) | 68.06% | 87.32% variance preserved |
| PCA (100 components) | 68.52% | More components, similar performance |
| Ensemble with 20 PCA components | In progress | Optimal dimensionality |
