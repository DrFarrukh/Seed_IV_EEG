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
- Tested different numbers of components (10-200)
- Best accuracy: 68.52% with 100 PCA components

### 5. Current Work
- Implementing domain-specific EEG feature engineering
- Developing ensemble methods for improved classification
- Creating specialized features based on EEG literature

### 6. Future Plans
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
- `eeg_domain_features.py`: Domain-specific EEG feature engineering (current work)
- `eeg_ensemble_classification.py`: Ensemble methods for classification (current work)

## Results Summary

| Approach | Accuracy | Features |
|----------|----------|----------|
| Flattened Data | ~33-35% | Original features |
| Trial-based | ~65-70% | Channel-preserved features |
| PCA (94 components) | 67.59% | 95% variance preserved |
| PCA (100 components) | 68.52% | Optimal component count |
| Domain-specific + Ensemble | In progress | EEG-specific features |
