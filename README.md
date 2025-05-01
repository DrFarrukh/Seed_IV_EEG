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

### 7. Advanced Deep Learning Approaches
- Implemented CNN, Transformer, and Dense neural networks with PCA features
- Applied k-fold cross-validation (5-fold) for robust performance evaluation
- Increased regularization with L1=5e-5, L2=5e-4 and higher dropout rates
- Implemented ensemble methods combining models from different folds
- Applied data augmentation techniques to improve model robustness

### 8. Future Plans
- Explore hybrid architectures combining CNN and Transformer elements
- Implement attention mechanisms specific to EEG data
- Apply transfer learning from pre-trained EEG models

## Files Description

- `eeg_classification.py`: Initial data exploration script
- `eeg_pca_analysis.py`: PCA analysis on flattened data
- `eeg_pca_efficient.py`: Optimized PCA implementation
- `eeg_3d_analysis.py`: Analysis of 3D structure and trial-based classification
- `eeg_trial_classification.py`: Classification based on trial features
- `eeg_trial_pca_fixed.py`: PCA applied to trial-based features
- `eeg_optimal_pca.py`: Analysis to find optimal number of PCA components
- `eeg_ensemble_optimal.py`: Ensemble methods with optimal PCA components
- `eeg_domain_features.py`: Domain-specific feature engineering for EEG data
- `eeg_3d_classification.py`: 3D classification using CNN, LSTM, and EEGNet architectures
- `eeg_hybrid_model.py`: Combined approach using channel-specific, epoch-specific, and global PCA
- `eeg_specialized_ensemble.py`: Optimized model with class-specific classifiers
- `eeg_enhanced_deep_pca.py`: Enhanced Deep PCA model with data augmentation and regularization
- `eeg_deep_pca_final.py`: Final Deep PCA model with CNN, Transformer, and Dense architectures

## Results Summary

| Approach | Accuracy | Features | Notes |
|----------|----------|----------|-------|
| Flattened Data | ~33-35% | Original features | Baseline approach |
| Trial-based | ~65-70% | Channel-preserved features | Preserves channel structure |
| PCA (20 components) | 68.06% | 20 (87.32% variance) | Good performance/complexity ratio |
| PCA (94 components) | 67.59% | 94 (95% variance) | More features, similar performance |
| PCA (100 components) | 68.52% | 100 (96% variance) | High accuracy with many features |
| Channel-CNN | 25.00% | 3D structure | Spatial patterns only |
| Epoch-LSTM | 42.59% | 3D structure | Temporal patterns only |
| EEGNet-inspired | 49.07% | 3D structure | Combined spatial-temporal |
| Hybrid Model | 63.89% | 522 combined | Channel, epoch, and global features |
| Specialized Ensemble | 66.67% | 50 optimized | Class-specific models, reduced features |
| Balanced Final Model | 66.67% | 30 optimized | Reduced features, balanced performance |
| Super Learner | 62.96% | Combined approaches | Meta-learning with deep learning |
| Advanced Ensemble | 30.56% | Augmented data | Deep learning with data augmentation |
| Optimized Stacking | 72.22% | 20 optimized | Traditional model stacking |
| Deep PCA Model | 61.11% | 20 optimized | Neural network on PCA features |
| Deep PCA + Weighted Ensemble | 72.22% | 20 optimized | Equal-weighted model combination |
| Deep PCA + Cross-validation | 62.22% ± 5.28% | 30 optimized | 5-fold cross-validation |
| **Deep PCA Dense Ensemble** | **66.67%** | **30 optimized** | **Dense architecture with ensemble** |
| Deep PCA CNN Ensemble | 57.41% | 30 optimized | CNN architecture with ensemble |
| Deep PCA Transformer Ensemble | 58.33% | 30 optimized | Transformer architecture with ensemble |

## Key Insights

1. **Dimensionality reduction is crucial**: PCA-based approaches consistently outperform models using raw features, with 20-30 components providing the optimal balance.

2. **Channel information is most important**: Channel-specific features contribute significantly to predictive power, particularly from channels 61, 58, 2, and 31.

3. **Class-specific performance varies**: Class 2 (F1-score: 0.75-0.82) and Class 1 (F1-score: 0.71-0.83) are consistently better classified than Classes 0 and 3.

4. **Feature selection matters**: Reducing from 490 to 20-30 features while improving performance demonstrates the importance of targeted feature engineering.

5. **Ensemble architecture is critical**: Stacking ensemble (72.22%) outperforms voting ensemble (66.67%) by using a meta-learner that learns optimal combinations of base models.

6. **Hyperparameter tuning yields significant gains**: Grid search optimization of base models improved overall performance by ~4% compared to default parameters.

7. **Neural network architecture selection is important**: Dense networks (66.67%) outperformed both CNN (57.41%) and Transformer (58.33%) architectures when using PCA features.

8. **Cross-validation provides robust estimates**: 5-fold cross-validation showed consistent performance across folds with standard deviations of 5-6%.

9. **Data augmentation improves robustness**: Applying noise, time warping, and spectral augmentation helped models generalize better to unseen data.

10. **Model-specific strengths**: SVM works best for Class 0, Random Forest for Class 1, Gradient Boosting for Class 2, and a combination approach for Class 3.
