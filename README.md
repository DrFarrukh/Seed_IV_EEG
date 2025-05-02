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

### 6. Optimized Stacking Ensemble (72.22% Accuracy)
- Implemented a stacking ensemble approach with four base models and a meta-learner
- Used only the top 10 most important EEG channels (61, 58, 2, 31, 23, 4, 10, 15, 30, 40)
- Applied PCA to reduce dimensionality to 20 components (87.32% variance explained)
- Employed hyperparameter-tuned base models:
  - Support Vector Machine (SVM) with optimized kernel, C, and gamma
  - Random Forest with tuned estimators, depth, and min samples
  - Gradient Boosting with optimized estimators, depth, and learning rate
  - Multi-Layer Perceptron (MLP) with tuned layers, regularization, and learning rate
- Used Random Forest (n_estimators=200) as the meta-learner with 5-fold cross-validation
- Achieved balanced performance across all four classes

### 7. Multi-Session Analysis
- Analyzed data from three separate EEG recording sessions (1080 total trials)
- Performed individual session analysis and combined sessions analysis
- Observed significant performance differences across sessions:
  - Session 1: 72.22% accuracy (Stacking Ensemble, CV: 66.94% ± 1.84%)
  - Session 2: 56.48% accuracy (Stacking Ensemble, CV: 62.22% ± 5.51%)
  - Session 3: 56.48% accuracy (Voting Ensemble, CV: 57.22% ± 4.51%)
  - Combined Sessions: 44.75% accuracy (Voting Ensemble, CV: 48.98% ± 4.22%)
- Cross-session generalization was poor (18.89% - 37.22% accuracy)
- Findings confirm strong session-specific patterns that don't generalize well

### 8. Optimal PCA Analysis for Combined Sessions
- Analyzed the optimal number of PCA components for the combined dataset (all three sessions)
- Evaluated explained variance and classification performance for different component counts
- Key findings:
  - 70% variance: 5 components
  - 80% variance: 12 components
  - 85% variance: 22 components (recommended optimal value)
  - 90% variance: 42 components
  - 95% variance: 76 components
- Best direct accuracy: 50.31% with 20 components
- Best cross-validated accuracy: 41.57% with 22 components
- Recommended 22 components as the optimal balance between accuracy and variance explained

### 9. Session Variability Analysis
- Created visualizations to understand differences between EEG recording sessions
- t-SNE visualization revealed clear clustering by session rather than by class

### 10. Transfer Learning and Domain Adaptation
- Implemented transfer learning using a pre-trained EEGNet model
- Developed domain adaptation techniques to reduce domain shift between sessions
- Used gradient reversal layers to learn session-invariant features
- Achieved 50.00% accuracy on Session 3 data using knowledge from Session 1
- Domain adaptation improved performance by 2.8% over the baseline model

### 11. Advanced Deep Learning with ResNet
- Implemented ResNet architectures with residual connections for EEG classification
- Compared standard ResNet (18-layer) with deeper ResNet (34-layer) architectures
- Deeper ResNet achieved 42.90% accuracy on the combined dataset
- Residual connections allowed training deeper networks without vanishing gradient issues
- Deeper architecture showed 2.5% improvement over standard ResNet
- Feature distributions showed significant shifts between sessions
- PCA components demonstrated session-specific patterns in the data
- Cross-session generalization performance was poor (18.89% - 37.22%)
- Findings suggest the need for session-specific calibration or transfer learning techniques

### 9. Advanced Deep Learning Approaches
- Implemented CNN, Transformer, and Dense neural networks with PCA features
- Applied data augmentation techniques (Gaussian noise, time warping)
- Explored ensemble methods with cross-validation
- Increased regularization with L1=5e-5, L2=5e-4 and higher dropout rates
- Implemented ensemble methods combining models from different folds
- Applied data augmentation techniques to improve model robustness

### 9. Future Plans
- Explore hybrid architectures combining CNN and Transformer elements
- Implement attention mechanisms specific to EEG data
- Apply transfer learning from pre-trained EEG models

## Files Description

The files are organized with a numbered prefix system to make it easier to track their purpose and progression:

### Data Preparation and Exploration (01-05)
- `01_eeg_data_exploration.py`: Principal Component Analysis of EEG data
- `02_eeg_3d_visualization.py`: 3D visualization of EEG data
- `03_eeg_summary_visualization.py`: Summary visualizations of results
- `04_eeg_feature_engineering.py`: Domain-specific feature engineering for EEG data
- `05_eeg_initial_classification.py`: Quick classification baseline

### Basic Classification (06-10)
- `06_eeg_basic_classification.py`: Basic classification of EEG data
- `07_eeg_trial_based_classification.py`: Classification based on trial features
- `08_eeg_advanced_classification.py`: Advanced classification techniques
- `09_eeg_balanced_model.py`: Balanced model with optimal features
- `10_eeg_baseline_model.py`: Final model implementation

### PCA Analysis and Optimization (11-15)
- `11_eeg_pca_implementation.py`: Efficient PCA implementation
- `12_eeg_trial_pca_classification.py`: Trial-based PCA classification
- `13_eeg_trial_pca_improved.py`: PCA applied to trial-based features (improved)
- `14_eeg_optimal_pca_components.py`: Analysis to find optimal number of PCA components
- `15_eeg_deep_pca_model.py`: Deep learning with PCA features

### Ensemble Methods (16-20)
- `16_eeg_basic_ensemble.py`: Ensemble methods with optimal PCA components
- `17_eeg_specialized_ensemble.py`: Specialized ensemble methods
- `18_eeg_super_learner.py`: Super learner ensemble approach
- `19_eeg_advanced_ensemble.py`: Advanced ensemble methods
- `20_eeg_advanced_ensemble_improved.py`: Improved version of advanced ensemble methods

### Advanced Models (21-25)
- `21_eeg_deep_stacking.py`: Deep stacking ensemble methods
- `22_eeg_deep_stacking_improved.py`: Improved version of deep stacking ensemble
- `23_eeg_recurrent_model.py`: Recurrent neural networks with stacking
- `24_eeg_hybrid_architecture.py`: Hybrid model combining multiple approaches
- `25_eeg_advanced_architectures.py`: Advanced neural network architectures

### Combined Sessions and Specialized Approaches (26-33)
- `26_eeg_optimized_stacking.py`: Implementation of the Optimized Stacking Ensemble (72.22% accuracy)
- `27_eeg_combined_sessions.py`: Analysis of combined data from two EEG sessions (55.56% accuracy)
- `28_eeg_3d_classification.py`: 3D classification using CNN, LSTM, and EEGNet architectures
- `29_eeg_enhanced_deep_pca.py`: Enhanced deep learning with PCA
- `30_eeg_ultimate_model.py`: Ultimate model combining best approaches
- `31_eeg_multi_session_analysis.py`: Comprehensive analysis of three EEG sessions (individual and combined)
- `32_eeg_session_visualization.py`: Visualizations of session differences and cross-session generalization
- `33_eeg_optimal_pca_combined.py`: Analysis of optimal PCA components for combined sessions data

A complete mapping between old and new filenames can be found in `file_name_mapping.txt`.

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
| Optimized Stacking | **72.22%** | 20 optimized | Stacking ensemble with SVM, RF, GB, MLP and RF meta-learner |
| Session 1 Analysis | **72.22%** | 20 optimized | Stacking ensemble on Session 1 data |
| Session 2 Analysis | 56.48% | 20 optimized | Stacking ensemble on Session 2 data |
| Session 3 Analysis | 56.48% | 20 optimized | Voting ensemble on Session 3 data |
| Combined Sessions | 44.75% | 20 optimized | Voting ensemble on all three sessions |
| Deep PCA Model | 61.11% | 20 optimized | Neural network on PCA features |
| Deep PCA + Weighted Ensemble | 72.22% | 20 optimized | Equal-weighted model combination |
| Deep PCA + Cross-validation | 62.22% ± 5.28% | 30 optimized | 5-fold cross-validation |
| **Deep PCA Dense Ensemble** | **66.67%** | **30 optimized** | **Dense architecture with ensemble** |
| Deep PCA CNN Ensemble | 57.41% | 30 optimized | CNN architecture with ensemble |
| Deep PCA Transformer Ensemble | 58.33% | 30 optimized | Transformer architecture with ensemble |
| PCA Optimized (25 components) | 49.07% | 25 optimized | Optimal PCA components for combined dataset |
| EEGNet (Base) | 40.10% | Raw EEG | Base EEGNet architecture |
| EEGNet (Transfer Learning) | 39.20% | Raw EEG | Fine-tuned EEGNet with transfer learning |
| Baseline (Session 3) | 47.22% | 25 optimized | Direct training on target session |
| **Domain Adaptation** | **50.00%** | 25 optimized | **Domain adaptation with gradient reversal** |
| ResNet (Standard) | 40.43% | 25 components, CNN | ResNet with residual connections |
| ResNet (Deeper) | 42.90% | 25 components, CNN | Deeper ResNet with more residual blocks |

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

11. **Session-specific patterns limit generalization**: Strong session-specific patterns in EEG data limit cross-session generalization, requiring specialized approaches like domain adaptation.

12. **Domain adaptation outperforms transfer learning**: Domain adaptation with gradient reversal (50.00%) outperformed direct transfer learning (39.20%) by explicitly addressing domain shift between sessions.

13. **Optimal PCA components vary by task**: While 20 components were optimal for single-session analysis, 25 components provided better performance (49.07%) for the combined dataset.
