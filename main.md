# Evolution Strategy for Logistic Regression

## Overview

This project implements a self-adaptive Evolution Strategy (ES) to optimize a binary Logistic Regression classifier for heart disease prediction.

Instead of using conventional gradient-based optimization methods, the parameters of the Logistic Regression model are evolved through a population-based evolutionary process.

The project combines concepts from Evolutionary Computation and Machine Learning, including:

- Logistic Regression
- Evolution Strategies
- Self-Adaptive Mutation
- `(μ + λ)` Selection
- Binary Classification
- Cross-Entropy Optimization
- L2 Regularization
- Population-Based Search
- Model Evaluation

The complete optimization process is implemented in Python and NumPy.

---

# Project Objectives

The main objectives of this project are:

- Implement binary Logistic Regression manually
- Optimize model parameters without gradient descent
- Implement a self-adaptive Evolution Strategy
- Evolve both model parameters and mutation step sizes
- Apply `(μ + λ)` elitist selection
- Analyze evolutionary convergence
- Evaluate classification performance
- Investigate the effect of L2 regularization
- Compare different ES population configurations

---

# Dataset

The project uses a Heart Disease dataset for binary classification.

The dataset contains:

```text
609 samples
14 original columns
```

The target variable is originally stored in:

```text
num
```

with values:

```text
0, 1, 2, 3, 4
```

For binary classification, the target is transformed as follows:

```text
num = 0  -> No heart disease
num > 0  -> Heart disease present
```

The resulting binary target is created using:

```python
target = 1 if num > 0 else 0
```

---

# Class Distribution

After converting the original target into a binary label, the dataset contains:

| Class | Meaning | Samples |
|---|---|---:|
| 0 | No heart disease | 335 |
| 1 | Heart disease present | 274 |

The dataset is relatively balanced between the two classes.

---

# Input Features

The dataset contains clinical variables related to cardiovascular health.

The features used for classification include variables such as:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG
- Maximum heart rate
- Exercise-induced angina
- ST depression
- Slope
- Thalassemia-related information

The `dataset` column is treated as metadata and removed before model training.

The original `num` column is also removed after constructing the binary target.

---

# Data Preprocessing

The preprocessing pipeline consists of:

1. Loading the dataset
2. Converting the original target into binary labels
3. Removing unnecessary columns
4. Performing a stratified train-test split
5. Calculating normalization statistics from the training data
6. Applying z-score normalization
7. Preparing NumPy arrays for Evolution Strategy optimization

---

# Train-Test Split

The dataset is divided into:

```text
Training: 70%
Testing:  30%
```

A stratified split is used to preserve the class distribution in both sets.

The random state is fixed to:

```text
42
```

for reproducibility.

---

# Feature Standardization

The input features are standardized using z-score normalization:

```text
x_standardized = (x - μ) / σ
```

where:

```text
μ = training-set mean
σ = training-set standard deviation
```

The mean and standard deviation are calculated using only the training data.

The same statistics are then applied to the test set.

This avoids information leakage from the test data into the training process.

---

# Logistic Regression Model

The binary classifier is based on Logistic Regression.

For an input vector `x`, the model first calculates:

```text
z = W^T x + b
```

The output probability is then obtained using the sigmoid function:

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

Therefore:

```text
y_hat = sigmoid(W^T x + b)
```

where:

- `W` represents the model weights
- `b` represents the bias
- `y_hat` represents the predicted probability of heart disease

A probability threshold of `0.5` is used for binary classification.

---

# Model Parameters

The complete Logistic Regression parameter vector is represented as:

```text
θ = [W, b]
```

Instead of updating these parameters using gradients, the Evolution Strategy treats the complete parameter vector as an individual in an evolutionary population.

The optimization problem therefore becomes:

```text
Find θ that minimizes classification loss
```

---

# Cross-Entropy Loss

Binary Cross-Entropy is used as the main objective function.

The loss is:

```text
L = -[y log(y_hat) + (1-y) log(1-y_hat)]
```

The average loss across the training dataset is used to evaluate each candidate solution.

Predicted probabilities are clipped before calculating logarithms to improve numerical stability.

---

# L2 Regularization

L2 regularization is added to the objective function:

```text
L_total =
L_cross_entropy + λ_reg ||W||²
```

The regularization term is applied only to the weight parameters and not to the bias.

The baseline experiment uses:

```text
λ_reg = 0.01
```

An additional experiment evaluates stronger regularization using:

```text
λ_reg = 0.1
```

The goal is to investigate how regularization strength affects model performance and generalization.

---

# Evolution Strategy

Evolution Strategies are population-based optimization algorithms designed mainly for continuous parameter optimization.

Unlike gradient-based algorithms, Evolution Strategies do not require derivatives of the objective function.

Instead, candidate parameter vectors are repeatedly:

```text
Generated
Evaluated
Mutated
Selected
```

until the population converges toward stronger solutions.

---

# Individual Representation

Each individual in the population contains two main components:

```text
θ = Model parameters
σ = Mutation step sizes
```

Therefore, each individual contains:

1. A candidate Logistic Regression model
2. Parameters controlling how strongly that model mutates

This allows the optimization process to be self-adaptive.

---

# Parameter Initialization

The Logistic Regression parameters are initialized uniformly:

```text
θ ~ Uniform(-0.1, 0.1)
```

The mutation step sizes are initialized using:

```text
σ ~ Uniform(0.01, 0.1)
```

Each model parameter has its own mutation step size.

---

# Self-Adaptive Mutation

A main feature of the implementation is self-adaptation.

Instead of using a fixed mutation strength throughout training, the Evolution Strategy evolves the mutation step sizes together with the Logistic Regression parameters.

The step sizes are updated using log-normal mutation.

Conceptually:

```text
σ_i' =
σ_i * exp(
    τ' * N(0,1)
    +
    τ * N_i(0,1)
)
```

The model parameters are then mutated according to:

```text
θ_i' =
θ_i + σ_i' * N_i(0,1)
```

This allows the algorithm to automatically adjust the scale of exploration during the optimization process.

---

# Global and Local Learning Rates

The self-adaptive mutation mechanism uses both global and local learning rates.

These rates depend on the number of optimized parameters.

The global component changes all mutation strengths together, while the local component allows individual parameters to adapt independently.

This provides a balance between:

- Global exploration
- Parameter-specific adaptation

---

# Mutation Constraints

Several constraints are applied to improve numerical stability.

Model parameters are clipped to:

```text
[-5, 5]
```

Mutation step sizes are prevented from becoming smaller than:

```text
1e-6
```

These constraints help prevent extreme parameter values and mutation collapse.

---

# Fitness Function

Evolution Strategies traditionally maximize fitness.

Because Logistic Regression training is formulated as a loss-minimization problem, fitness is defined as:

```text
fitness = -loss
```

Therefore:

```text
Lower loss
    ↓
Higher fitness
    ↓
Greater probability of survival
```

---

# (μ + λ) Evolution Strategy

The project uses an elitist `(μ + λ)` Evolution Strategy.

In this notation:

```text
μ = Number of parent individuals
λ = Number of offspring
```

At every generation:

1. The current population contains `μ` parents
2. `λ` offspring are generated
3. Parents and offspring are combined
4. All individuals are evaluated
5. The best `μ` individuals survive

Because parents compete directly with offspring, strong solutions can survive across multiple generations.

---

# Evolutionary Training Process

The overall training procedure is:

```text
1. Initialize μ individuals

2. Evaluate their fitness

3. Generate λ offspring

4. Apply self-adaptive mutation

5. Evaluate offspring fitness

6. Combine parents and offspring

7. Sort according to fitness

8. Keep the best μ individuals

9. Record training statistics

10. Repeat for the configured number of generations
```

---

# Default ES Configuration

The main experiment uses the following configuration:

| Hyperparameter | Value |
|---|---:|
| μ | 30 |
| λ | 210 |
| Generations | 100 |
| L2 Regularization | 0.01 |

The notebook also investigates alternative configurations such as:

```text
μ = 50
λ = 250
```

and:

```text
μ = 30
λ = 400
```

A stronger regularization experiment additionally uses:

```text
μ = 30
λ = 400
λ_reg = 0.1
```

---

# Convergence Analysis

During training, several statistics are recorded for every generation:

- Best training loss
- Mean population loss
- Training accuracy
- Test accuracy

These measurements provide information about both evolutionary convergence and model performance.

---

# Baseline Training Results

The main experiment uses:

```text
μ = 30
λ = 210
Generations = 100
L2 = 0.01
```

The training loss decreases rapidly during the early generations.

The best loss reaches approximately:

```text
0.3438
```

by generation 10.

At the end of training, the best loss reaches approximately:

```text
0.3242
```

After the early improvement, the population becomes relatively stable.

---

# Training and Test Accuracy

During the baseline experiment, training accuracy increased to approximately:

```text
0.8779
```

while test accuracy stabilized around:

```text
0.90
```

The recorded experiment did not show a large gap between training and test performance.

---

# Baseline Test Performance

The final baseline model achieved:

| Metric | Score |
|---|---:|
| Accuracy | 0.9016 |
| Precision | 0.8902 |
| Recall | 0.8902 |
| F1-Score | 0.8902 |

These results show that the implemented Evolution Strategy was able to optimize the Logistic Regression classifier without using gradient-based training.

---

# Baseline Confusion Matrix

The baseline confusion matrix contains:

```text
True Negatives  = 92
False Positives = 9
False Negatives = 9
True Positives  = 73
```

or:

```text
[[92, 9],
 [ 9, 73]]
```

The model therefore produced the same number of false-positive and false-negative predictions in the baseline experiment.

---

# Baseline Training Curves

The baseline training curves show:

- Best population loss
- Mean population loss
- Training accuracy
- Test accuracy

The curves show rapid improvement during the early generations followed by more stable convergence.

---

# Stronger L2 Regularization

A second experiment increases the regularization coefficient to:

```text
λ_reg = 0.1
```

while also using a larger offspring population.

The configuration is:

```text
μ = 30
λ = 400
λ_reg = 0.1
```

The purpose of this experiment is to investigate whether stronger regularization changes model generalization.

---

# Stronger L2 Results

The saved confusion matrix for the stronger L2 experiment contains:

```text
True Negatives  = 92
False Positives = 9
False Negatives = 8
True Positives  = 74
```

or:

```text
[[92, 9],
 [ 8, 74]]
```

Compared with the baseline experiment, this configuration correctly identifies one additional positive case.

---

# Sensitivity Analysis

The project also investigates the sensitivity of the Evolution Strategy to several important hyperparameters.

## Population Size

Population size directly affects the balance between exploration and computational cost.

A larger offspring population can increase diversity and explore a larger area of the search space.

However, it also increases the number of fitness evaluations required during each generation.

Smaller populations reduce computational cost but may increase the risk of premature convergence.

---

## Self-Adaptation Rates

The global and local mutation learning rates control how quickly mutation step sizes change.

Aggressive adaptation may improve early exploration but can also introduce instability.

Very slow adaptation may prevent the algorithm from adjusting efficiently during different stages of optimization.

The self-adaptive formulation used in this project attempts to balance these behaviors.

---

## Regularization Strength

The L2 coefficient controls the penalty applied to large Logistic Regression weights.

Smaller values allow the model to fit the training data more freely.

Larger values encourage smaller parameter magnitudes and simpler models.

However, excessive regularization may lead to underfitting.

The experiments compare:

```text
λ_reg = 0.01
```

and:

```text
λ_reg = 0.1
```

to investigate this trade-off.

---

# Medical Classification Considerations

For medical classification problems, overall accuracy alone is not sufficient.

False negatives are particularly important because they represent patients with heart disease being classified as healthy.

The project therefore evaluates:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The baseline model produced:

```text
9 false negatives
```

while the stronger L2 experiment produced:

```text
8 false negatives
```

This shows why class-specific errors should also be considered when evaluating classification models.

---

# Why Use Evolution Strategies?

Gradient-based optimization is the standard approach for Logistic Regression.

However, Evolution Strategies provide an alternative way to study optimization without derivative information.

Some useful properties include:

- No gradient calculation
- Applicability to non-differentiable objectives
- Population-based exploration
- Self-adaptive mutation
- Flexible objective functions

The project demonstrates how evolutionary optimization can be used to train a conventional machine learning classifier.

---

# Limitations

Several limitations should be considered when interpreting the results.

## Computational Cost

Evolution Strategies require evaluating many candidate models.

For example, with:

```text
μ = 30
λ = 210
Generations = 100
```

a large number of fitness evaluations are required compared with standard gradient-based Logistic Regression.

## Single Train-Test Split

The reported performance is based on a fixed stratified train-test split.

Using cross-validation or repeated experiments with different random seeds would provide a more reliable estimate of generalization performance.

## Dataset Size

The dataset contains:

```text
609 samples
```

Therefore, the results should not be interpreted as clinical evidence.

The project is primarily an Evolutionary Computation and Machine Learning experiment rather than a medical diagnostic system.

## Limited Hyperparameter Search

The project investigates several ES configurations but does not perform exhaustive hyperparameter optimization.

A broader search over population sizes, mutation parameters, number of generations, and regularization coefficients could provide additional information.

---

# Possible Improvements

Possible future extensions include:

- K-fold cross-validation
- Repeated experiments with multiple random seeds
- ROC curve analysis
- ROC-AUC evaluation
- Precision-Recall curves
- Cost-sensitive fitness functions
- Class-weighted objectives
- Alternative Evolution Strategy configurations
- `(μ, λ)` selection
- Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
- Hybrid evolutionary-gradient optimization
- Comparison with standard Gradient Descent
- Comparison with scikit-learn Logistic Regression
- Automated hyperparameter tuning

---

# Project Structure

```text
evolution-strategy-logistic-regression/
|
|-- main.ipynb
|-- Heart Disease dataset.csv
|
|-- accuracy_plot.png
|-- confusion_matrix.png
|-- accuracy_plot_l2_0_1.png
|-- confusion_matrix_l2_0_1.png
|
|-- requirements.txt
|-- README.md
`-- main.md
```

---

# File Description

| File | Description |
|---|---|
| `main.ipynb` | Complete preprocessing, Logistic Regression, Evolution Strategy training, evaluation, and visualization |
| `Heart Disease dataset.csv` | Heart Disease dataset used in the experiments |
| `accuracy_plot.png` | Baseline evolutionary loss and accuracy curves |
| `confusion_matrix.png` | Baseline test-set confusion matrix |
| `accuracy_plot_l2_0_1.png` | Training curves using stronger L2 regularization |
| `confusion_matrix_l2_0_1.png` | Confusion matrix using stronger L2 regularization |
| `requirements.txt` | Python dependencies |
| `README.md` | Short project overview |
| `main.md` | Complete project documentation |

---

# Requirements

The project requires:

```text
numpy
pandas
matplotlib
scikit-learn
jupyter
```

Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Clone or download the repository and install the required dependencies.

Then start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
main.ipynb
```

and run the notebook cells sequentially.

The dataset should remain in the same directory as the notebook:

```text
Heart Disease dataset.csv
```

---

# Technologies and Methods

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Logistic Regression
- Evolution Strategies
- Evolutionary Computation
- Self-Adaptive Mutation
- `(μ + λ)` Selection
- Cross-Entropy Loss
- L2 Regularization
- Binary Classification
- Feature Standardization
- Population-Based Optimization
- Confusion Matrix Analysis

---

# Key Concepts Demonstrated

This project demonstrates:

- Evolutionary optimization
- Continuous parameter optimization
- Self-adaptation
- Strategy parameters
- Population initialization
- Mutation
- Elitist selection
- Logistic Regression
- Sigmoid classification
- Cross-Entropy optimization
- Regularization
- Binary classification
- Feature normalization
- Train-test splitting
- Classification metrics
- Convergence analysis
- Hyperparameter sensitivity
- Evolutionary Machine Learning

---

# Main Results

The baseline Evolution Strategy achieved:

```text
Accuracy  = 0.9016
Precision = 0.8902
Recall    = 0.8902
F1-Score  = 0.8902
```

with the confusion matrix:

```text
[[92, 9],
 [ 9, 73]]
```

The stronger L2 experiment produced:

```text
[[92, 9],
 [ 8, 74]]
```

showing a small improvement in positive-class detection in the saved experiment.

Overall, the experiments demonstrate that a self-adaptive Evolution Strategy can optimize the parameters of a Logistic Regression classifier without gradient-based training.

---

# Conclusion

This project demonstrates the use of self-adaptive Evolution Strategies for continuous machine learning optimization.

Instead of training Logistic Regression using gradients, the model parameters and their mutation step sizes are evolved simultaneously through a `(μ + λ)` population-based process.

The experiments show stable convergence and strong binary classification performance on the Heart Disease dataset. The baseline experiment achieved approximately 90.16% test accuracy with balanced precision and recall.

The comparison with stronger L2 regularization also shows how evolutionary optimization can be combined with conventional regularization techniques.

Overall, the project provides practical experience with Evolution Strategies, self-adaptive mutation, population-based optimization, Logistic Regression, regularization, and classification performance analysis.

---

# Course Information

**Course:** Evolutionary Computation  
**University:** Shiraz University  
**Year:** 2026

---

# Author

Saghar Kheradmand
