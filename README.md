# Evolution Strategy for Logistic Regression

A machine learning project that uses a self-adaptive Evolution Strategy (ES) to optimize the parameters of a binary Logistic Regression classifier for heart disease prediction.

Instead of gradient descent, the model parameters and mutation step sizes are evolved using a `(μ + λ)` Evolution Strategy implemented with Python and NumPy.

## Main Features

- Logistic Regression implemented manually
- Self-adaptive mutation
- `(μ + λ)` selection
- Cross-entropy loss with L2 regularization
- Heart disease binary classification
- Training convergence and classification evaluation

## Results

The baseline experiment achieved approximately:

- Accuracy: **90.16%**
- Precision: **89.02%**
- Recall: **89.02%**
- F1-score: **89.02%**

## Run

Install the dependencies:

```bash
pip install -r requirements.txt
