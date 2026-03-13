# Training Analysis

## Dataset

Dataset used: Iris dataset (classification)

Rows: 150
Features: 4
Target: species

Missing values handled using: drop strategy.

---

## Model Experiments

### Decision Tree

Hyperparameters:

* max_depth: 3
* min_samples_leaf: 2

Results:

* Accuracy: 0.96
* Training loss: stable
* Validation loss: slightly higher than training

Diagnosis:
Model performs well with minimal overfitting.

---

### MLP Model

Hyperparameters:

* hidden_units: 16
* learning_rate: 0.01
* batch_size: 16
* epochs: 50

Results:

* Training loss decreased steadily
* Validation loss started increasing after epoch 35

Diagnosis:
Overfitting detected.

Suggested fix:

* Add dropout
* Reduce model size
* Increase dataset size

---

## Diagnostics Observed

Overfitting flag triggered:
Training loss ↓ while validation loss ↑

No unstable gradients or NaN values detected.

---

## Conclusion

Decision Tree performed best for this dataset due to its simplicity and resistance to overfitting.

MLP required more regularization to generalize well.
