# CAPSTONE PROJECT REPORT

## Hybrid Quantum-Classical Machine Learning for Titanic Survival Prediction

### 1. Abstract

This capstone develops a hybrid quantum-classical machine-learning workflow for binary classification using the Titanic survival dataset. A classical Logistic Regression model is established as a baseline and is compared with a variational quantum classifier implemented with Qiskit and Qiskit Aer. The workflow includes data cleaning, missing-value handling, numerical scaling, categorical encoding, train-test splitting, quantum angle encoding, parameterized quantum circuits, measurement-based probability estimation, and classical optimization using COBYLA. The project demonstrates how a classical optimizer can train a parameterized quantum circuit, illustrating the central hybrid-computing model of near-term quantum machine learning. Because the dataset is small and the quantum circuit is executed in simulation, the project does not claim quantum advantage. Instead, it provides a reproducible educational and research foundation for testing quantum machine-learning methods against classical baselines.

### 2. Objectives

1. Build a reproducible ML pipeline using a real classification dataset.
2. Establish a classical Logistic Regression baseline.
3. Encode selected features into a four-qubit variational circuit.
4. Train the quantum circuit with a classical optimizer.
5. Compare classical and quantum metrics.
6. Provide a foundation for noisy simulation and hardware experiments.

### 3. Dataset

The supplied Titanic CSV contains 418 records and 12 columns. The target variable is `Survived`. The model uses passenger class, sex, age, number of siblings/spouses, number of parents/children, fare, and embarkation port for the classical pipeline.

The quantum model uses four features: `Pclass`, `Age`, `Fare`, and `SibSp`. Missing numerical values are imputed with the median. The quantum features are scaled to the interval [0, pi] for angle encoding.

### 4. Methodology

#### Classical model
Logistic Regression is used as a transparent baseline. Numerical features are median-imputed and standardized. Categorical variables are imputed and one-hot encoded.

#### Quantum model
A four-qubit variational circuit is used:

```text
Input features
     |
Angle encoding: RY(x_i)
     |
Variational RY/RZ layer
     |
Ring CNOT entanglement
     |
Variational RY layer
     |
Measurement
     |
P(y=1)
```

The first qubit measurement probability is used as the predicted probability of survival.

The binary cross-entropy loss is minimized by the classical COBYLA optimizer.

### 5. Evaluation

The following metrics are reported:

- Accuracy
- Precision
- Recall
- F1-score

A confusion matrix is produced for the classical baseline and a comparison chart is produced when quantum execution completes.

### 6. Results

Run `hybrid_quantum_titanic.py` to populate the numerical results. Do not insert invented quantum results into the report. The generated `model_comparison.csv` should be used as the source for the final results table.

Suggested table:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | RUN | RUN | RUN | RUN |
| Quantum Variational Classifier | RUN | RUN | RUN | RUN |

### 7. Discussion

The experiment should be interpreted as a methodological comparison. A small simulated quantum circuit does not establish quantum advantage. The classical baseline may perform strongly because Titanic survival is a relatively small tabular classification problem. Quantum-model performance can also be affected by shot noise, optimizer choice, circuit depth, parameter initialization, and simulator/hardware noise.

### 8. Limitations

- Small dataset.
- Simulator-based quantum execution.
- Restricted four-feature quantum representation.
- Shot noise during quantum probability estimation.
- COBYLA and circuit design are not necessarily optimal.
- No claim of quantum advantage.
- Results can vary with random seed and software version.

### 9. Future Work

1. Implement a quantum kernel classifier.
2. Compare VQC with SVM and XGBoost.
3. Test deeper QNN architectures.
4. Add depolarizing and readout noise.
5. Run the trained circuit on IBM Quantum hardware.
6. Repeat experiments over multiple seeds.
7. Report confidence intervals and runtime.
8. Integrate QAOA for a separate resource-allocation or feature-selection optimization problem.
9. Integrate VQE as a separate quantum simulation experiment.
10. Explore a cybersecurity dataset for a domain-specific version.

### 10. Conclusion

The capstone demonstrates a complete hybrid quantum-classical machine-learning workflow. The key contribution is not a claim that the quantum model is superior, but a reproducible framework for studying how parameterized quantum circuits can be incorporated into a classical machine-learning pipeline. The project can be extended toward noisy quantum simulation, quantum kernels, QAOA-based optimization, and real quantum hardware experiments.
