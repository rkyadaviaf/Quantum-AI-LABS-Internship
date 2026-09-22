# Hybrid Quantum-Classical ML Capstone

## Project
**Titanic Survival Prediction using Classical Machine Learning and a Variational Quantum Classifier**

This capstone compares a classical Logistic Regression baseline with a small Qiskit variational quantum classifier. The goal is to demonstrate a complete hybrid workflow rather than claim quantum advantage.

## Dataset
`titanic.csv` is the uploaded Titanic dataset.

Target:
- `Survived`

Classical features:
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

Quantum feature subset:
- Pclass
- Age
- Fare
- SibSp

The quantum subset is restricted to four continuous features so they can be angle encoded on four qubits.

## Architecture

```text
Titanic Dataset
      |
      v
Preprocessing
      |
      +--------------------+
      |                    |
      v                    v
Classical ML          Quantum ML
Logistic Regression   Variational Circuit
      |                    |
      v                    v
Prediction            Quantum probability
      |                    |
      +---------+----------+
                |
                v
       Accuracy / Precision
       Recall / F1 / Runtime
                |
                v
          Comparison
```

## Quantum model

The variational classifier uses:
1. Four-qubit angle encoding.
2. Parameterized RY/RZ rotations.
3. Ring CNOT entanglement.
4. A second variational layer.
5. Measurement of the first qubit.
6. COBYLA as the classical optimizer.
7. Binary cross-entropy as the training objective.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python hybrid_quantum_titanic.py
```

On Windows:

```text
.venv\Scripts\activate
```

The script automatically reports the classical baseline. If compatible Qiskit/Aer packages are installed, it also trains and evaluates the quantum model.

## Important research point

The dataset is small and classical Logistic Regression is a strong baseline. The purpose of this capstone is to demonstrate hybrid quantum-classical ML methodology, not to assume quantum advantage. Any claim of advantage must be supported by controlled experiments, identical data splits, runtime measurements, and appropriate statistical analysis.

## Suggested extensions

- Add a classical SVM/XGBoost baseline.
- Compare QNN/VQC with a quantum kernel classifier.
- Add noise using Qiskit Aer.
- Test multiple random seeds.
- Report mean ± standard deviation.
- Compare training time and inference time.
- Add a QAOA module for a separate optimization task.
- Add VQE as a quantum chemistry/ground-state demonstration.
