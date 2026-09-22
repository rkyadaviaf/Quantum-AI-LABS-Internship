"""
Hybrid Quantum-Classical ML Capstone
Titanic Survival Prediction using Classical ML + Variational Quantum Classifier

Dataset: Titanic CSV supplied by the project author.
Target: Survived
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
DATA_FILE = "titanic.csv"

df = pd.read_csv(DATA_FILE)

# Keep a compact, reproducible feature set.
# Name, Ticket, Cabin and PassengerId are excluded because they
# are identifiers/high-cardinality fields for this capstone.
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target = "Survived"

X = df[features].copy()
y = df[target].astype(int)

print("=" * 65)
print("HYBRID QUANTUM-CLASSICAL ML CAPSTONE")
print("Titanic Survival Prediction")
print("=" * 65)
print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{y.value_counts().sort_index()}\n")

# ---------------------------------------------------------
# 2. Train/test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Embarked"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_features)
    ]
)

# ---------------------------------------------------------
# 3. Classical baseline
# ---------------------------------------------------------
classical_model = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

classical_model.fit(X_train, y_train)
classical_pred = classical_model.predict(X_test)

def metrics_dict(y_true, pred):
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }

classical_metrics = metrics_dict(y_test, classical_pred)

print("\n--- CLASSICAL BASELINE: LOGISTIC REGRESSION ---")
for k, v in classical_metrics.items():
    print(f"{k:10s}: {v:.4f}")

# ---------------------------------------------------------
# 4. Prepare a 4-feature quantum dataset
# ---------------------------------------------------------
# Four continuous features are selected so they can be encoded
# directly into a 4-qubit parameterized circuit.
q_features = ["Pclass", "Age", "Fare", "SibSp"]

q_train = X_train[q_features].copy()
q_test = X_test[q_features].copy()

q_imputer = SimpleImputer(strategy="median")
q_scaler = MinMaxScaler(feature_range=(0, np.pi))

q_train = q_imputer.fit_transform(q_train)
q_test = q_imputer.transform(q_test)

q_train = q_scaler.fit_transform(q_train)
q_test = q_scaler.transform(q_test)

# ---------------------------------------------------------
# 5. Quantum model
# ---------------------------------------------------------
# This section uses Qiskit Aer + a parameterized circuit.
# The cost is binary cross entropy on probabilities produced
# by measuring the first qubit.
#
# Circuit:
#   |0> -- Ry(x0) -- RY(theta) -- RZ(theta) -- entanglement
#   |0> -- Ry(x1) -- RY(theta) -- RZ(theta) -- ...
#
# The optimizer is classical; the circuit supplies the quantum
# probability distribution. This is a variational hybrid model.

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from scipy.optimize import minimize

    N_QUBITS = 4
    SHOTS = 512
    MAXITER = 30

    simulator = AerSimulator()

    def make_circuit(x, theta, measure=True):
        qc = QuantumCircuit(N_QUBITS, 1 if measure else 0)

        # Angle encoding
        for q in range(N_QUBITS):
            qc.ry(float(x[q]), q)

        # Variational layer
        for q in range(N_QUBITS):
            qc.ry(float(theta[q]), q)
            qc.rz(float(theta[N_QUBITS + q]), q)

        # Ring entanglement
        for q in range(N_QUBITS - 1):
            qc.cx(q, q + 1)
        qc.cx(N_QUBITS - 1, 0)

        # Second variational layer
        offset = 2 * N_QUBITS
        for q in range(N_QUBITS):
            qc.ry(float(theta[offset + q]), q)

        if measure:
            qc.measure(0, 0)

        return qc

    def quantum_probability(x, theta):
        qc = make_circuit(x, theta, measure=True)
        result = simulator.run(qc, shots=SHOTS).result()
        counts = result.get_counts()
        # Classical bit 0 is the measured first qubit.
        return counts.get("1", 0) / SHOTS

    # To keep runtime practical, train on a reproducible subset.
    rng = np.random.default_rng(42)
    n_train_q = min(160, len(q_train))
    idx = rng.choice(len(q_train), size=n_train_q, replace=False)

    Xq_train = q_train[idx]
    yq_train = y_train.iloc[idx].to_numpy()

    def q_loss(theta):
        eps = 1e-7
        probs = np.array([quantum_probability(x, theta) for x in Xq_train])
        probs = np.clip(probs, eps, 1 - eps)
        return -np.mean(
            yq_train * np.log(probs) +
            (1 - yq_train) * np.log(1 - probs)
        )

    initial_theta = np.zeros(3 * N_QUBITS)

    print("\n--- TRAINING QUANTUM VARIATIONAL CLASSIFIER ---")
    print(f"Quantum training samples: {n_train_q}")
    print(f"Shots per circuit: {SHOTS}")
    print(f"Maximum optimizer iterations: {MAXITER}")

    q_result = minimize(
        q_loss,
        initial_theta,
        method="COBYLA",
        options={"maxiter": MAXITER, "rhobeg": 0.3}
    )

    theta_opt = q_result.x

    # Evaluate quantum model on the complete test set.
    q_probs = np.array([quantum_probability(x, theta_opt) for x in q_test])
    q_pred = (q_probs >= 0.5).astype(int)
    quantum_metrics = metrics_dict(y_test, q_pred)

    print("\n--- QUANTUM VARIATIONAL CLASSIFIER ---")
    for k, v in quantum_metrics.items():
        print(f"{k:10s}: {v:.4f}")

    quantum_available = True

except Exception as exc:
    quantum_available = False
    quantum_metrics = None
    print("\nQuantum section could not be executed.")
    print("Install compatible packages and rerun:")
    print("  pip install qiskit qiskit-aer scipy scikit-learn pandas matplotlib")
    print(f"Reason: {exc}")

# ---------------------------------------------------------
# 6. Comparison
# ---------------------------------------------------------
if quantum_available:
    comparison = pd.DataFrame(
        [classical_metrics, quantum_metrics],
        index=["Classical Logistic Regression", "Quantum Variational Classifier"]
    )
    print("\n--- MODEL COMPARISON ---")
    print(comparison.round(4))

    comparison.to_csv("model_comparison.csv")

    ax = comparison[["accuracy", "precision", "recall", "f1"]].plot(
        kind="bar", figsize=(10, 5)
    )
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Classical vs Quantum Hybrid ML")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=200)
    plt.close()

# ---------------------------------------------------------
# 7. Confusion matrix for classical baseline
# ---------------------------------------------------------
cm = confusion_matrix(y_test, classical_pred)

plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Classical Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout()
plt.savefig("classical_confusion_matrix.png", dpi=200)
plt.close()

print("\nProject outputs generated:")
print("  model_comparison.csv (if quantum training completed)")
print("  model_comparison.png  (if quantum training completed)")
print("  classical_confusion_matrix.png")
print("\nNOTE: This is a research/educational capstone, not evidence that")
print("quantum ML outperforms classical ML on this dataset.")
