# ================================
# IMPORTS
# ================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import pennylane as qml
from pennylane import numpy as pnp


# ================================
# DATASET GENERATION
# ================================
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Dataset created")
print("Train:", X_train.shape, "Test:", X_test.shape)


# ================================
# DATASET VISUALIZATION
# ================================
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
plt.title("Moons Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# ================================
# HELPER: DECISION BOUNDARY
# ================================
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


# ================================
# CLASSICAL MODEL 1: LOGISTIC REGRESSION
# ================================
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)

print("Logistic Regression Accuracy:", acc_lr)

plot_decision_boundary(
    log_reg, X_train, y_train,
    "Logistic Regression Decision Boundary"
)


# ================================
# CLASSICAL MODEL 2: SVM (RBF)
# ================================
svm = SVC(kernel="rbf", gamma="scale")
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)

print("SVM (RBF) Accuracy:", acc_svm)

plot_decision_boundary(
    svm, X_train, y_train,
    "SVM (RBF Kernel) Decision Boundary"
)


# ================================
# QUANTUM MODEL (VQC)
# ================================
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)


# Feature Map
def feature_map(x):
    qml.RY(x[0], wires=0)
    qml.RY(x[1], wires=1)
    qml.CNOT(wires=[0, 1])


# Variational Circuit
def variational_circuit(weights):
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.CNOT(wires=[0, 1])


# Quantum Circuit
@qml.qnode(dev)
def quantum_circuit(x, weights):
    feature_map(x)
    variational_circuit(weights)
    return qml.expval(qml.PauliZ(0))


# Loss Function (AUTOGRAD SAFE)
def loss(weights, X, y):
    targets = 1 - 2 * y   # {0,1} → {+1,-1}
    preds = pnp.array([quantum_circuit(x, weights) for x in X])
    return pnp.mean((preds - targets) ** 2)


# ================================
# TRAIN QUANTUM MODEL
# ================================
pnp.random.seed(42)
weights = pnp.random.randn(2, requires_grad=True)

optimizer = qml.AdamOptimizer(stepsize=0.1)

for epoch in range(50):
    weights = optimizer.step(lambda w: loss(w, X_train, y_train), weights)
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss(weights, X_train, y_train):.4f}")


# ================================
# QUANTUM ACCURACY
# ================================
def predict_quantum(x, weights):
    return 1 if quantum_circuit(x, weights) < 0 else 0

y_pred_q = np.array([predict_quantum(x, weights) for x in X_test])
acc_q = np.mean(y_pred_q == y_test)

print("Quantum Model Accuracy:", acc_q)


# ================================
# QUANTUM DECISION BOUNDARY
# ================================
def plot_quantum_boundary(X, y, weights):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([predict_quantum(p, weights) for p in grid])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
    plt.title("Quantum Decision Boundary (VQC)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_quantum_boundary(X_train, y_train, weights)


# ================================
# FINAL COMPARISON
# ================================
print("\n===== FINAL RESULTS =====")
print(f"Logistic Regression Accuracy : {acc_lr:.3f}")
print(f"SVM (RBF) Accuracy           : {acc_svm:.3f}")
print(f"Quantum VQC Accuracy         : {acc_q:.3f}")
