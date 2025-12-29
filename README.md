# Quantum vs Classical Machine Learning – A Comparison
## About the Project
This project is a simple comparison between **classical machine learning models** and a **quantum machine learning model** using a non-linear dataset.  
The main idea is to understand how different models behave, rather than trying to prove that one model is always better than the other.

---

## Why I Did This Project
I worked on this project to:
- Explore **quantum machine learning**, which is an emerging field  
- Compare it with **well-known classical models**  
- Understand the **real limitations** of quantum models in practice  
- Learn how **non-linear datasets** are handled differently by each model  

---

## Dataset Used
- **Moons dataset** (synthetic)  
- **Binary classification** problem  
- **Non-linearly separable** data  

### Train–Test Split
- Training data: **240 samples**  
- Testing data: **60 samples**

This dataset was chosen because it clearly shows the difference between **linear**, **non-linear**, and **quantum** approaches.

---

## Models Implemented

### Classical Models
- **Logistic Regression**  
  Used as a basic linear baseline model.

- **Support Vector Machine (RBF Kernel)**  
  Used to handle non-linear patterns effectively.

### Quantum Model
- **Variational Quantum Classifier (VQC)**  
  Implemented using **PennyLane**, with:
  - Quantum feature maps  
  - Rotation gates  
  - Entanglement using **CNOT** gates  

The model is trained using a **hybrid quantum–classical approach**.

---

## Results
Logistic Regression Accuracy : 0.783
SVM (RBF) Accuracy : 0.900
Quantum VQC Accuracy : 0.55


---

## What I Observed
- Logistic Regression struggles because the dataset is non-linear.  
- SVM performs the best since it is designed to handle such patterns.  
- The quantum model was able to learn basic patterns but did not outperform classical models.

This behavior is expected because current quantum models are limited by:
- Small number of qubits  
- Shallow circuits  
- Optimization challenges  
- NISQ-era hardware limitations  

---

## What I Learned
- Classical machine learning models are still very strong for small datasets.  
- Quantum machine learning is still in an early stage of development.  
- Accuracy alone should not be used to judge quantum models.  
- Understanding **model behavior and limitations** is more important than forcing better results.

---

## Conclusion
This project helped me understand both **classical** and **quantum** approaches to machine learning.  
Even though the quantum model did not achieve the highest accuracy, it gave me valuable insight into how **quantum feature maps** and **variational circuits** work in practice.

---

## Tools & Technologies
- Python  
- Scikit-learn  
- PennyLane  
- NumPy  
- Matplotlib  

---

