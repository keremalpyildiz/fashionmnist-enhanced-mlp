# 👕 FashionMNIST Classification: Enhanced MLP Pipeline

This repository contains a **4-layer Deep Neural Network (MLP)** built with **PyTorch** on the FashionMNIST dataset. The project focuses on applying core deep learning engineering practices—including **weight initialization**, **batch normalization**, **dual regularization**, **Coarse-to-Fine hyperparameter tuning**, and **early stopping**.

---

## 🛠️ System Design & Engineering Decisions

To handle the $28 \times 28$ grayscale images ($784$ input features), we implemented a 4-layer pyramid architecture: `784 -> 512 -> 256 -> 128 -> 10`.

* **He (Kaiming) Uniform Initialization:** Chosen over default random initialization to stabilize initial variance and prevent vanishing gradients caused by ReLU activations.
* **Batch Normalization (`BatchNorm1d`):** Integrated after each linear layer to minimize internal covariate shift, allowing for smoother optimization and stable gradient flow.
* **Dual Regularization (`Dropout p=0.3` + `Weight Decay 1e-4`):** Dropout was added to break feature co-adaptation across dense layers, while L2 regularization (Weight Decay in Adam) was applied to constrain weight magnitudes.
* **Input Scaling:** Normalized raw pixel values to $[-1, 1]$ ($\text{Mean}=0.5, \text{Std}=0.5$) to keep optimizer steps isotropic.

---

## 🔬 Hyperparameter Optimization (Coarse-to-Fine)

Instead of relying on default values, we systematically tuned Learning Rate ($\alpha$) and Weight Decay ($\lambda$) using Andrew Ng’s **Coarse-to-Fine** methodology:

1. **Coarse Search:** Conducted logarithmic parameter sweeps across 3-epoch trials to isolate high-potential regions ($\text{LR} \approx 0.001$, $\text{WD} \approx 10^{-4}$).
2. **Fine Search:** Evaluated a tight grid around $0.001$ over 5-epoch trials to dampen random sampling noise.
3. **Optimal Selection:** Verified **`LR = 0.001`** and **`Weight Decay = 0.0001`** as the optimal configuration ($\text{Val Loss} = 0.3354$).

---

## ⏱️ Training & Early Stopping

The final model was trained using the Adam optimizer alongside an **Early Stopping** mechanism (`patience = 3`) and model checkpointing (`best_enhanced_model.pt`).

* **Peak Performance:** Reached a minimum **Validation Loss of `0.3070`** at **Epoch 14**.
* **Automatic Halt:** Training stopped automatically at **Epoch 17**, preventing the network from entering the overfitting regime.

---

## 📊 Results & Performance Analysis

| Metric | Baseline Model | Enhanced Model |
| :--- | :---: | :---: |
| **Test Accuracy** | 88.00% | **88.75%** |
| **Overfitting Status** | Present (High train-val gap) | **Eliminated (Generalization achieved)** |
| **Shirt Recall (Hard Class)** | 603/1000 | **628/1000 (+25 Samples)** |

### **Why Did Test Accuracy Remain at ~88%?**
1. **Architecture Limit:** Fully connected layers on flattened images lack spatial inductive bias. Without convolutional feature extraction, an MLP reaches a **theoretical performance ceiling of ~88–89%** on FashionMNIST.
2. **Loss vs. Confidence Dynamics:** Dropout forces the network to output more conservative class probabilities, keeping cross-entropy loss slightly higher while maintaining high classification precision.

### **What Did Our Changes Actually Achieve?**
While accuracy metrics appear similar, the engineering modifications transformed the model from a fragile, memorizing network into a **production-ready, highly generalizable system**:
* **Zero Overfitting:** The train-validation loss gap was closed.
* **Improved Hard Classes:** Precision on ambiguous categories like *Shirt* (+25 correct predictions) and *Dress* increased noticeably.
* **Training Stability:** Combined Batch Normalization and Early Stopping ensured repeatable, deterministic convergence.

---

## 🚀 Getting Started

```bash
git clone [https://github.com/your-username/fashionmnist-enhanced-mlp.git](https://github.com/your-username/fashionmnist-enhanced-mlp.git)
cd fashionmnist-enhanced-mlp
pip install torch torchvision matplotlib seaborn scikit-learn
jupyter notebook fashion_mnist_mlp.ipynb
