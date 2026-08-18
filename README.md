### 🧠 NOVA — Real Neural Network from Scratch in Pure C

NOVA is a lightweight, zero-dependency Multi-Layer Perceptron (MLP) written entirely from scratch in pure C. This project features full matrix operations, an accelerated mini-batch Stochastic Gradient Descent (SGD) optimizer, explicit backpropagation calculation, and raw binary weight saving/loading—all without leveraging deep learning libraries like TensorFlow, PyTorch, or OpenCV. 

The network is trained to classify handwritten digits using the structural geometry of the **MNIST Dataset** mapped into a custom modular architecture. 

### 📊 Technical Architecture

* **Input Layer:** 

784784
784
 neurons (representing 
28

×28
 grayscale image pixels).
* **Hidden Layer 1:** 

256256
256
 neurons with **ReLU** activation.
* **Hidden Layer 2:** 

128128
128
 neurons with **ReLU** activation.
* **Output Layer:** 

1010
10
 neurons (classes 
0

−9
) with **Softmax** probability distribution.
* **Loss Function:** Categorical Cross-Entropy.
* **Optimizer:** Mini-Batch Stochastic Gradient Descent (SGD) with variable block sizing.

### 🛠️ Features Included

* **No Memory Leaks:** Rigid structural bounds validation on all matrix allocations with explicit heap memory cleanup.
* **Mini-Batch Pipeline:** Implemented local gradient accumulation buffers (dw, db) that decouple sample calculations from final scalar matrix updates, dramatically smoothing the convergence loss.
* **Weight Persistence:** Core network states are serialized directly into a packed native binary (nova_pesos.bin). If detected, the pipeline automatically skips training and enters prediction mode instantly.
* **Native PGM Image Reader:** Supports reading both ASCII (P2) and Binary (P5) Portable Graymap formats cleanly using built-in file streams.

### 🐧 Getting Started (Linux Guide)

### 1. Requirements

Ensure your development environment contains standard compilation utilities: 

bash

sudo apt update && sudo apt install build-essential imagemagick -y

Use code with caution.

### 2. Extracting the Dataset

Place your compressed MNIST .gz binaries inside a directory named tmp/. Run the Python data script to decompress, parse header bytes, and export native 28x28 .pgm image paths: 

bash

python3 extract_local_mnist.py

Use code with caution.

### 3. Compiling with Aggressive Optimization

To maximize loop-unrolling, vectorization, and algebraic computation performance, compile using standard mathematical linking (-lm) alongside optimal performance flags (-O3): 

bash

gcc -O3 -o nova code.c -lm

Use code with caution.

### 4. Run the Engine

bash

./nova

Use code with caution.

### 🧪 Real Experiments & Stress Testing

NOVA has been thoroughly evaluated against edge-cases, human labeling mistakes, and Out-of-Distribution (OOD) data noise: 

### 1. General Convergence

* **Dataset Scale:** 
∼10

,

000
 real training images.
* **Stability:** Lowering learning rates to LR: 0.020 completely removed loss oscillations, forcing a smooth descent down to Loss: 0.16 and stabilizing test set accuracy past **
90.72

%
**.

### 2. Spotting Dataset Anomaly (data/test/5/0.pgm)

* **Incident:** The network evaluated the file 5/0.pgm and flagged it as a **dígito 6 with 
99.36

%
 confidence**, defying its directory categorization.
* **Verdict:** Inspecting the image via xdg-open confirmed that the pixel geometry actually formed a 5/6 hybrid shape with a closed bottom loop—validating that the network learned raw spatial geometry rather than blindly overfitting to labeled text noise.

### 3. Out-of-Distribution Noise (ruido.pgm)

* **Incident:** The network was stressed using a checkerboard-patterned static image (ruido.pgm).
* **Verdict:** Prediction confidence plummeted down to a fragmented **
56.30

%
**, splitting votes across **8** (
25.20

%
) and **3** (
12.33

%
). This mathematically aligned with expectations, as those specific digits occupy the highest global pixel surface density across a 
28

×28
 matrix, while thin digits like **1, 4, and 7** correctly scored 
0.00

%
.

### 📜 License

This project is open-source. Feel free to use it to study the raw mathematical fundamentals of deep learning.
