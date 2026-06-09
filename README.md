# neural-network-from-scratch

A lightweight, high-performance Deep Learning engine written in **C++20** from scratch, using raw memory management and custom matrix optimization pipelines. The engine relies entirely on **BitMth**, a custom low-level linear algebra library tailored for multi-dimensional matrix operations and efficient backpropagation.

---

## 🚀 Performance Snapshot

* **Dataset:** MNIST Handwritten Digits (60,000 training samples, 10,000 test samples).
* **Network Architecture:** `784 (Input) ➔ 128 (ReLU) ➔ 64 (ReLU) ➔ 10 (Sigmoid/MSE Output)`.
* **Execution Layout:** Row-major mini-batch gradient descent (Batch Size: 64).
* **Evaluation Accuracy:** **`97.37%`** true accuracy achieved in 5 epochs.

---

## 📂 Project Structure

```text
├── CMakeLists.txt              # Root build configuration (C++20, -O3 optimizations)
├── run                         # Automation script for building and execution
├── datasets/                   # Binary ubyte MNIST raw storage
│   ├── train-images-idx3-ubyte
│   └── train-labels-idx1-ubyte
├── examples/
│   └── main.cpp                # Core training pipeline, hyperparameter loops & evaluation
├── include/
│   ├── nn/
│   │   ├── Layer/
│   │   │   ├── ILayer.hpp      # Polimorphic layer interface (forward/backward/update)
│   │   │   └── DenseLayer.hpp  # Fully Connected Layer with manual matrix broadcasting
│   │   └── NeuralNetwork.hpp   # Network sequential container and training core choreographed
│   └── utils/
│       └── readDataSet.hpp     # Fast IDX binary dataset parser
└── third_party/
    └── BitMth/                 # Underlying Linear Algebra Submodule
        ├── include/BitMth/
        │   ├── linalg/         # Optimized continuous flat memory Matrix implementation
        │   ├── ia/             # Loss functions (MSE) and activation derivatives
        │   └── random/         # Xavier/He parameter initialization tools
        └── CMakeLists.txt
```

## 🛠️ Key Technical Features

* **Row-Major Memory Alignment:** Matrices map to an optimized unidimensional continuous buffer (`m[i * cols + j]`), guaranteeing strict L1/L2 cache locality during massive multi-threaded or sequential batch dot products.
* **Native Matrix Broadcasting:** Custom implementation of row-vector diffusion (`addRowVector`) to natively scale layer biases ($1 \times N$) against floating dynamic mini-batches ($64 \times N$) safely checking dimensions under aggressive custom validation macros.
* **Compilation Flag Tuning:** Tailored compilation builds leveraging `-O3` vectorizations and `-ffast-math` to drop floating-point precision safety barriers in favor of raw hardware execution speed.

---

## ⚙️ Requirements & Installation

### Dependencies

* **Compiler:** `GCC` or `Clang` with native support for **C++20** standard.
* **Build System:** `CMake` (version 3.15 or higher) and `GNU Make`.

### Building and Running

Since this project relies on **BitMth** as an underlying submodule dependency, you must clone the repository recursively to fetch all components:

```bash
# Clone the repository including all nested dependencies
git clone --recursive https://github.com/MaisenRv/neural-network-from-scratch.git

# Move into the project directory
cd neural-network-from-scratch

# Grant execution permissions to the script
chmod +x run

# Build the framework and run the MNIST training pipeline
./run

```

Note: If you already cloned the project without the recursive flag, you can fetch the missing dependencies by running `git submodule update --init --recursive` inside the root folder.

## Training Logs

Upon starting execution, the IDX binaries are un-packaged and loaded straight into `BitMth` core tensor buffers. The network tracks execution stability dynamically across training mini-batches:

```text
Cargando dataset de entrenamiento en matrices de BitMth...
Iniciando entrenamiento...
Epoch 1/5 | Avg Loss: 0.012391
Epoch 2/5 | Avg Loss: 0.004432
Epoch 3/5 | Avg Loss: 0.003102
Epoch 4/5 | Avg Loss: 0.002404
Epoch 5/5 | Avg Loss: 0.001969

Cargando dataset de validación (Test)...

--- RESULTADO DE LA EVALUACIÓN ---
Muestras correctas: 9737 / 10000
Precisión Final (Accuracy): 97.37%
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

* **MaisenRv** - *Main Developer / Core Architecture* - [MaisenRv](https://github.com/MaisenRv)

---
