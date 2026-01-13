# AMLS Assignment 25/26 - SN25129580

This project implements **two models** for the **BreastMNIST** dataset:

- **Task A (Classical ML):** feature preprocessing + a classical classifier(SVM) (see `A/model_a.py`)
- **Task B (Deep Learning):** a CNN trained with PyTorch (see `B/model_b.py`)

Run everything from the project root with:

```bash
python main.py
```


## 1. Repository structure

```
AMLS_25_26_SN25129580/
├─ A/
│  └─ model_a.py          # Task A implementation (classical ML: SVM)
├─ B/
│  └─ model_b.py          # Task B implementation (deep learning)
├─ Datasets/              
│  └─ BreastMNIST/        
├─ main.py                # Entry point: runs Task A then Task B
└─ README.md
```

---

## 2. What each file does

### `main.py`
- Locates the dataset root folder and passes it to both tasks.
- Runs the full pipeline:
  1) Train + test **Task A**
  2) Train + test **Task B**
- Prints test accuracy for each task.

### `A/model_a.py`
- Defines class **`TaskA`** with:
  - `train()` — trains the classical ML model
  - `test()` — evaluates on the test set

### `B/model_b.py`
- Defines class **`TaskB`** (and the CNN model) with:
  - `train()` — trains the neural network
  - `test()` — evaluates on the test set

---

## 3. Dependencies

Python 3.9+ recommended.

Install requirements:

```bash
pip install numpy scipy scikit-learn torch torchvision medmnist pillow
```


---

## 4. Dataset placement (compatibility)

The code is intended to work with either of the following dataset locations:

- `Datasets/BreastMNIST/breastmnist.npz`
- `Datasets/breastmnist.npz`

In the marking environment, the dataset is expected to be copied into `Datasets/` by the assessors.

If you are running locally and want to download automatically via `medmnist`, you can create the folder first:

```bash
mkdir -p Datasets/BreastMNIST
```

Then run:

```bash
python main.py
```

---

## 5. How to run

From the repository root:

```bash
python main.py
```

The console will show:
- Task A training and testing
- Task B training and testing
- Final test accuracy for both tasks

To test different parameters impacts:
```bash
python test.py
```

The code will test all the parameters of it and export the results as a txt file.
