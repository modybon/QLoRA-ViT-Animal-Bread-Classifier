# Efficient Fine-tuning of Vision Transformers using QLoRA

COMP 473/6731 Pattern Recognition, Fall 2026, Concordia University
Mohamed Ahmed · Han Yan · Nic Angelo Solina

We classify 37 cat and dog breeds from the Oxford-IIIT Pets dataset with Vision Transformers (ViT), and compare QLoRA against full fine-tuning.

---

## Requirements

- An NVIDIA GPU
- An NVIDIA driver that supports CUDA 13.0
- [Miniconda or Anaconda](https://www.anaconda.com/download)
- Git

---

## Environment setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create the conda environment

Use Python 3.12. Python 3.10 no longer gets pre-built `pyarrow` packages, which breaks the install.

```bash
conda create -n qlora-animal-breed python=3.12 -y
conda activate qlora-animal-breed
```

### 3. Install the packages

```bash
pip install --prefer-binary -r requirements.txt
```

`requirements.txt` installs PyTorch with CUDA 13.0 support from the PyTorch index, and everything else from PyPI. `--prefer-binary` makes pip use pre-built packages instead of compiling from source.

To get exactly the versions the team uses, install from the lock file instead (if present):

```bash
pip install -r requirements-lock.txt
```

### 4. Check the GPU setup

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -m bitsandbytes
```

You should see versions ending in `+cu130`, then `True`, then `SUCCESS!` from bitsandbytes.
A warning that `triton` was not found is expected on Windows and can be ignored.

### 5. Register the environment as a Jupyter kernel

```bash
python -m ipykernel install --user --name qlora-animal-breed --display-name "Python (qlora-animal-breed)"
```

In VS Code: open a notebook, click **Select Kernel** (top right), and choose **Python (qlora-animal-breed)**.
If it does not appear, run **Developer: Reload Window** from the command palette (`Ctrl+Shift+P`).

---

## Dataset

The data is downloaded automatically the first time you run the loading code:

```python
from datasets import load_dataset
hf = load_dataset("timm/oxford-iiit-pet")   # 3,680 train / 3,669 test
```

It is stored in the Hugging Face cache (`C:\Users\<you>\.cache\huggingface\` on Windows, `~/.cache/huggingface/` on Linux and macOS), about 800 MB. It is not committed to git.

We hold out 735 training images for validation with `train_test_split(..., test_size=735, stratify=labels, random_state=42)`.
Keep `random_state=42` so everyone has the same split.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `torch.cuda.is_available()` prints `False`, or versions lack `+cu130` | `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130` |
| `pyarrow requires NumPy 2.0 or newer` | `pip install --prefer-binary -U numpy pandas` |
| `module 'pyarrow' has no attribute 'PyExtensionType'` | `pip install -U --prefer-binary datasets`, then restart the kernel |
| `Failed to build 'pyarrow'` | Make sure the environment uses Python 3.12, then reinstall with `--prefer-binary` |
| Kernel missing in VS Code | Repeat step 5 and reload the window |
| A package was installed but the notebook can't find it | Install from inside the notebook with `%pip install <package>`, then restart the kernel |

---

## Updating dependencies

After adding a package and confirming everything works:

```bash
pip freeze > requirements-lock.txt
```

Then add `--extra-index-url https://download.pytorch.org/whl/cu130` as the first line of `requirements-lock.txt`, and commit it.
