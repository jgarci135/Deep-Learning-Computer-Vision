# WSL2 GPU Setup For This Project

This guide sets up GPU training with WSL2 on Windows 10/11 using Ubuntu.

## 1) One-time Windows setup (Admin PowerShell)

Run PowerShell as Administrator and execute:

```powershell
wsl --install -d Ubuntu
wsl --set-default-version 2
```

If WSL is already installed, update it:

```powershell
wsl --update
```

Reboot Windows if prompted.

## 2) Verify NVIDIA driver on Windows

In normal PowerShell:

```powershell
nvidia-smi
```

You should see your GPU listed (GTX 1080 Ti).

## 3) Initialize Ubuntu

Open Ubuntu from Start menu and complete first-time username/password setup.

Then run:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip git
```

## 4) Clone or open project in WSL

Option A (recommended for performance): clone inside Linux home:

```bash
cd ~
git clone https://github.com/jgarci135/Deep-Learning-Computer-Vision.git
cd Deep-Learning-Computer-Vision
```

Option B (no duplicate repo): use your Windows copy via /mnt/c:

```bash
cd /mnt/c/Users/Joshua/coding_projects/git/Deep-Learning-Computer-Vision
```

## 5) Create WSL virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6) Verify TensorFlow GPU access in WSL

```bash
python - << 'PY'
import tensorflow as tf
print('TF:', tf.__version__)
print('Built with CUDA:', tf.test.is_built_with_cuda())
print('GPU devices:', tf.config.list_physical_devices('GPU'))
PY
```

Expected: at least one GPU device listed.

## 7) Run this project pipeline

```bash
python scripts/data_preprocessing.py
python scripts/check_class_balance.py
python scripts/eda_visualization.py
python scripts/model_training.py
python scripts/evaluation.py
```

## 8) If GPU is not detected

- Run `wsl --update` in Admin PowerShell.
- Reboot Windows.
- Verify `nvidia-smi` on Windows still works.
- In Ubuntu, verify `nvidia-smi` also works.
- If still failing, reinstall latest NVIDIA Game Ready or Studio driver.

## Space-saving notes

- With around 100 GB free, you have enough space.
- To minimize extra storage, keep data in Windows path and access via `/mnt/c/...`.
- For best training speed, keep active data and repo in Linux home.
