# transform
# Transformer Model on Wikitext-103 Dataset

This repository implements a Transformer model for language modeling on the **Wikitext-103** dataset. It is designed for full reproducibility, with clear instructions for setup, training, evaluation, and result visualization.

---

## 📁 Project Structure
transform/
├── scripts/
│   └── run.sh                 # One-click script to run training & evaluation
├── data/
│   └── wikitext-103-raw-v1/   # Raw text data (downloaded separately)
├── results/
│   ├── accuracy_loss_comparison.png     
│   ├── loss_comparison.png           
│   └── loss_separate_comparison.png 
├── train.py                   # Training script
├── test.py                    # Evaluation script
├── model.py                   # Transformer model definition
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── LICENSE

---

## ⚙️ Environment Setup

### 1. Clone the repository
```bash
git clone https://github.com/abc1234299/transform.git
cd transform

### 2. Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 Windows:
# venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

### Hardware Requirements
GPU: 至少 12 GB 显存（推荐 RTX 3060 / 3090 / A100）
CUDA: 支持 CUDA 11.8 或更高版本
内存 (RAM): ≥ 16 GB
磁盘空间: ≥ 500 MB（不含数据集）

#设置随机种子以确保结果可复现
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

### train
python train.py \
    --data_dir data/wikitext-103-raw-v1 \
    --batch_size 64 \
    --lr 0.001 \
    --epochs 20 \
    --model_dim 512 \
    --n_layers 6 \
    --n_heads 8 \
    --dropout 0.1 \
    --seed 42 \
    --save_dir results/checkpoints \
    --log_interval 100

### test
python test.py \
    --data_dir data/wikitext-103-raw-v1 \
    --checkpoint results/checkpoints/best_model.pth \
    --batch_size 64 \
    --seed 42
    
### 使用提供的脚本一键完成训练与测试：
bash scripts/run.sh

## ⚙️ Dataset 下载说明
本项目使用 Wikitext-103 数据集。
#请手动下载并解压：
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip -d data/