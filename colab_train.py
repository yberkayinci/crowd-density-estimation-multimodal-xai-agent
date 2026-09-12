"""
colab_train.py — Google Colab'da eğitimi başlatmak için hazır script.

Bu dosyadaki hücreleri sırasıyla Colab notebook'a kopyalayın.
GPU: T4 (ücretsiz) / L4 / A100 (Colab Pro)

============================================================
ADIM ADIM KULLANIM:
  1) Proje klasörünü ZIP'leyip Google Drive'a yükleyin
     VEYA klasörü olduğu gibi Drive'a kopyalayın.
  2) Aşağıdaki hücreleri sırasıyla çalıştırın.
============================================================
"""

# ══════════════════════════════════════════════════════════════
# CELL 1: Google Drive'ı bağla
# ══════════════════════════════════════════════════════════════
# from google.colab import drive
# drive.mount('/content/drive')

# ══════════════════════════════════════════════════════════════
# CELL 2: Proje dizinine geç
# ══════════════════════════════════════════════════════════════
# import os
# PROJECT_DIR = "/content/drive/MyDrive/LLM Asst. Dense Crowd Counting"
# os.chdir(PROJECT_DIR)
# print("Çalışma dizini:", os.getcwd())
# print("Dosyalar:", os.listdir("."))

# ══════════════════════════════════════════════════════════════
# CELL 3: Gerekli kütüphaneleri kur
# ══════════════════════════════════════════════════════════════
# !pip install -q h5py scipy tqdm pillow matplotlib

# ══════════════════════════════════════════════════════════════
# CELL 4: GPU kontrolü
# ══════════════════════════════════════════════════════════════
# import torch
# print(f"PyTorch  : {torch.__version__}")
# print(f"CUDA     : {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"GPU      : {torch.cuda.get_device_name(0)}")
#     props = torch.cuda.get_device_properties(0)
#     print(f"VRAM     : {props.total_mem / 1e9:.1f} GB")

# ══════════════════════════════════════════════════════════════
# CELL 5: (İLK KEZ İSE) Density map oluştur
#   Eğer part_A/train_data/ground-truth-h5-s8 klasörü zaten
#   varsa bu hücreyi atlayabilirsiniz.
# ══════════════════════════════════════════════════════════════
# !python preprocessing.py --stride 8 --adaptive --verify

# ══════════════════════════════════════════════════════════════
# CELL 6: Eğitimi başlat
# ══════════════════════════════════════════════════════════════
#
# ---------- SEÇ-1: Terminal komutu olarak ----------
#
# A100 / L4 GPU (güçlü GPU, bol VRAM):
# !python train.py \
#   --model improved \
#   --epochs 300 \
#   --batch-size 1 \
#   --lr 1e-5 \
#   --optimizer adam \
#   --scheduler cosine \
#   --loss combined \
#   --ssim-weight 0.01 \
#   --count-weight 0.001 \
#   --crop-size 256 \
#   --num-workers 2 \
#   --seed 42
#
# T4 GPU (ücretsiz Colab):
# !python train.py \
#   --model improved \
#   --epochs 300 \
#   --batch-size 1 \
#   --lr 1e-5 \
#   --optimizer adam \
#   --scheduler cosine \
#   --loss combined \
#   --ssim-weight 0.01 \
#   --count-weight 0.001 \
#   --crop-size 256 \
#   --num-workers 2 \
#   --seed 42
#
# ---------- SEÇ-2: Python içinden çalıştır ----------
#   (Colab bazen !python komutunu keserse bu yöntem
#    daha güvenilir çalışır.)
#
# import sys
# sys.argv = [
#     'train.py',
#     '--model', 'improved',
#     '--epochs', '300',
#     '--batch-size', '1',
#     '--lr', '1e-5',
#     '--optimizer', 'adam',
#     '--scheduler', 'cosine',
#     '--loss', 'combined',
#     '--ssim-weight', '0.01',
#     '--count-weight', '0.001',
#     '--crop-size', '256',
#     '--num-workers', '2',
#     '--seed', '42',
# ]
# from train import main
# main()

# ══════════════════════════════════════════════════════════════
# CELL 6b: MEVCUT CHECKPOINT'TEN DEVAM ET (resume)
#   best_model.pth zaten varsa eğitime kaldığı yerden devam eder.
# ══════════════════════════════════════════════════════════════
#
# !python train.py \
#   --resume best_model.pth \
#   --model improved \
#   --epochs 300 \
#   --batch-size 1 \
#   --lr 1e-5 \
#   --optimizer adam \
#   --scheduler cosine \
#   --loss combined \
#   --ssim-weight 0.01 \
#   --count-weight 0.001 \
#   --crop-size 256 \
#   --num-workers 2 \
#   --seed 42

# ══════════════════════════════════════════════════════════════
# CELL 7: (Opsiyonel) Eğitim bitince modeli Drive'a kaydet
# ══════════════════════════════════════════════════════════════
# import shutil
# shutil.copy("best_model.pth",
#             "/content/drive/MyDrive/best_model_colab.pth")
# print("Model Drive'a kopyalandı!")

# ══════════════════════════════════════════════════════════════
# CELL 8: (Opsiyonel) Test et
# ══════════════════════════════════════════════════════════════
# from train import Config, train, validate, set_seed
# from model import create_model
# from dataset import ShanghaiTechDatasetImproved
# from torch.utils.data import DataLoader
# import torch
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# model = create_model('improved',
#                      load_weights=False,
#                      use_attention=True).to(device)
#
# ckpt = torch.load("best_model.pth", map_location=device, weights_only=False)
# model.load_state_dict(ckpt['model_state_dict'])
# print(f"Loaded model — Epoch: {ckpt.get('epoch','?')}, Best MAE: {ckpt.get('best_mae','?')}")
#
# val_ds = ShanghaiTechDatasetImproved("./part_A/test_data",
#                                       gt_folder="ground-truth-h5-s8",
#                                       training=False)
# val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)
# mae, mse = validate(model, val_loader, device, use_amp=True)
# print(f"Test MAE: {mae:.2f}  |  Test MSE (RMSE): {mse:.2f}")
