
import os
import sys
import time
import argparse
import logging
from datetime import datetime
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nnF
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from PIL import Image
from torchvision import transforms as T

from model import CSRNetImproved, CSRNetOriginal, CSRNetMultiScale, create_model
from dataset import ShanghaiTechDatasetImproved, ShanghaiTechDatasetSimple
from losses import CombinedLoss, MSELoss, create_loss_function



class Config:
    def __init__(self):
        # Paths
        self.TRAIN_PATH = "./part_A/train_data"
        self.TEST_PATH = "./part_A/test_data"
        self.CHECKPOINT_DIR = "./checkpoints"
        self.LOG_DIR = "./logs"

        # Model
        self.MODEL_TYPE = 'improved'
        self.BACKBONE = 'vgg19'
        self.USE_ATTENTION = True
        self.USE_BN = False
        self.DROPOUT_RATE = 0.0

        # Training
        self.EPOCHS = 400
        self.BATCH_SIZE = 1
        self.ACCUMULATION_STEPS = 4

        # Optimizer — differential LR: backend gets LR, frontend gets LR/10
        self.OPTIMIZER = 'adamw'
        self.LR = 1e-4
        self.FRONTEND_LR_SCALE = 0.1  # frontend_lr = LR * this
        self.WEIGHT_DECAY = 1e-4
        self.MOMENTUM = 0.95

        # Scheduler
        self.SCHEDULER = 'cosine'
        self.WARMUP_EPOCHS = 10
        self.LR_STEP_SIZE = 50
        self.LR_GAMMA = 0.5
        self.LR_MIN = 1e-7

        # Loss
        self.LOSS_TYPE = 'combined'
        self.MSE_WEIGHT = 1.0
        self.SSIM_WEIGHT = 0.01
        self.COUNT_WEIGHT = 0.01

        # Augmentation
        self.CROP_SIZE = 512
        self.SCALE_RANGE = (0.8, 1.2)
        self.FLIP_PROB = 0.5
        self.COLOR_JITTER = True

        # Hardware
        self.NUM_WORKERS = 4
        self.PIN_MEMORY = True
        self.USE_AMP = True
        self.GRAD_CLIP_NORM = 1.0

        # Validation / checkpointing
        self.VAL_EVERY = 1
        self.SAVE_EVERY = 10

        # Misc
        self.SEED = 42
        self.PRINT_FREQ = 20
        self.RESUME_PATH = None


def setup_logging(log_dir, experiment_name):
    
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    
    
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def set_seed(seed):
    
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def safe_set_matmul_precision():
    
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception as e:
            print(f"[Info] set_float32_matmul_precision skipped: {e}")



@torch.no_grad()
def validate(model, val_loader, device, use_amp=False):
    
    model.eval()
    
    mae_sum = 0.0
    mse_sum = 0.0
    n_samples = 0
    
    for img, target in val_loader:
        img = img.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        
        if device.type == 'cuda' and use_amp:
            with torch.amp.autocast(device_type='cuda'):
                output = model(img)
        else:
            output = model(img)
        

        pred_count = output.sum().item()
        gt_count = target.sum().item()
        
        
        diff = pred_count - gt_count
        mae_sum += abs(diff)
        mse_sum += diff ** 2
        n_samples += 1
    
    mae = mae_sum / max(1, n_samples)
    mse = (mse_sum / max(1, n_samples)) ** 0.5

    return mae, mse


TTA_SCALES = [1.0]  # flip-only by default; add [0.95, 1.0, 1.05] for mild multi-scale
STRIDE = 8

@torch.no_grad()
def validate_tta(model, data_root, device, use_amp=False, scales=None):
    """Validate with Test-Time Augmentation (multi-scale + flip).

    Uses raw images (not the dataloader) so we can resize freely.
    """
    import glob, h5py

    if scales is None:
        scales = TTA_SCALES

    model.eval()
    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_dir = os.path.join(data_root, "images")
    gt_dir = os.path.join(data_root, "ground-truth-h5-s8")
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    mae_sum = 0.0
    mse_sum = 0.0
    n = 0

    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        w = (w // STRIDE) * STRIDE
        h = (h // STRIDE) * STRIDE
        img = img.crop((0, 0, w, h))

        counts = []
        for scale in scales:
            sw = max(STRIDE, (int(w * scale) // STRIDE) * STRIDE)
            sh = max(STRIDE, (int(h * scale) // STRIDE) * STRIDE)
            scaled = img.resize((sw, sh), Image.BILINEAR)

            # Original
            x = tfm(scaled).unsqueeze(0).to(device)
            if device.type == 'cuda' and use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    out = model(x)
            else:
                out = model(x)
            counts.append(out.sum().item())

            # Flipped
            flipped = scaled.transpose(Image.FLIP_LEFT_RIGHT)
            x_f = tfm(flipped).unsqueeze(0).to(device)
            if device.type == 'cuda' and use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    out_f = model(x_f)
            else:
                out_f = model(x_f)
            counts.append(out_f.sum().item())

        pred = np.mean(counts)

        # Load GT
        basename = os.path.basename(img_path).replace('.jpg', '.h5')
        gt_path = os.path.join(gt_dir, basename)
        if os.path.exists(gt_path):
            with h5py.File(gt_path, 'r') as f:
                gt = float(f['density'][:].sum())
            diff = pred - gt
            mae_sum += abs(diff)
            mse_sum += diff ** 2
            n += 1

    mae = mae_sum / max(1, n)
    mse = (mse_sum / max(1, n)) ** 0.5
    return mae, mse



def train_one_epoch(
    model, 
    train_loader, 
    optimizer, 
    criterion, 
    device, 
    epoch,
    config,
    scaler=None,
    logger=None,
):
    
    model.train()
    
    running_loss = 0.0
    running_mse = 0.0
    running_ssim = 0.0
    running_count = 0.0
    
    optimizer.zero_grad(set_to_none=True)
    
    for i, (img, target) in enumerate(train_loader):
        img = img.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
    
        if device.type == 'cuda' and config.USE_AMP and scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                output = model(img)
                
                if config.LOSS_TYPE == 'combined':
                    loss, components = criterion(output, target)
                else:
                    loss = criterion(output, target)
                    components = {'mse': loss.item(), 'ssim': 0, 'count': 0}
                
                loss = loss / config.ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            
            if (i + 1) % config.ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            output = model(img)
            
            if config.LOSS_TYPE == 'combined':
                loss, components = criterion(output, target)
            else:
                loss = criterion(output, target)
                components = {'mse': loss.item(), 'ssim': 0, 'count': 0}
            
            loss = loss / config.ACCUMULATION_STEPS
            loss.backward()
            
            if (i + 1) % config.ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        
    
        running_loss += loss.item() * config.ACCUMULATION_STEPS
        running_mse += components['mse']
        running_ssim += components['ssim']
        running_count += components['count']
        
    
        if (i + 1) % config.PRINT_FREQ == 0:
            avg_loss = running_loss / (i + 1)
            if logger:
                logger.info(
                    f"Epoch [{epoch+1}/{config.EPOCHS}] "
                    f"Step [{i+1}/{len(train_loader)}] "
                    f"Loss: {loss.item()*config.ACCUMULATION_STEPS:.4f} "
                    f"AvgLoss: {avg_loss:.4f}"
                )
            else:
                print(
                    f"Epoch [{epoch+1}/{config.EPOCHS}] "
                    f"Step [{i+1}/{len(train_loader)}] "
                    f"Loss: {loss.item()*config.ACCUMULATION_STEPS:.4f} "
                    f"AvgLoss: {avg_loss:.4f}"
                )
    
    
    if len(train_loader) % config.ACCUMULATION_STEPS != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    avg_loss = running_loss / len(train_loader)
    avg_components = {
        'mse': running_mse / len(train_loader),
        'ssim': running_ssim / len(train_loader),
        'count': running_count / len(train_loader),
    }
    
    return avg_loss, avg_components



def train(config):
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"csrnet_{config.MODEL_TYPE}_{timestamp}"
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    logger = setup_logging(config.LOG_DIR, experiment_name)
    
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Config: {vars(config)}")
    
    
    set_seed(config.SEED)
    
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Device: cuda ({torch.cuda.get_device_name(0)})")
        safe_set_matmul_precision()
    else:
        device = torch.device("cpu")
        logger.warning("Device: cpu (WARNING: training will be very slow)")
        config.USE_AMP = False
    
    
    logger.info("Loading datasets...")
    
    train_ds = ShanghaiTechDatasetImproved(
        config.TRAIN_PATH,
        gt_folder="ground-truth-h5-s8",
        training=True,
        crop_size=config.CROP_SIZE,
        scale_range=config.SCALE_RANGE,
        flip_prob=config.FLIP_PROB,
        color_jitter=config.COLOR_JITTER,
    )
    
    val_ds = ShanghaiTechDatasetImproved(
        config.TEST_PATH,
        gt_folder="ground-truth-h5-s8",
        training=False,
    )
    
    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    
    
    logger.info(f"Creating model: {config.MODEL_TYPE}")
    
    model = create_model(
        config.MODEL_TYPE,
        load_weights=True,
        backbone=config.BACKBONE,
        use_attention=config.USE_ATTENTION,
        use_bn=config.USE_BN,
        dropout_rate=config.DROPOUT_RATE,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    
    
    if config.LOSS_TYPE == 'combined':
        criterion = CombinedLoss(
            mse_weight=config.MSE_WEIGHT,
            ssim_weight=config.SSIM_WEIGHT,
            count_weight=config.COUNT_WEIGHT,
        )
    else:
        criterion = create_loss_function(config.LOSS_TYPE)
    
    logger.info(f"Loss function: {config.LOSS_TYPE}")
    
    
    # Differential LR: lower LR for pretrained frontend, higher for backend/head
    frontend_params = list(model.frontend.parameters())
    frontend_ids = {id(p) for p in frontend_params}
    backend_params = [p for p in model.parameters() if id(p) not in frontend_ids]

    frontend_lr = config.LR * config.FRONTEND_LR_SCALE
    param_groups = [
        {'params': frontend_params, 'lr': frontend_lr},
        {'params': backend_params, 'lr': config.LR},
    ]

    if config.OPTIMIZER == 'adam':
        optimizer = optim.Adam(param_groups, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(param_groups, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    else:
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")

    logger.info(f"Optimizer: {config.OPTIMIZER}, backend LR: {config.LR}, frontend LR: {frontend_lr}")

    # Scheduler: warmup + cosine (or other)
    if config.SCHEDULER == 'cosine':
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=config.EPOCHS - config.WARMUP_EPOCHS, eta_min=config.LR_MIN,
        )
        scheduler = cosine_scheduler  # warmup handled manually in loop
    elif config.SCHEDULER == 'step':
        scheduler = StepLR(optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA)
    elif config.SCHEDULER == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=config.LR_GAMMA, patience=10, min_lr=config.LR_MIN,
        )
    else:
        scheduler = None

    logger.info(f"Scheduler: {config.SCHEDULER}, warmup: {config.WARMUP_EPOCHS} epochs")
    
    
    if config.USE_AMP and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None
    
    
    best_mae = float('inf')
    start_epoch = 0
    history = {
        'train_loss': [],
        'val_mae': [],
        'val_mse': [],
        'lr': [],
    }
    
    # ── Resume from checkpoint ─────────────────────────────────
    if config.RESUME_PATH and os.path.isfile(config.RESUME_PATH):
        logger.info(f"Resuming from checkpoint: {config.RESUME_PATH}")
        checkpoint = torch.load(config.RESUME_PATH, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_mae = checkpoint.get('best_mae', float('inf'))
        
        # Fast-forward scheduler to correct epoch (only post-warmup steps)
        if scheduler and config.SCHEDULER != 'plateau':
            steps_past_warmup = max(0, start_epoch - config.WARMUP_EPOCHS)
            for _ in range(steps_past_warmup):
                scheduler.step()
        
        logger.info(f"Resumed from epoch {start_epoch}, best MAE so far: {best_mae:.2f}")
    elif config.RESUME_PATH:
        logger.warning(f"Checkpoint not found: {config.RESUME_PATH}, training from scratch.")
    
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(start_epoch, config.EPOCHS):
        epoch_start = time.time()
        
        
        train_loss, train_components = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, config, scaler, logger
        )
        
        history['train_loss'].append(train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        
        do_val = ((epoch + 1) % config.VAL_EVERY == 0) or (epoch == 0) or (epoch == config.EPOCHS - 1)
        
        if do_val:
            val_mae, val_mse = validate(model, val_loader, device, config.USE_AMP)
            history['val_mae'].append(val_mae)
            history['val_mse'].append(val_mse)
            
            logger.info(
                f"Epoch [{epoch+1}/{config.EPOCHS}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val MAE: {val_mae:.2f} | Val MSE: {val_mse:.2f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
            
            
            if config.SCHEDULER == 'plateau':
                scheduler.step(val_mae)
            
            
            if val_mae < best_mae:
                best_mae = val_mae
                checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mae': best_mae,
                    'config': vars(config),
                }, checkpoint_path)
                logger.info(f"New best model saved! MAE: {best_mae:.2f}")
        else:
            logger.info(
                f"Epoch [{epoch+1}/{config.EPOCHS}] "
                f"Train Loss: {train_loss:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
        
        
        # LR scheduling: warmup then cosine/step
        if epoch < config.WARMUP_EPOCHS:
            # Linear warmup: scale from 0 to target LR
            warmup_factor = (epoch + 1) / config.WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                if pg is optimizer.param_groups[0]:  # frontend
                    pg['lr'] = config.LR * config.FRONTEND_LR_SCALE * warmup_factor
                else:  # backend
                    pg['lr'] = config.LR * warmup_factor
        elif scheduler and config.SCHEDULER != 'plateau':
            scheduler.step()
        
    
        if (epoch + 1) % config.SAVE_EVERY == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mae': best_mae,
            }, checkpoint_path)
        
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch time: {epoch_time:.1f}s")
    
    
    total_time = time.time() - start_time
    logger.info(f"\nTraining complete!")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Best MAE (single-pass): {best_mae:.2f}")

    # Final TTA evaluation on best model
    logger.info("Running TTA evaluation on best model...")
    best_ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
        tta_mae, tta_mse = validate_tta(model, config.TEST_PATH, device, config.USE_AMP)
        logger.info(f"Best MAE (TTA): {tta_mae:.2f} | RMSE (TTA): {tta_mse:.2f}")
    
    
    history_path = os.path.join(config.LOG_DIR, f"{experiment_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    return best_mae, history


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train CSRNet for crowd counting')
    
    # Model
    parser.add_argument('--model', type=str, default='improved',
                        choices=['original', 'improved', 'multiscale'])
    parser.add_argument('--backbone', type=str, default='vgg19',
                        choices=['vgg16', 'vgg19'])
    parser.add_argument('--no-attention', action='store_true')
    parser.add_argument('--use-bn', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.0)
    
    # Training
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--frontend-lr-scale', type=float, default=0.1,
                        help='Frontend LR = LR * this (default: 0.1)')
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'plateau', 'none'])

    # Loss
    parser.add_argument('--loss', type=str, default='combined',
                        choices=['mse', 'combined', 'bayesian', 'adaptive'])
    parser.add_argument('--ssim-weight', type=float, default=0.01)
    parser.add_argument('--count-weight', type=float, default=0.01)

    # Data
    parser.add_argument('--train-path', type=str, default='./part_A/train_data')
    parser.add_argument('--test-path', type=str, default='./part_A/test_data')
    parser.add_argument('--crop-size', type=int, default=512)
    parser.add_argument('--no-augmentation', action='store_true')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint (.pth) to resume training from')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--num-workers', type=int, default=4)
    
    args = parser.parse_args()
    
    # Update config
    config = Config()
    config.MODEL_TYPE = args.model
    config.BACKBONE = args.backbone
    config.USE_ATTENTION = not args.no_attention
    config.USE_BN = args.use_bn
    config.DROPOUT_RATE = args.dropout
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LR = args.lr
    config.FRONTEND_LR_SCALE = args.frontend_lr_scale
    config.WARMUP_EPOCHS = args.warmup_epochs
    config.OPTIMIZER = args.optimizer
    config.SCHEDULER = args.scheduler
    config.LOSS_TYPE = args.loss
    config.SSIM_WEIGHT = args.ssim_weight
    config.COUNT_WEIGHT = args.count_weight
    config.TRAIN_PATH = args.train_path
    config.TEST_PATH = args.test_path
    config.CROP_SIZE = args.crop_size
    if args.no_augmentation:
        config.COLOR_JITTER = False
        config.FLIP_PROB = 0.0
        config.SCALE_RANGE = (1.0, 1.0)
    config.SEED = args.seed
    config.USE_AMP = not args.no_amp
    config.NUM_WORKERS = args.num_workers
    config.RESUME_PATH = args.resume
    
    # Check paths
    if not os.path.exists(config.TRAIN_PATH) or not os.path.exists(config.TEST_PATH):
        print("Error: Dataset paths not found.")
        print(f"TRAIN_PATH: {config.TRAIN_PATH}")
        print(f"TEST_PATH: {config.TEST_PATH}")
        sys.exit(1)
    
    # Train
    train(config)


if __name__ == "__main__":
    main()