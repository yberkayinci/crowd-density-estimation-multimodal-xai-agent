import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
import torch.nn.functional as F


def get_image_paths(data_root):
    
    return sorted(glob.glob(os.path.join(data_root, "images", "*.jpg")))


class ShanghaiTechDatasetImproved(Dataset):
    
    def __init__(
        self,
        data_root,
        gt_folder="ground-truth-h5-s8",
        training=True,
        crop_size=256,
        scale_range=(0.8, 1.2),
        flip_prob=0.5,
        color_jitter=True,
        blur_prob=0.1,
        output_stride=8,
    ):
        
        self.data_root = data_root
        self.gt_folder = gt_folder
        self.training = training
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.color_jitter = color_jitter
        self.blur_prob = blur_prob
        self.output_stride = output_stride
        
        self.img_paths = get_image_paths(data_root)
        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No .jpg images found under: {os.path.join(data_root, 'images')}")
        
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        
        self.color_transform = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )
    
    def __len__(self):
        return len(self.img_paths)
    
    def _load_image_and_target(self, idx):
        
        img_path = self.img_paths[idx]
        
        img = Image.open(img_path).convert("RGB")
        
        gt_path = img_path.replace("images", self.gt_folder).replace(".jpg", ".h5")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Missing GT file: {gt_path}")
        
        with h5py.File(gt_path, "r") as hf:
            target = np.asarray(hf["density"], dtype=np.float32)
        
        return img, target
    
    def _random_horizontal_flip(self, img, target):
        
        if random.random() < self.flip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            target = np.fliplr(target).copy()
        return img, target
    
    def _random_crop(self, img, target):
       
        w, h = img.size
        target_h, target_w = target.shape
        
        
        crop_h = min(self.crop_size, h)
        crop_w = min(self.crop_size, w)
        crop_h = (crop_h // self.output_stride) * self.output_stride
        crop_w = (crop_w // self.output_stride) * self.output_stride
        
        if crop_h < self.output_stride or crop_w < self.output_stride:
            
            return img, target
        
    
        x = random.randint(0, w - crop_w)
        y = random.randint(0, h - crop_h)
        
        
        x = (x // self.output_stride) * self.output_stride
        y = (y // self.output_stride) * self.output_stride
        
    
        img = img.crop((x, y, x + crop_w, y + crop_h))
        
    
        target_x = x // self.output_stride
        target_y = y // self.output_stride
        target_crop_w = crop_w // self.output_stride
        target_crop_h = crop_h // self.output_stride
        
        target = target[target_y:target_y + target_crop_h, 
                       target_x:target_x + target_crop_w].copy()
        
        return img, target
    
    def _random_scale(self, img, target):
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        
        w, h = img.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        

        new_w = (new_w // self.output_stride) * self.output_stride
        new_h = (new_h // self.output_stride) * self.output_stride
        
        if new_w < self.output_stride or new_h < self.output_stride:
            return img, target
        
        
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        
        target_new_h = new_h // self.output_stride
        target_new_w = new_w // self.output_stride
        
    
        target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)
        target_scaled = F.interpolate(
            target_tensor, 
            size=(target_new_h, target_new_w), 
            mode='bilinear',
            align_corners=False
        )
        
        
        original_count = target.sum()
        target = target_scaled.squeeze().numpy()
        current_count = target.sum()
        
        if current_count > 0:
            target *= (original_count / current_count)
        
        return img, target
    
    def _apply_color_jitter(self, img):
        
        if self.color_jitter and random.random() < 0.5:
            img = self.color_transform(img)
        return img
    
    def _apply_gaussian_blur(self, img):
        
        if random.random() < self.blur_prob:
            radius = random.uniform(0.5, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img
    
    def __getitem__(self, idx):
       
        img, target = self._load_image_and_target(idx)
        
        if self.training:
           
            img, target = self._random_scale(img, target)
            img, target = self._random_crop(img, target)
            img, target = self._random_horizontal_flip(img, target)
            img = self._apply_color_jitter(img)
            img = self._apply_gaussian_blur(img)
        else:
           
            w, h = img.size
            new_w = (w // self.output_stride) * self.output_stride
            new_h = (h // self.output_stride) * self.output_stride
            
            if new_w != w or new_h != h:
                img = img.crop((0, 0, new_w, new_h))
                target_new_h = new_h // self.output_stride
                target_new_w = new_w // self.output_stride
                target = target[:target_new_h, :target_new_w].copy()
        
        
        img = transforms.ToTensor()(img)
        img = self.normalize(img)
        
        target = torch.from_numpy(target).unsqueeze(0) 
        
        return img, target


class ShanghaiTechDatasetSimple(Dataset):
    
    
    def __init__(
        self,
        data_root,
        gt_folder="ground-truth-h5-s8",
        training=True,
        crop_size=256,
        output_stride=8,
    ):
        self.data_root = data_root
        self.gt_folder = gt_folder
        self.training = training
        self.crop_size = crop_size
        self.output_stride = output_stride
        
        self.img_paths = get_image_paths(data_root)
        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No images found in {data_root}")
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        
        img = Image.open(img_path).convert("RGB")
        
        
        gt_path = img_path.replace("images", self.gt_folder).replace(".jpg", ".h5")
        with h5py.File(gt_path, "r") as hf:
            target = np.asarray(hf["density"], dtype=np.float32)
        
        if self.training:
        
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                target = np.fliplr(target).copy()
            
            
            w, h = img.size
            crop_h = min(self.crop_size, h)
            crop_w = min(self.crop_size, w)
            crop_h = (crop_h // self.output_stride) * self.output_stride
            crop_w = (crop_w // self.output_stride) * self.output_stride
            
            if crop_h >= self.output_stride and crop_w >= self.output_stride:
                x = random.randint(0, max(0, w - crop_w))
                y = random.randint(0, max(0, h - crop_h))
                x = (x // self.output_stride) * self.output_stride
                y = (y // self.output_stride) * self.output_stride
                
                img = img.crop((x, y, x + crop_w, y + crop_h))
                
                target_x = x // self.output_stride
                target_y = y // self.output_stride
                target_crop_w = crop_w // self.output_stride
                target_crop_h = crop_h // self.output_stride
                target = target[target_y:target_y + target_crop_h,
                               target_x:target_x + target_crop_w].copy()
        
        img = self.transform(img)
        target = torch.from_numpy(target).unsqueeze(0)
        
        return img, target


def collate_fn_variable_size(batch):
    
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    max_th = max(t.shape[1] for t in targets)
    max_tw = max(t.shape[2] for t in targets)
    
   
    padded_images = []
    for img in images:
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        padded = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_images.append(padded)
    
    
    padded_targets = []
    for target in targets:
        pad_h = max_th - target.shape[1]
        pad_w = max_tw - target.shape[2]
        padded = F.pad(target, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_targets.append(padded)
    
    return torch.stack(padded_images), torch.stack(padded_targets)


if __name__ == "__main__":
    
    train_root = "part_A/train_data"
    
    print("Testing ImprovedDataset...")
    try:
        ds = ShanghaiTechDatasetImproved(train_root, training=True)
        x, y = ds[0]
        print(f"Image shape: {x.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Target sum (count): {float(y.sum()):.2f}")
        
        
        for i in range(min(5, len(ds))):
            x, y = ds[i]
            print(f"Sample {i}: img={x.shape}, target={y.shape}, count={float(y.sum()):.2f}")
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
    
    print("\nDataset test complete!")