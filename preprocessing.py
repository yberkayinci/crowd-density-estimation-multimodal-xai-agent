import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
import scipy.spatial as spatial
from tqdm import tqdm


ROOT_PATH = 'part_A'
OUTPUT_STRIDE = 8
BETA = 0.3
K_NEIGHBORS = 4
MIN_SIGMA = 1.0
MAX_SIGMA = 40.0
FIXED_SIGMA = 15.0


def downsample_sum_pool(density, stride):
    if stride == 1:
        return density
    h, w = density.shape
    h2 = (h // stride) * stride
    w2 = (w // stride) * stride
    density = density[:h2, :w2]
    return density.reshape(h2 // stride, stride, w2 // stride, stride).sum(axis=(1, 3))


def _add_gaussian_kernel(density, x, y, sigma):
    """Stamp a truncated Gaussian kernel at (x, y) onto the density map in-place.

    Instead of creating a full-image array and convolving, we compute a small
    kernel patch (radius = 3*sigma) and add it directly. This is O(sigma^2)
    per point instead of O(H*W).
    """
    h, w = density.shape
    radius = int(np.ceil(sigma * 3))

    # Bounding box (clipped to image bounds)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)

    if y0 >= y1 or x0 >= x1:
        return 0.0

    # Coordinate grids relative to center
    gy = np.arange(y0, y1) - y
    gx = np.arange(x0, x1) - x
    gx, gy = np.meshgrid(gx, gy)

    kernel = np.exp(-(gx ** 2 + gy ** 2) / (2 * sigma ** 2))
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum  # Normalize so each point contributes count=1

    density[y0:y1, x0:x1] += kernel
    return kernel.sum()


def generate_density_map_improved(img_path, mat_path, stride=8, use_adaptive=True):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size

    try:
        mat_data = io.loadmat(mat_path)
        points = mat_data["image_info"][0, 0][0, 0][0]
    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        return None

    density = np.zeros((h, w), dtype=np.float64)
    num_gt = len(points)

    if num_gt == 0:
        return downsample_sum_pool(density.astype(np.float32), stride)

    pts = np.array(points, dtype=np.float32)

    # Pre-compute adaptive sigmas using KD-tree
    if use_adaptive and num_gt > 1:
        tree = spatial.KDTree(pts.copy(), leafsize=2048)
        k = min(K_NEIGHBORS, num_gt - 1)
        distances, _ = tree.query(pts, k=k + 1)

    for i, pt in enumerate(pts):
        x, y = int(round(pt[0])), int(round(pt[1]))
        if x < 0 or y < 0 or x >= w or y >= h:
            continue

        if use_adaptive and num_gt > 1:
            neighbor_dists = distances[i][1:k + 1]
            neighbor_dists = neighbor_dists[np.isfinite(neighbor_dists)]
            sigma = BETA * np.mean(neighbor_dists) if len(neighbor_dists) > 0 else FIXED_SIGMA
        else:
            sigma = FIXED_SIGMA

        sigma = float(np.clip(sigma, MIN_SIGMA, MAX_SIGMA))
        _add_gaussian_kernel(density, x, y, sigma)

    # Ensure total count is preserved
    current_sum = density.sum()
    if current_sum > 0:
        density *= (num_gt / current_sum)

    density = downsample_sum_pool(density.astype(np.float32), stride)

    # Re-normalize after downsampling to preserve exact count
    current_sum = float(density.sum())
    if current_sum > 0 and abs(current_sum - num_gt) > 0.01:
        density *= (num_gt / current_sum)

    return density


def generate_fixed_sigma_density(img_path, mat_path, stride=8, sigma=FIXED_SIGMA):
    return generate_density_map_improved(img_path, mat_path, stride=stride, use_adaptive=False)


def process_single_image(args):
    img_path, mat_path, out_path, stride, use_adaptive = args
    dmap = generate_density_map_improved(img_path, mat_path, stride=stride, use_adaptive=use_adaptive)
    if dmap is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with h5py.File(out_path, 'w') as hf:
            hf['density'] = dmap.astype(np.float32)
        return True
    return False


def process_folder(dataset_type, stride=8, use_adaptive=True):
    print(f"Processing {dataset_type} (stride={stride}, adaptive={use_adaptive})...")

    img_dir = os.path.join(ROOT_PATH, dataset_type, 'images')
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))

    if not img_paths:
        print(f"No images found in {img_dir}.")
        return

    out_dir = f'ground-truth-h5-s{stride}'

    args_list = []
    for img_path in img_paths:
        mat_path = (img_path
                    .replace('images', 'ground-truth')
                    .replace('IMG_', 'GT_IMG_')
                    .replace('.jpg', '.mat'))
        out_path = img_path.replace('images', out_dir).replace('.jpg', '.h5')
        args_list.append((img_path, mat_path, out_path, stride, use_adaptive))

    success_count = 0
    for args in tqdm(args_list, desc=f"Generating {dataset_type}"):
        if process_single_image(args):
            success_count += 1

    print(f"Processed {success_count}/{len(img_paths)} images successfully.")


def verify_density_maps(dataset_type, stride=8):
    print(f"\nVerifying {dataset_type} density maps...")

    img_dir = os.path.join(ROOT_PATH, dataset_type, 'images')
    gt_dir = os.path.join(ROOT_PATH, dataset_type, f'ground-truth-h5-s{stride}')
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))

    errors = []
    for img_path in tqdm(img_paths[:20], desc="Verifying"):
        basename = os.path.basename(img_path).replace('.jpg', '.h5')
        h5_path = os.path.join(gt_dir, basename)

        mat_path = (img_path
                    .replace('images', 'ground-truth')
                    .replace('IMG_', 'GT_IMG_')
                    .replace('.jpg', '.mat'))
        try:
            mat_data = io.loadmat(mat_path)
            gt_count = len(mat_data["image_info"][0, 0][0, 0][0])
        except Exception:
            continue

        if os.path.exists(h5_path):
            with h5py.File(h5_path, 'r') as hf:
                density_count = float(hf['density'][:].sum())
            error = abs(gt_count - density_count)
            errors.append(error)
            if error > 0.1:
                print(f"  {basename}: GT={gt_count}, Density={density_count:.2f}, Error={error:.2f}")

    if errors:
        print(f"Average count error: {np.mean(errors):.4f}")
        print(f"Max count error: {np.max(errors):.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate density maps for CSRNet')
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--adaptive', action='store_true', default=True)
    parser.add_argument('--fixed', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()

    use_adaptive = not args.fixed

    process_folder('train_data', stride=args.stride, use_adaptive=use_adaptive)
    process_folder('test_data', stride=args.stride, use_adaptive=use_adaptive)

    if args.verify:
        verify_density_maps('train_data', stride=args.stride)
        verify_density_maps('test_data', stride=args.stride)

    print("\nDensity map generation complete!")
