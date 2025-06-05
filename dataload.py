import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
from typing import List, Optional, Tuple


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        image_size: int = 720,
        transform: Optional[transforms.Compose] = None,
        apply_clahe: bool = True,
        clahe_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.transform = transform
        self.apply_clahe = apply_clahe
        self.clahe_limit = clahe_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size

        # Validate inputs
        if len(image_paths) != len(mask_paths):
            raise ValueError("Number of images and masks must be equal")

        # Image normalization transform
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1] range
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image and mask
        # image = Image.open(self.image_paths[idx]).convert('L')  # Convert to grayscale
        # mask = Image.open(self.mask_paths[idx]).convert('L')
        # Use Open cv to read in gray scale image and mask
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # Apply CLAHE for contrast enhancement if enabled
        if self.apply_clahe:
            # image_np = np.array(image)
            clahe = cv2.createCLAHE(clipLimit=self.clahe_limit, tileGridSize=self.clahe_tile_grid_size)
            image = Image.fromarray(clahe.apply(image))

        # Resize both image and mask
        image = self._resize_to_square(np.array(image), self.image_size)
        mask = self._resize_to_square(np.array(mask), self.image_size)

        # Convert mask to binary (0 or 1)
        mask = (mask > 0).astype(np.float32)  # float32 for PyTorch compatibility

        # Convert to PIL for potential transforms
        # image_pil = Image.fromarray(image)
        # mask_pil = Image.fromarray(mask)

        # Apply additional transforms if specified

        augmented = train_transform(image=image, mask=mask)
            # image_pil = self.transform(image_pil)
            # mask_pil = self.transform(mask_pil)

        # Apply normalization to image only
        image_tensor = augmented['image']
        mask_tensor = augmented['mask'] # Add channel dimension

        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        return image_tensor, mask_tensor

    @staticmethod
    def _resize_to_square(image: np.ndarray, size: int) -> np.ndarray:
        """Resize image to square while maintaining aspect ratio with padding"""
        h, w = image.shape
        scale = size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(image, (new_w, new_h))

        # Pad to make square
        delta_w = size - new_w
        delta_h = size - new_h
        top = delta_h // 2
        bottom = delta_h - top
        left = delta_w // 2
        right = delta_w - left

        return cv2.copyMakeBorder(
            resized,
            top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=0
        )


class ClassificationDataset(Dataset):
    def __init__(self, catheter_predictions, atria_predictions,
                 labels, ids=None, original_images=None,
                 transform=None, normalize=False):
        """
        Modified dataset class with Albumentations support and proper normalization

        Args:
            catheter_predictions: Tensor/Numpy array of shape (N, H, W)
            atria_predictions: Tensor/Numpy array of shape (N, H, W)
            labels: List/Tensor of labels
            original_images: Optional original images (N, H, W) or (N, 3, H, W)
            transform: Albumentations transform pipeline
            normalize: Whether to normalize masks from 0-255 to 0-1
        """
        self.predictions = catheter_predictions
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.transform = transform
        self.atrial_mask = atria_predictions
        self.original_images = original_images
        self.ids = ids
        self.normalize = normalize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get raw data elements
        mask = self.predictions[idx]  # Shape (H, W)
        label = self.labels[idx]
        atria = self.atrial_mask[idx]

        # Convert to tensors if needed
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        if isinstance(atria, np.ndarray):
            atria = torch.from_numpy(atria)

        # Normalization for binary masks
        if self.normalize:
            mask = mask.float().div(255)
            atria = atria.float().div(255)
        else:
            mask = mask.float()
            atria = atria.float()

        # Handle third channel
        if self.original_images is not None:
            third_channel = self.original_images[idx]
            if isinstance(third_channel, np.ndarray):
                third_channel = torch.from_numpy(third_channel)
            third_channel = third_channel.float().div(255)
        else:
            third_channel = torch.zeros_like(mask)

        # Create 3-channel input (C, H, W)
        feature = torch.stack([mask, atria, third_channel], dim=0)

        # Convert to numpy array for Albumentations (H, W, C)
        feature_np = feature.permute(1, 2, 0).numpy()

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=feature_np)
            feature = transformed['image']  # Albumentations handles HWC -> CHW

        if self.ids is not None:
            return feature, label, self.ids[idx]
        return feature, label
