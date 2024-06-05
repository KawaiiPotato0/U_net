import numpy as np
import albumentations as A
from skimage.transform import resize
from utils import normalize_image
from pathlib import Path

def prepare(source, target, norm=False, aug=False, gray=False):
    # Load original data
    images = np.load(f"{source}/images.npy", allow_pickle=True)
    masks = np.load(f"{source}/masks.npy", allow_pickle=True)
    labels = np.load(f"{source}/labels.npy", allow_pickle=True)

    target_shape = (128, 128)

    images = np.array([img if img.shape == target_shape else resize(img, target_shape) for img in images], dtype = np.float32)
    masks = np.array([img if img.shape == target_shape else resize(img, target_shape) for img in masks], dtype = np.float32)

    if norm:
        images = np.array([normalize_image(img) for img in images], dtype = np.float32)
        masks = np.array([normalize_image(img) for img in masks], dtype = np.float32) 

    if aug:
        # Define augmentation transformations
        transform = A.Compose([
            A.RandomRotate90(p=1),    # Apply RandomRotate90
            A.Flip(p=1)  # Apply Flip
        ])

        # Augment images and masks
        augmented_images = []
        augmented_masks = []

        for img, mask in zip(images, masks):
            augmented = transform(image=img, mask=mask)
            augmented_img = augmented['image']
            augmented_mask = augmented['mask']
            augmented_images.append(augmented_img)
            augmented_masks.append(augmented_mask)

        # Convert lists to numpy arrays
        augmented_images = np.array(augmented_images, dtype=np.float32)
        augmented_masks = np.array(augmented_masks, dtype=np.float32)

        # Replicate labels for augmented data
        num_augmented_samples = len(augmented_images)
        augmented_labels = np.repeat(labels, num_augmented_samples // len(labels))

        # Concatenate original and augmented data
        images = np.concatenate((images, augmented_images), axis=0)
        masks = np.concatenate((masks, augmented_masks), axis=0)
        labels = np.concatenate((labels, augmented_labels), axis=0)

    if gray:
        # Define augmentation transformations
        transform = A.Compose([
            A.RandomBrightnessContrast(p=1)  # Apply RandomBrightnessContrast
        ])

        # Augment images and masks
        augmented_images = []
        augmented_masks = []

        for img, mask in zip(images, masks):
            augmented = transform(image=img, mask=mask)
            augmented_img = augmented['image']
            augmented_mask = augmented['mask']
            augmented_images.append(augmented_img)
            augmented_masks.append(augmented_mask)

        # Convert lists to numpy arrays
        augmented_images = np.array(augmented_images, dtype=np.float32)
        augmented_masks = np.array(augmented_masks, dtype=np.float32)

        # Replicate labels for augmented data
        num_augmented_samples = len(augmented_images)
        augmented_labels = np.repeat(labels, num_augmented_samples // len(labels))

        # Concatenate original and augmented data
        images = np.concatenate((images, augmented_images), axis=0)
        masks = np.concatenate((masks, augmented_masks), axis=0)
        labels = np.concatenate((labels, augmented_labels), axis=0)

    Path(target).mkdir(parents=True, exist_ok=True)

    with open(f"{target}/images.npy", 'wb+') as f:
        np.save(f, images)
    with open(f"{target}/masks.npy", 'wb+') as f:
        np.save(f, masks)
    with open(f"{target}/labels.npy", 'wb+') as f:
        np.save(f, labels)

prepare("brain_tumor_dataset/original", "brain_tumor_dataset/norm_aug_gray", True, True, True)
