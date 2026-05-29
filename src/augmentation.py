"""
Image augmentation utilities for chest X-ray images.
Provides augmentation strategies for training set.
"""

import cv2
import numpy as np
from config import (
    AUGMENTATION_ENABLED, ROTATION_RANGE, BRIGHTNESS_RANGE,
    HORIZONTAL_FLIP, VERTICAL_FLIP
)

class ImageAugmentor:
    """Apply augmentations to chest X-ray images."""
    
    def __init__(self, image_size=224, augment=AUGMENTATION_ENABLED):
        self.image_size = image_size
        self.augment = augment
    
    def normalize(self, image):
        """Normalize image to [0, 1] range."""
        if image is None:
            return None
        
        image = image.astype(np.float32)
        
        # If max > 1, assume it's in [0, 255]
        if image.max() > 1.0:
            image = image / 255.0
        
        # Clip to valid range
        image = np.clip(image, 0, 1)
        
        return image
    
    def random_horizontal_flip(self, image):
        """Randomly flip image horizontally."""
        if not HORIZONTAL_FLIP:
            return image
        
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        return image
    
    def random_vertical_flip(self, image):
        """Randomly flip image vertically."""
        if not VERTICAL_FLIP:
            return image
        
        if np.random.random() > 0.5:
            image = cv2.flip(image, 0)
        
        return image
    
    def random_rotation(self, image):
        """Randomly rotate image."""
        angle = np.random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
        h, w = image.shape[:2]
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        
        # Apply rotation
        if len(image.shape) == 2:  # Grayscale
            image = cv2.warpAffine(image, M, (w, h), 
                                  borderMode=cv2.BORDER_REFLECT)
        else:  # Color
            image = cv2.warpAffine(image, M, (w, h),
                                  borderMode=cv2.BORDER_REFLECT)
        
        return image
    
    def random_brightness(self, image):
        """Randomly adjust brightness."""
        brightness = np.random.uniform(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])
        image = image * brightness
        return np.clip(image, 0, 1)
    
    def random_contrast(self, image):
        """Randomly adjust contrast."""
        contrast = np.random.uniform(0.8, 1.2)
        
        # Adjust contrast around mean
        mean = image.mean()
        image = (image - mean) * contrast + mean
        
        return np.clip(image, 0, 1)
    
    def augment_image(self, image, augment=None):
        """
        Apply augmentations to a single image.
        
        Args:
            image: Input image (H x W for grayscale or H x W x C)
            augment: Override augmentation setting (if None, use self.augment)
        
        Returns:
            Augmented image (normalized to [0, 1])
        """
        if image is None:
            return None
        
        # Normalize first
        image = self.normalize(image)
        
        # Determine if we should augment
        do_augment = augment if augment is not None else self.augment
        
        if not do_augment:
            return image
        
        # Apply augmentations with some probability
        # Horizontal flip: 50% chance
        if HORIZONTAL_FLIP:
            image = self.random_horizontal_flip(image)
        
        # Vertical flip: 20% chance (less likely for chest X-rays)
        if VERTICAL_FLIP:
            image = self.random_vertical_flip(image)
        
        # Rotation: always applied (angle can be 0)
        image = self.random_rotation(image)
        
        # Brightness: 80% chance
        if np.random.random() > 0.2:
            image = self.random_brightness(image)
        
        # Contrast: 60% chance
        if np.random.random() > 0.4:
            image = self.random_contrast(image)
        
        return image
    
    def batch_augment(self, images, augment=None):
        """
        Apply augmentations to a batch of images.
        
        Args:
            images: List or array of images
            augment: Override augmentation setting
        
        Returns:
            List of augmented images
        """
        augmented = []
        for image in images:
            aug_image = self.augment_image(image, augment=augment)
            augmented.append(aug_image)
        
        return augmented


class TestTimeAugmentation:
    """Test-time augmentation (TTA) for improved predictions."""
    
    def __init__(self, image_size=224, num_augmentations=5):
        self.augmentor = ImageAugmentor(image_size=image_size, augment=True)
        self.num_augmentations = num_augmentations
    
    def apply_tta(self, image):
        """
        Apply test-time augmentation.
        Returns multiple augmented versions of the same image.
        
        Args:
            image: Single image
        
        Returns:
            List of augmented images
        """
        augmented_images = []
        
        # Add original (non-augmented) version
        original = self.augmentor.normalize(image)
        augmented_images.append(original)
        
        # Add augmented versions
        for _ in range(self.num_augmentations - 1):
            aug_image = self.augmentor.augment_image(image, augment=True)
            augmented_images.append(aug_image)
        
        return augmented_images


if __name__ == "__main__":
    # Example usage
    print("Image Augmentation Module")
    print("=" * 50)
    
    # Create augmentor
    augmentor = ImageAugmentor(image_size=224, augment=True)
    print("✅ ImageAugmentor created")
    
    # Create dummy image
    dummy_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    print("✅ Dummy image created: shape =", dummy_image.shape)
    
    # Test augmentation
    augmented = augmentor.augment_image(dummy_image)
    print("✅ Augmentation applied: shape =", augmented.shape)
    print("   Min value: {:.3f}, Max value: {:.3f}".format(augmented.min(), augmented.max()))
    
    # Test TTA
    print("\nTesting Test-Time Augmentation...")
    tta = TestTimeAugmentation(num_augmentations=5)
    tta_images = tta.apply_tta(dummy_image)
    print(f"✅ TTA generated {len(tta_images)} augmented versions")