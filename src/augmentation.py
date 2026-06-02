import cv2
import numpy as np
from config import (
    AUGMENTATION_ENABLED, ROTATION_RANGE, BRIGHTNESS_RANGE,
    HORIZONTAL_FLIP, VERTICAL_FLIP
)

class ImageAugmentor:
    
    def __init__(self, image_size=224, augment=AUGMENTATION_ENABLED):
        self.image_size = image_size
        self.augment = augment
    
    def normalize(self, image):
        if image is None:
            return None
        
        image = image.astype(np.float32)
        
        if image.max() > 1.0:
            image = image / 255.0
        
        image = np.clip(image, 0, 1)
        
        return image
    
    def random_horizontal_flip(self, image):
        if not HORIZONTAL_FLIP:
            return image
        
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        return image
    
    def random_vertical_flip(self, image):
        if not VERTICAL_FLIP:
            return image
        
        if np.random.random() > 0.5:
            image = cv2.flip(image, 0)
        
        return image
    
    def random_rotation(self, image):
        angle = np.random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
        h, w = image.shape[:2]
        
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        
        if len(image.shape) == 2:
            image = cv2.warpAffine(image, M, (w, h), 
                                  borderMode=cv2.BORDER_REFLECT)
        else:
            image = cv2.warpAffine(image, M, (w, h),
                                  borderMode=cv2.BORDER_REFLECT)
        
        return image
    
    def random_brightness(self, image):
        brightness = np.random.uniform(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])
        image = image * brightness
        return np.clip(image, 0, 1)
    
    def random_contrast(self, image):
        contrast = np.random.uniform(0.8, 1.2)
        
        mean = image.mean()
        image = (image - mean) * contrast + mean
        
        return np.clip(image, 0, 1)
    
    def augment_image(self, image, augment=None):
        if image is None:
            return None
        
        image = self.normalize(image)
        
        do_augment = augment if augment is not None else self.augment
        
        if not do_augment:
            return image
        
        if HORIZONTAL_FLIP:
            image = self.random_horizontal_flip(image)
        
        if VERTICAL_FLIP:
            image = self.random_vertical_flip(image)
        
        image = self.random_rotation(image)
        
        if np.random.random() > 0.2:
            image = self.random_brightness(image)
        
        if np.random.random() > 0.4:
            image = self.random_contrast(image)
        
        return image
    
    def augment_image_aggressive(self, image):
        if image is None:
            return None
        
        image = self.normalize(image)
        
        angle = np.random.uniform(-25, 25)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        if len(image.shape) == 2:
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        else:
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        brightness = np.random.uniform(0.7, 1.3)
        image = image * brightness
        image = np.clip(image, 0, 1)
        
        contrast = np.random.uniform(0.7, 1.3)
        mean = image.mean()
        image = (image - mean) * contrast + mean
        image = np.clip(image, 0, 1)
        
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        if np.random.random() > 0.5:
            image = self._elastic_deform(image, alpha=30, sigma=3)
        
        return image
    
    def _elastic_deform(self, image, alpha=30, sigma=3):
        if len(image.shape) == 2:
            h, w = image.shape
            dx = np.random.normal(0, sigma, (h, w)) * alpha
            dy = np.random.normal(0, sigma, (h, w)) * alpha
            
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            x = np.clip(x + dx, 0, w - 1).astype(np.float32)
            y = np.clip(y + dy, 0, h - 1).astype(np.float32)
            
            image = cv2.remap(image, x, y, cv2.INTER_LINEAR)
        
        return image
    
    def batch_augment(self, images, augment=None):
        augmented = []
        for image in images:
            aug_image = self.augment_image(image, augment=augment)
            augmented.append(aug_image)
        
        return augmented


class TestTimeAugmentation:
    
    def __init__(self, image_size=224, num_augmentations=5):
        self.augmentor = ImageAugmentor(image_size=image_size, augment=True)
        self.num_augmentations = num_augmentations
    
    def apply_tta(self, image):
        augmented_images = []
        
        original = self.augmentor.normalize(image)
        augmented_images.append(original)
        
        for _ in range(self.num_augmentations - 1):
            aug_image = self.augmentor.augment_image(image, augment=True)
            augmented_images.append(aug_image)
        
        return augmented_images


if __name__ == "__main__":
    augmentor = ImageAugmentor(image_size=224, augment=True)
    dummy = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    aug = augmentor.augment_image(dummy)
    print("augmented", aug.shape, aug.min(), aug.max())
    tta = TestTimeAugmentation(num_augmentations=5)
    print("tta count", len(tta.apply_tta(dummy)))