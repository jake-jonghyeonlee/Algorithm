import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to image.
    
    Args:
        image: Input image
        mean: Mean of the Gaussian noise
        sigma: Standard deviation of the Gaussian noise
    """
    if len(image.shape) == 3:  # Color image
        noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    else:  # Grayscale image
        noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    
    noisy_image = cv2.add(image, noise)
    return noisy_image

def elastic_transform(image, alpha=2000, sigma=50, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    
    Args:
        image: Image to be deformed
        alpha: Scaling factor that controls intensity of deformation
        sigma: Smoothing factor
        random_state: Random state for reproducibility
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    if len(shape) == 3:  # Color image
        result = np.zeros_like(image)
        for i in range(shape[2]):
            result[:,:,i] = map_coordinates(image[:,:,i], indices, order=1).reshape(shape[:2])
    else:  # Grayscale image
        result = map_coordinates(image, indices, order=1).reshape(shape)
    
    return result

def random_scale(image, scale_range=(0.8, 1.2)):
    """Random scaling of the image.
    
    Args:
        image: Image to be scaled
        scale_range: Tuple of (min_scale, max_scale)
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    
    height, width = image.shape[:2]
    new_height = int(height * scale)
    new_width = int(width * scale)
    
    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Pad or crop to original size if necessary
    if scale > 1:  # Image is larger, need to crop
        start_h = (new_height - height) // 2
        start_w = (new_width - width) // 2
        scaled_image = scaled_image[start_h:start_h+height, start_w:start_w+width]
    else:  # Image is smaller, need to pad
        pad_h = (height - new_height) // 2
        pad_w = (width - new_width) // 2
        if len(image.shape) == 3:  # Color image
            scaled_image = np.pad(scaled_image, ((pad_h, height-new_height-pad_h),
                                               (pad_w, width-new_width-pad_w),
                                               (0, 0)), mode='reflect')
        else:  # Grayscale image
            scaled_image = np.pad(scaled_image, ((pad_h, height-new_height-pad_h),
                                               (pad_w, width-new_width-pad_w)), mode='reflect')
    
    return scaled_image

def apply_augmentation(image, mask=None):
    """Apply augmentations to image and mask.
    
    Args:
        image: Input image
        mask: Input mask (optional)
    """
    # Apply elastic transform
    transformed_image = elastic_transform(image)
    if mask is not None:
        transformed_mask = elastic_transform(mask)
    
    # Apply random scaling
    scaled_image = random_scale(transformed_image)
    if mask is not None:
        scaled_mask = random_scale(transformed_mask)
    
    # Apply Gaussian noise (only to image, not to mask)
    noisy_image = add_gaussian_noise(scaled_image)
    
    if mask is not None:
        return noisy_image, scaled_mask
    
    return noisy_image

if __name__ == "__main__":
    # Load image and mask
    image = cv2.imread('path_to_your_image.jpg')
    mask = cv2.imread('path_to_your_mask.jpg', 0)  # Read mask as grayscale
    
    # Apply augmentation
    augmented_image, augmented_mask = apply_augmentation(image, mask)
    
    # Display results
    cv2.imshow('Original Image', image)
    cv2.imshow('Augmented Image', augmented_image)
    cv2.imshow('Original Mask', mask)
    cv2.imshow('Augmented Mask', augmented_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()