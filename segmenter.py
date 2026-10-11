import cv2
import numpy as np

class GarmentSegmenter:
    """Handles background removal and garment segmentation for e-commerce images."""
    
    def __init__(self):
        pass

    def segment(self, img):
        """
        Remove background using automatic threshold detection.
        img: numpy array (BGR)
        Returns a 4-channel numpy array (BGRA) with transparent background.
        """
        if img is None:
            return None
        
        height, width = img.shape[:2]
        
        # Convert to different color spaces for better background detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Detect white/light background (common in e-commerce)
        white_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY_INV, 15, 10)
        
        # Method 2: Use color-based segmentation from corners
        corner_samples = [
            img[0:20, 0:20],  # Top-left
            img[0:20, width-20:width],  # Top-right
            img[height-20:height, 0:20],  # Bottom-left
            img[height-20:height, width-20:width]  # Bottom-right
        ]
        
        # Calculate average background color from corners
        bg_color = np.mean([np.mean(sample.reshape(-1, 3), axis=0) for sample in corner_samples], axis=0)
        
        # Create mask based on color similarity to background
        color_diff = np.sqrt(np.sum((img - bg_color) ** 2, axis=2))
        color_threshold = np.mean(color_diff) + np.std(color_diff) * 0.5
        color_mask = (color_diff > color_threshold).astype(np.uint8) * 255
        
        # Combine masks
        combined_mask = cv2.bitwise_or(white_mask, color_mask)
        
        # Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Remove small noise and keep only the largest contour (main garment)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Keep only the largest contour (main garment)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create clean mask from largest contour
            final_mask = np.zeros_like(combined_mask)
            cv2.fillPoly(final_mask, [largest_contour], 255)
            
            # Smooth the edges
            final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)
        else:
            final_mask = combined_mask
        
        # Create alpha channel
        alpha = final_mask
        
        # Convert to 4-channel image
        b, g, r = cv2.split(img)
        img_with_alpha = cv2.merge([b, g, r, alpha])
        
        return img_with_alpha
