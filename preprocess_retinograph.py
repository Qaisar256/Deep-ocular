import cv2
import numpy as np
from skimage import exposure

def preprocess_retinograph(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(gray_image)
    
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)
    
    # Apply contrast stretching
    p2, p98 = np.percentile(blurred_image, (2, 98))
    stretched_image = exposure.rescale_intensity(blurred_image, in_range=(p2, p98))
    
    return stretched_image

# Load your retinograph image using OpenCV
image_path = "path_to_your_image.jpg"
original_image = cv2.imread(image_path)

# Preprocess the image
preprocessed_image = preprocess_retinograph(original_image)
enhanced_image = BenGraham_enhance(preprocessed_image )
# Display the original and preprocessed images (you can use a GUI library or matplotlib)
cv2.imshow("Original Image", original_image)
cv2.imshow("Preprocessed Image", preprocessed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
