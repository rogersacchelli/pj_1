import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
image = mpimg.imread('exit-ramp.png')
plt.imshow(image)

import cv2  #bringing in OpenCV libraries

gray = (cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8) #grayscale conversion
plt.imshow(gray, cmap='gray')

# Define a kernel size for Gaussian smoothing / blurring
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


low_threshold = 1.
high_threshold = 180.
edges = cv2.Canny(gray, low_threshold, high_threshold)

# Display the image
plt.imshow(edges, cmap='Greys_r')
plt.show()