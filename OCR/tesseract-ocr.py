import cv2
import numpy as np
import pytesseract

img = cv2.imread("t_3.png")

# 1) Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2) Upscale (critical for small fonts)
scale = 3.0
gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

# 3) Illumination correction (background normalization)
bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=25)
norm = cv2.divide(gray, bg, scale=255)
norm = np.clip(norm, 0, 255).astype(np.uint8)

# 4) Local contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
contrast = clahe.apply(norm)

# 5) Edge-preserving denoise
denoised = cv2.fastNlMeansDenoising(
    contrast, None, h=12, templateWindowSize=7, searchWindowSize=21
)

# 6) Adaptive threshold (best choice here)
bw = cv2.adaptiveThreshold(
    denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 8
)

# 7) Light morphology cleanup
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

# OCR
config = (
    "--oem 3 --psm 6 "
    "-c tessedit_char_whitelist="
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789- "
)

text = pytesseract.image_to_string(bw, config=config)
print(text)
