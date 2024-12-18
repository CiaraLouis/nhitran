# Truong Dai hoc Cong nghe TP.HCM
#Tran Thi Yen Nhi - 2186300543 - 21DRTA1
#Tien xu ly du lieu

import cv2
import numpy as np
import os
from scipy.signal import convolve2d

# Tao bo loc Gaussian
def gauss_filter(kernel_size, sigma, mu=0):
    x, y = np.meshgrid(np.linspace(-kernel_size//2, kernel_size//2, kernel_size),
                       np.linspace(-kernel_size//2, kernel_size//2, kernel_size))
    dst = np.sqrt(x**2 + y**2)
    normal = 1 / (2.0 * np.pi * sigma**2)
    gauss = np.exp(-((dst - mu)**2 / (2.0 * sigma**2))) * normal
    return gauss

# Chuyen BGR sang Gray
def BGRtoGRAY(img):
    R, G, B = img[:,:,2], img[:,:,1], img[:,:,0]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    return Y.astype(np.uint8)

# Ham xu ly anh
def process_image(img_path, output_path, kernel_size=5, sigma=1, resize_dim=(640, 640)):

    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        return

    gray_img = BGRtoGRAY(img)

    gaussian_kernel = gauss_filter(kernel_size, sigma)
    filter_img = convolve2d(gray_img, gaussian_kernel, mode='same', boundary='symm')
    
    # chuan hoa [0, 255]
    norm_img = np.clip(filter_img, 0, 255)
    norm_img = norm_img.astype(np.uint8)
    
    # Resize image về 640x640
    resized_img = cv2.resize(norm_img, resize_dim, interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(output_path, resized_img)

# Duong dan thu muc input va output
input_dir = r'C:\Users\HP\Desktop\Brain Tumor Segmentation\data\mask\3'
output_dir = r'C:\Users\HP\Desktop\Brain Tumor Segmentation\mask_pre\3'

# kiem tra thu muc output
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Ap dung xu ly tung anh
for filename in os.listdir(input_dir):
    img_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, f'{filename}')
    
    if os.path.isfile(img_path):
        process_image(img_path, output_path)
        print(f'Processed {filename} and saved to {output_path}')
