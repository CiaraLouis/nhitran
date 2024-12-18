# Truong Dai hoc Cong nghe TP.HCM
#Tran Thi Yen Nhi - 2186300543 - 21DRTA1
#Mask to Polygonn

import os
import cv2

#Duong dan input va output
input_dir = r'C:\Users\HP\Desktop\Brain Tumor Segmentation\mask_pre\1'
output_dir = r'C:\Users\HP\Desktop\Brain Tumor Segmentation\labels'

for j in os.listdir(input_dir):
    image_path = os.path.join(input_dir, j)
    
    # Load the binary mask and get its contours
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Không thể đọc ảnh: {image_path}")
        continue
    
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    H, W = mask.shape
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the contours to polygons
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 0:
            polygon = []
            for point in cnt:
                x, y = point[0]
                polygon.append(x / W)
                polygon.append(y / H)
            polygons.append(polygon)

    # Generate output filename
    output_name = os.path.splitext(j)[0].rsplit('_', 1)[0]  # Lấy phần tên trước "_m"
    output_path = os.path.join(output_dir, f"{output_name}.txt")

    # Save polygons to a text file
    with open(output_path, 'w') as f:
        for polygon in polygons:
            f.write('0 ')  # Ghi nhãn, ví dụ: 3
            f.write(' '.join(map(str, polygon)))  # Ghi toàn bộ tọa độ polygon
            f.write('\n')
