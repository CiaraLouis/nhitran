import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np

# Tải mô hình đã huấn luyện
model = YOLO(r'C:\Users\HP\Desktop\train\runs\segment\train11\weights\best.pt')

# Hàm xử lý khi nhấn nút nhập ảnh
def load_image():
    # Chọn file ảnh
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
    if not file_path:
        return  # Thoát nếu không chọn ảnh

    # Hiển thị ảnh gốc lên giao diện
    original_img = Image.open(file_path)
    original_img_tk = ImageTk.PhotoImage(original_img)  # Giữ nguyên kích thước ảnh
    original_label.configure(image=original_img_tk)
    original_label.image = original_img_tk

    # Thực hiện phân đoạn ảnh bằng mô hình
    results = model.predict(file_path)
    result = results[0]  # Lấy kết quả đầu tiên

    # Vẽ ảnh kết quả
    if result.masks is not None:
        result_img = result.plot()  # Vẽ ảnh kết quả phân đoạn
        result_img = Image.fromarray(result_img)  # Chuyển sang định dạng PIL Image
        result_img_tk = ImageTk.PhotoImage(result_img)
        result_label.configure(image=result_img_tk)
        result_label.image = result_img_tk

        # Lấy thông tin loại và kích thước mask
        total_mask_pixels = 0
        tumor_type = ""
        for idx, mask in enumerate(result.masks.data):
            total_mask_pixels += np.sum(mask.cpu().numpy())  # Tổng số pixel trong mask

            # Kiểm tra loại khối u (ví dụ: GL, ME, PI)
            # Giả sử bạn đã huấn luyện mô hình với 3 loại là GL (0), ME (1), PI (2)
            if result.boxes.cls[idx] == 0:
                tumor_type = "GL"
            elif result.boxes.cls[idx] == 1:
                tumor_type = "ME"
            elif result.boxes.cls[idx] == 2:
                tumor_type = "PI"

        # Hiển thị loại và kích thước khối u
        type_entry.delete(0, tk.END)
        type_entry.insert(0, f"Loại: {tumor_type}")

        size_entry.delete(0, tk.END)
        size_entry.insert(0, f"Kích thước: {total_mask_pixels} px")
    else:
        type_entry.delete(0, tk.END)
        type_entry.insert(0, "Không phát hiện")
        size_entry.delete(0, tk.END)
        size_entry.insert(0, "NaN")

# Tạo giao diện Tkinter
window = tk.Tk()
window.title("Phân đoạn khối u não")
window.geometry("800x600")  # Kích thước cửa sổ lớn hơn

# Khung hiển thị ảnh gốc (kích thước tự động dựa trên ảnh)
original_label = tk.Label(window, text="Ảnh gốc", bg="gray")
original_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# Khung hiển thị ảnh kết quả (kích thước tự động dựa trên ảnh)
result_label = tk.Label(window, text="Ảnh kết quả", bg="gray")
result_label.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

# Nút nhập ảnh
load_button = tk.Button(window, text="Nhập ảnh", command=load_image, width=20)
load_button.grid(row=1, column=0, columnspan=2, pady=10)

# Ô nhập loại khối u
type_label = tk.Label(window, text="Loại khối u:")
type_label.grid(row=2, column=0, sticky="e")
type_entry = tk.Entry(window, width=30)
type_entry.grid(row=2, column=1, sticky="w")

# Ô nhập kích thước pixel
size_label = tk.Label(window, text="Kích thước (px):")
size_label.grid(row=3, column=0, sticky="e")
size_entry = tk.Entry(window, width=30)
size_entry.grid(row=3, column=1, sticky="w")

# Cài đặt chế độ cho giao diện
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
window.grid_columnconfigure(1, weight=1)

# Chạy giao diện
window.mainloop()
