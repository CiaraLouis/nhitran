# Truong Dai hoc Cong nghe TP.HCM
#Tran Thi Yen Nhi - 2186300543 - 21DRTA1
#Kiem nhan khong hop le

import os

# Định nghĩa thư mục
images_dir = 'images/'  # Thư mục chứa ảnh
labels_dir = 'labels/'  # Thư mục chứa nhãn
# Lấy danh sách tên tệp (chỉ lấy tên, không có phần mở rộng)
image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))}
label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith('.txt')}

# Tìm các tệp nhãn không có tệp ảnh tương ứng
invalid_labels = label_files - image_files

# Xử lý và in ra thông báo
if invalid_labels:
    print("Các nhãn không có ảnh tương ứng:")
    for label in invalid_labels:
        print(f"- {label}.txt")
        # Xóa tệp nhãn
        os.remove(os.path.join(labels_dir, f"{label}.txt"))
    print("Đã xóa các nhãn không hợp lệ.")
else:
    print("Tất cả nhãn đều có ảnh tương ứng.")

#########Chú ý cho ngượi khác chạy lại từ đầu
########có 1 số file kiểu Tr-glTr, Tr-piTr, Tr-meTr do sai tên nên lúc split nó hơi lạ xíu, nhớ sửa đúng tên nhé


#######################################################################################################3

# Truong Dai hoc Cong nghe TP.HCM
#Tran Thi Yen Nhi - 2186300543 - 21DRTA1
#Splitdata

import os
import shutil
import random
import math

# Đường dẫn tới các thư mục gốc
images_dir = 'images/'  # Thư mục chứa ảnh
labels_dir = 'labels/'  # Thư mục chứa nhãn
output_dir = '.'  # Thư mục chứa các thư mục train, valid, test

# Tạo các thư mục đích
train_dir = os.path.join(output_dir, 'train')
valid_dir = os.path.join(output_dir, 'valid')
test_dir = os.path.join(output_dir, 'test')
os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(valid_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(valid_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'labels'), exist_ok=True)

# Hàm chia dữ liệu
def split_data_balanced(images_dir, labels_dir):
    # Liệt kê tất cả các file
    images = sorted(os.listdir(images_dir))
    labels = sorted(os.listdir(labels_dir))
    
    # Nhóm các file theo nhãn
    label_groups = {}
    for image, label in zip(images, labels):
        # Giả sử nhãn là phần đầu tiên trong tên file (vd: "label_123.jpg")
        label_name = label.split('_')[0]
        if label_name not in label_groups:
            label_groups[label_name] = []
        label_groups[label_name].append((image, label))
    
    # Tìm số lượng nhỏ nhất
    min_count = min(len(files) for files in label_groups.values())
    print(f"Số lượng nhỏ nhất trong các nhãn: {min_count}")

    # Tính số lượng cho từng tập
    train_count = math.floor(0.7 * min_count)
    valid_count = math.floor(0.2 * min_count)
    test_count = min_count - train_count - valid_count

    print(f"Chia tệp: train={train_count}, valid={valid_count}, test={test_count}")

    # Thống kê số lượng tệp mỗi nhãn trong từng tập
    stats = {'train': {}, 'valid': {}, 'test': {}}

    # Chia và di chuyển các tệp
    for label, files in label_groups.items():
        random.shuffle(files)  # Trộn ngẫu nhiên
        train_files = files[:train_count]
        valid_files = files[train_count:train_count + valid_count]
        test_files = files[train_count + valid_count:train_count + valid_count + test_count]

        # Di chuyển các tệp
        def move_files(file_list, target_dir, split_name):
            if label not in stats[split_name]:
                stats[split_name][label] = 0
            for image, label_file in file_list:
                shutil.copy(os.path.join(images_dir, image), os.path.join(target_dir, 'images', image))
                shutil.copy(os.path.join(labels_dir, label_file), os.path.join(target_dir, 'labels', label_file))
                stats[split_name][label] += 1

        move_files(train_files, train_dir, 'train')
        move_files(valid_files, valid_dir, 'valid')
        move_files(test_files, test_dir, 'test')

    # In thống kê
    for split_name, label_stats in stats.items():
        print(f"--- Số lượng tệp mỗi nhãn trong tập {split_name.upper()} ---")
        for label, count in label_stats.items():
            print(f"  Nhãn {label}: {count} tệp")
        print()

# Chia dữ liệu
split_data_balanced(images_dir, labels_dir)
print("Hoàn thành việc chia dữ liệu.")



