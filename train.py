#import libraries
import random
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
import os 
import PIL 

from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense


# ----------- Chia tập dữ liệu thành tập train và tập test ---------------- 
# Đường dẫn đến thư mục chứa gốc dữ liệu hình ảnh
folder_main = r"D:\Learning\Ky1_Nam4\PBL4\Trash_Classification\archive\Garbage_classification\Garbage_classification"
# Đường dẫn đến thư mục chứa tập train và tập test 
train_test = r"D:\Learning\Ky1_Nam4\PBL4\Trash_Classification\archive_split"
# Chọn tỉ lệ phần trăm cho tập train 
train_percentage = 90

#Tạo thư mục train và thư mục test 
train_folder = os.path.join(train_test, "train")
test_folder  = os.path.join(train_test, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
# print(train_folder)
# print(test_folder)

#Lặp qua các thư mục con trong thư mục gốc 
for folder_name in os.listdir(folder_main):
    # print(folder_name) # In ra các thư mục con 
    folder_dir = os.path.join(folder_main, folder_name)
    # print(folder_dir) # In ra link của thư mục main + folder_name
    # Kiểm tra đường dẫn có phải thư mục hay không 
    if os.path.isdir(folder_dir):
        # Tạo thư mục con trong thư mục train và test 
        train_class_dir = os.path.join(train_folder, folder_name)
        test_class_dir = os.path.join(test_folder, folder_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        #Lấy danh sách các tệp tin có trong thư mục của lỗi lớp 
        files = os.listdir(folder_dir)
        # print(files) # Xuất từng ảnh 
        # Xáo trộn danh sách các tệp tin 
        random.shuffle(files)
        # print(len(files))
        # Tính chỉ số cắt để chia thành tập train và tập test 
        split_index = int(len(files)*train_percentage/100)
        # print(split_index)
        #Chia các tệp tin thành tập train và test 
        train_files = files[:split_index]
        # print("Số file tập train")
        # print(len(train_files))
        test_files = files[split_index:]
        # print(len(train_files))
        # print("Số file tập test")
        # print(len(test_files))
        #Di chuyển các tệp train vào thư mục train 
        for file in train_files:
            file_path = os.path.join(folder_dir, file)
        # print(len(file_path))
            shutil.copy(file_path, train_class_dir)
        #Di chuyển các tệp train vào thư mục test
        for file in test_files:
            file_path = os.path.join(folder_dir, file)
        # print(len(file_path))
            shutil.copy(file_path, test_class_dir)
# ------------------- TIỀN XỬ LÝ ------------------------------- 
# Số lần lặp tập dữ liệu để huấn luyện 
NUMBER_OF_PORCH = 30
# Chuẩn hoá lại ảnh từ 0 đến 1  
RESCALING = False
# Size ảnh 
IMAGE_SIZE = 256
# Chế độ màu 
COLOR_MODE = "rbg"
# COLOR_MODE = "grayscale"
# Lựa kênh theo chế độ màu 
if COLOR_MODE == "rbg":
    CHANNELS = 3
else: 
    CHANNELS = 1

#Thiết lập train 
train_ds = tf.keras.utils.image_dataset_from_directory(
# Đưa đường dẫn tập train
    directory = r"D:\Learning\Ky1_Nam4\PBL4\Trash_Classification\archive_split\train",
# Gắn nhãn tương ứng với các thư mục con: cardboard = 0, glass = 1 ..... 
    labels = "inferred",
    label_mode = "int",
#Tên của các lớp (các thư mục cardboard, glass .... ), None vì lấy từ thư mục con 
    class_names = None,
# Số ảnh được đưa vào huấn luyện mỗii lần là 32 ảnh 
    # batch_size=32,
    batch_size=16,
#Tham số hình ảnh đầu vào
    image_size = (IMAGE_SIZE, IMAGE_SIZE),
#Xáo trộn dữ liệu trước khi lấy 32 ảnh 
    shuffle = True, 
    seed = None, 
#Chia tập dữ liệu val
    validation_split=None,
    subset = None,
#Phương pháp nội suy sử dụng khi thay đổi kích thước ảnh
    interpolation= "bilinear",
    follow_links= False, 
    crop_to_aspect_ratio= False, 
)
# print(train_ds)
# print(type(train_ds))

#---------CHUẨN HOÁ  ----------- 
rescaling_layer = layers.Rescaling(1./255)
train_ds_rescaled = train_ds.map(lambda x,y: (rescaling_layer(x), y))
train_ds_rescaled = train_ds_rescaled.prefetch(tf.data.AUTOTUNE)

class_names = train_ds.class_names
print(class_names)

# --------- XÂY DỰNG MÔ HÌNH ------------------------------

model = tf.keras.models.Sequential([
#Thêm convolution và maxpooling 
    tf.keras.layers.Conv2D(64 ,(3,3), activation = "relu", input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128 ,(3,3), activation = "relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64 ,(3,3), activation = "relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    # tf.keras.layers.Conv2D(64 ,(3,3), activation = "relu", input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
    # tf.keras.layers.MaxPooling2D(2,2),
    # tf.keras.layers.Conv2D(32 ,(3,3), activation = "relu"),
    # tf.keras.layers.MaxPooling2D(2,2),

#Biến đổi thành vector một chiều 
    tf.keras.layers.Flatten(),
#Lớp Fully Connected
    tf.keras.layers.Dense(128, activation= "relu"),
    tf.keras.layers.Dense(64, activation= "relu"),
    tf.keras.layers.Dense(6, activation= "softmax"),
])
#Print model
model.summary()
#Complie models 
model.compile(optimizer="adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
#Huấn luyện mô hình 
if RESCALING == True:
    model.fit(train_ds_rescaled, epochs = NUMBER_OF_PORCH)
#Đánh giá mô hình 
    print("\nModel evaluation: ")
    test_loss = model.evaluate(train_ds_rescaled)
else:
    model.fit(train_ds, epochs = NUMBER_OF_PORCH)
#Đánh giá mô hình 
    print("\nModel evaluation: ")
    test_loss = model.evaluate(train_ds)

#Lưu models 
tf.keras.models.save_model(
    model, 
    filepath = "./ai_models/model_trashclassification3_bs_16.h5",
    overwrite = True,
    include_optimizer = False,
    save_format = None, # Đuôi h5
    signatures = None,
    options = None, 
    save_traces = True,
)