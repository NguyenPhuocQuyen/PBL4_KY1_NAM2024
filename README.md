# test_camera
- Import thư viện open vc 
- Lấy camera từ máy ảnh 
- Dùng hai biến ret và frame để kiểm tra khung hình của mình ( với ret là kiểm tra đúng sai của màn hình và frame là lấy khung hình )
- Sau đó dùng cv2.imshow để hiển thị màn hình lên và lấy theo frame mình đã lấy ở lệnh cap.read
- cap.release để giải phóng thiết bị camera hoặc tệp đang được mở bởi đối tượng VideoCapture
- estroyAllWindows để đóng tất cả các cửa sổ hiển thị được tạo bởi OpenCV 
# Image Processing
- Thu thập dữ liệu 
- Tiền xử lý dữ liệu
- Xây dựng mô hình phân loại 
- Triển khai trên cánh tay robot 
#   ------------------------------------- 
##1 First step is: the imports

##2 Second step is: data collection

##3 third step is: data preparation

##4 fourth step is data normalization

##5 fifth step is data tranformation and augmentation

##6 sixth step is: defining the model

##7 seventh step is: compiling the created model, thus this model will become as a computaional graph

##8 eighth step: after doing all above steps, we can start the model training

##9 the last step is model evaluation

#MODULE 
ai.py: This is file load model 
train.py: This is file traing to create model
test_camera.py: This is file camera to display and known object 
