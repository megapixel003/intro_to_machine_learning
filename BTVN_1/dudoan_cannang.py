import numpy as np
from sklearn.linear_model import LinearRegression

# Dữ liệu từ bảng
chieu_cao = np.array([147, 150, 153, 155, 158, 160, 163, 165, 168, 170, 173, 175, 178, 180, 183]).reshape(-1, 1)
can_nang = np.array([49, 50, 51, 52, 54, 56, 58, 59, 60, 72, 63, 64, 66, 67, 68])

# Khởi tạo mô hình hồi quy tuyến tính
model = LinearRegression()

# Huấn luyện mô hình với dữ liệu
model.fit(chieu_cao, can_nang)

# Nhập chiều cao mới từ bàn phím
newHeight = float(input("Nhập chiều cao mới (cm): "))

# Dự đoán cân nặng dựa trên chiều cao mới
predictedWeight = model.predict(np.array([[newHeight]]))

# In ra kết quả dự đoán
print(f"Cân nặng dự đoán :185 {predictedWeight[0]:.2f} kg.")
