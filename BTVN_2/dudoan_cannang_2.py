import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def du_doan_can_nang(model):
    # Nhập chiều cao mới từ bàn phím
    new_height = float(input("Nhập chiều cao mới (cm): "))
    # Dự đoán cân nặng dựa trên chiều cao mới
    predicted_weight = model.predict(np.array([[new_height]]))
    # In ra kết quả dự đoán
    print(f"Cân nặng dự đoán cho người có chiều cao {new_height} cm là {predicted_weight[0]:.2f} kg.")
    return new_height, predicted_weight[0]

def nhap_du_lieu_moi():
    # Nhập số lượng mẫu
    so_mau = int(input("Nhập số lượng mẫu dữ liệu: (tối thiểu 15 mẫu) "))
    while so_mau < 15:
        print("Bạn phải nhập tối thiểu 15 mẫu dữ liệu!")
        so_mau = int(input("Nhập số lượng mẫu dữ liệu: (tối thiểu 15 mẫu) "))
    
    # Nhập chiều cao tất cả mẫu cùng lúc
    chieu_cao_str = input(f"Nhập chiều cao của {so_mau} mẫu (cách nhau bằng dấu cách): ")
    chieu_cao_moi = list(map(float, chieu_cao_str.split()))
    
    # Kiểm tra số lượng mẫu có khớp không
    while len(chieu_cao_moi) != so_mau:
        print(f"Bạn phải nhập đủ {so_mau} mẫu chiều cao!")
        chieu_cao_str = input(f"Nhập chiều cao của {so_mau} mẫu (cách nhau bằng dấu cách): ")
        chieu_cao_moi = list(map(float, chieu_cao_str.split()))
    
    # Nhập cân nặng tất cả mẫu cùng lúc
    can_nang_str = input(f"Nhập cân nặng của {so_mau} mẫu (cách nhau bằng dấu cách): ")
    can_nang_moi = list(map(float, can_nang_str.split()))
    
    # Kiểm tra số lượng mẫu cân nặng có khớp không
    while len(can_nang_moi) != so_mau:
        print(f"Bạn phải nhập đủ {so_mau} mẫu cân nặng!")
        can_nang_str = input(f"Nhập cân nặng của {so_mau} mẫu (cách nhau bằng dấu cách): ")
        can_nang_moi = list(map(float, can_nang_str.split()))
    
    return np.array(chieu_cao_moi).reshape(-1, 1), np.array(can_nang_moi)

def ve_bieu_do(chieu_cao, can_nang, model):
    # Dự đoán cân nặng dựa trên chiều cao
    chieu_cao_moi = np.linspace(min(chieu_cao), max(chieu_cao), 100).reshape(-1, 1)
    can_nang_du_doan = model.predict(chieu_cao_moi)

    # Trực quan hóa dữ liệu
    plt.scatter(chieu_cao, can_nang, color='blue', label='Dữ liệu thực tế')  # Dữ liệu gốc
    plt.plot(chieu_cao_moi, can_nang_du_doan, color='red', label='Đường hồi quy')  # Đường hồi quy

    # Thêm tiêu đề và nhãn cho các trục
    plt.title("Biểu đồ hồi quy tuyến tính")
    plt.xlabel("Chiều cao (cm)")
    plt.ylabel("Cân nặng (kg)")

    # Hiển thị đồ thị với chú thích
    plt.legend()
    plt.show()

# Dữ liệu từ bảng ban đầu
chieu_cao = np.array([147, 150, 153, 155, 158, 160, 163, 165, 168, 170, 173, 175, 178, 180, 183]).reshape(-1, 1)
can_nang = np.array([49, 50, 51, 52, 54, 56, 58, 59, 60, 72, 63, 64, 66, 67, 68])

# Lựa chọn chế độ
print("Tùy chọn:")
print("1. Nhập chiều cao mới để dự đoán cân nặng.")
print("2. Nhập mẫu dữ liệu mới để dự đoán cân nặng.")
option = int(input("Chọn 1 hoặc 2: "))

if option == 1:
    # Khởi tạo mô hình hồi quy tuyến tính với dữ liệu có sẵn
    model = LinearRegression()
    model.fit(chieu_cao, can_nang)
    new_height, predicted_weight = du_doan_can_nang(model)
    ve_bieu_do(chieu_cao, can_nang, model)  # Vẽ biểu đồ với dữ liệu ban đầu
    
elif option == 2:
    # Nhập mẫu dữ liệu mới
    chieu_cao_moi, can_nang_moi = nhap_du_lieu_moi()
    # Khởi tạo mô hình hồi quy tuyến tính với dữ liệu mới
    model = LinearRegression()
    model.fit(chieu_cao_moi, can_nang_moi)
    new_height, predicted_weight = du_doan_can_nang(model)
    ve_bieu_do(chieu_cao_moi, can_nang_moi, model)  # Vẽ biểu đồ với dữ liệu mới

else:
    print("Không hợp lệ!")
