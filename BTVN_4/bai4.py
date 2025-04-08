from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Tạo dữ liệu mẫu
data = np.array([
    [22, 50], [24, 45], [23, 47], [21, 49], [25, 44], 
    [19, 55], [20, 60], [18, 58], [17, 57], [20, 62], 
    [23, 46], [24, 48], [19, 54], [18, 56]
])

labels = np.array([
    "Living Room", "Living Room", "Living Room", "Living Room", "Living Room", 
    "Bedroom", "Bedroom", "Bedroom", "Bedroom", "Bedroom", 
    "Living Room", "Living Room", "Bedroom", "Bedroom"
])

# Mẫu mới
new_sample = np.array([[21, 52]])

# Khởi tạo mô hình k-NN với k=5
k = 5
knn = KNeighborsClassifier(n_neighbors=k)

# Huấn luyện mô hình
knn.fit(data, labels)

# Dự đoán kiểu phòng cho mẫu mới
predicted_label = knn.predict(new_sample)
print(predicted_label[0])
