import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Dữ liệu mẫu [x, y] với x là nhiệt độ, y là độ ẩm
data = np.array([
    [22, 50], [24, 45], [23, 47], [21, 49], [25, 44], 
    [19, 55], [20, 60], [18, 58], [17, 57], [20, 62], 
    [23, 46], [24, 48], [19, 54], [18, 56]
])

# Nhãn tương ứng với các kiểu phòng
labels = np.array([
    "Living Room", "Living Room", "Living Room", "Living Room", "Living Room", 
    "Bedroom", "Bedroom", "Bedroom", "Bedroom", "Bedroom", 
    "Living Room", "Living Room", "Bedroom", "Bedroom"
])

# Tạo mô hình KNN và dự đoán
knn = KNeighborsClassifier(n_neighbors=3).fit(data, labels)
new_temp = float(input("Nhập nhiệt độ của mẫu mới (°C): "))
new_hum = float(input("Nhập độ ẩm của mẫu mới (%): "))
new_sample = np.array([[new_temp, new_hum]])
predicted_label = knn.predict(new_sample)[0]

# Vẽ biểu đồ với mẫu mới
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, s=100, 
                palette={"Bedroom": "blue", "Living Room": "red"}, 
                style=labels, markers={"Bedroom": "o", "Living Room": "s"})
plt.scatter(new_temp, new_hum, color='green', s=100, marker='s' if predicted_label == "Living Room" else 'o', 
            label=f"Mẫu mới ({new_temp}°C, {new_hum}%)")

plt.title("Phân biệt loại phòng với Nhiệt độ - Độ ẩm")
plt.xlabel("Nhiệt độ (°C)")
plt.ylabel("Độ ẩm (%)")
plt.legend()
plt.grid(True)
plt.show()

print(f"Kiểu phòng của mẫu mới (Nhiệt độ {new_temp}°C, Độ ẩm {new_hum}%): {predicted_label}")
