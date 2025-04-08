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

# Tạo DataFrame cho Seaborn
df = pd.DataFrame(data, columns=["Nhiệt độ", "Độ ẩm"])
df["Loại phòng"] = labels

# Tạo mô hình phân loại KNN
X = df[["Nhiệt độ", "Độ ẩm"]]
y = df["Loại phòng"]
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# Nhập mẫu mới 
new_temp = float(input("Nhập nhiệt độ của mẫu mới (°C): "))
new_hum = float(input("Nhập độ ẩm của mẫu mới (%): "))
new_sample = np.array([[new_temp, new_hum]])

# Dự đoán kiểu phòng của mẫu mới
label_dudoan = knn.predict(new_sample)[0]
print(f"Kiểu phòng của mẫu mới (Nhiệt độ {new_temp}°C, Độ ẩm {new_hum}%): {label_dudoan}")

# Tạo biểu đồ scatter với Seaborn
plt.figure(figsize=(10, 10))
sns.scatterplot(x="Nhiệt độ", y="Độ ẩm", hue="Loại phòng", data=df, s=100, palette={"Bedroom": "blue","Living Room": "red"}, style="Loại phòng", markers={"Bedroom": "o", "Living Room": "s"})

# Đánh dấu mẫu mới trên biểu đồ 
marker = 's' if label_dudoan == "Living Room" else 'o'
color = 'green'
plt.scatter(new_temp, new_hum, color=color, s=100, marker=marker, label=f"Mẫu mới ({new_temp}°C, {new_hum}%)")

# Cấu hình biểu đồ
plt.title("Phân biệt loại phòng với Nhiệt độ - Độ ẩm ")
plt.xlabel("Nhiệt độ (°C)")
plt.ylabel("Độ ẩm (%)")
plt.legend()
plt.grid(True)

# Hiển thị biểu đồ
plt.show()
