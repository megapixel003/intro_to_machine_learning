import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Dữ liệu mẫu (x: Nhiệt độ, y: Độ ẩm)
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

# Mẫu mới
new_sample = np.array([[21, 52]])

# Tạo DataFrame cho Seaborn
import pandas as pd

df = pd.DataFrame(data, columns=["Temperature", "Humidity"])
df["Room Type"] = labels

# Trực quan hóa dữ liệu bằng biểu đồ scatter
plt.figure(figsize=(10, 6))
sns.scatterplot(x="Temperature", y="Humidity", hue="Room Type", data=df, s=100, palette="coolwarm", style="Room Type", markers={"Living Room": "o", "Bedroom": "s"})

# Đánh dấu mẫu mới trên biểu đồ
plt.scatter(new_sample[0, 0], new_sample[0, 1], color='green', s=200, edgecolor='black', label="New Sample (21°C, 52%)")

# Cấu hình biểu đồ
plt.title("Temperature vs Humidity with Room Type")
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.legend()
plt.grid(True)

# Hiển thị biểu đồ
plt.show()
