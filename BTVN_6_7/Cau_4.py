import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Bước 1: Tạo dữ liệu huấn luyện
x_train = np.linspace(0, 1, 100)  # 100 giá trị x từ 0 đến 1
y_train = np.sin(6 * np.pi * x_train)  # f(x) = sin(6 * pi * x)

# Bước 2: Xây dựng mô hình mạng neuron
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(1,)),  # Lớp ẩn đầu tiên với 64 neuron
    layers.Dense(64, activation='relu'),  # Lớp ẩn thứ hai với 64 neuron
    layers.Dense(1)  # Lớp đầu ra chỉ có 1 giá trị (y dự đoán)
])

# Compile mô hình
model.compile(optimizer='adam', loss='mean_squared_error')

# Bước 3: Huấn luyện mô hình
model.fit(x_train, y_train, epochs=500, verbose=0)  # Huấn luyện với 500 epoch

# Bước 4: Dự đoán giá trị đầu ra
x_test = np.linspace(0, 1, 100)
y_pred = model.predict(x_test)

# Bước 5: Vẽ biểu đồ so sánh
plt.plot(x_test, np.sin(6 * np.pi * x_test), label='f(x) = sin(6πx)', color='blue')
plt.plot(x_test, y_pred, label='Neural Network Approximation', color='red')
plt.legend()
plt.title('Comparison between f(x) and Neural Network Output')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
