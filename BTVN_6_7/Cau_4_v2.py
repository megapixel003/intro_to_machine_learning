#Cải thiện độ chính xác cao hơn, bám hơn


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Tạo dữ liệu huấn luyện
x_train = np.linspace(0, 1, 500)  # Tăng số lượng mẫu lên 500
y_train = np.sin(6 * np.pi * x_train)

# Xây dựng mô hình với các cải tiến
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(1,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),  # Thêm Dropout để tránh overfitting
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Lớp đầu ra
])

# Sử dụng Adam với learning rate nhỏ hơn
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Huấn luyện mô hình
model.fit(x_train, y_train, epochs=1000, verbose=0)  # Huấn luyện với 1000 epoch

# Dự đoán giá trị đầu ra
x_test = np.linspace(0, 1, 100)
y_pred = model.predict(x_test)

# Vẽ biểu đồ so sánh
plt.plot(x_test, np.sin(6 * np.pi * x_test), label='f(x) = sin(6πx)', color='blue')
plt.plot(x_test, y_pred, label='Neural Network Approximation', color='red')
plt.legend()
plt.title('Comparison between f(x) and Neural Network Output')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
