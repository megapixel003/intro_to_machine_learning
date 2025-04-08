import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist

# Tải dữ liệu Fashion MNIST
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Chuẩn hóa dữ liệu
train_images = train_images / 255.0
test_images = test_images / 255.0

# Xây dựng mô hình mạng neuron mới
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(512, activation='relu'),  # Lớp ẩn đầu tiên với 512 neuron
    layers.Dropout(0.5),  # Dropout để giảm overfitting
    layers.Dense(256, activation='relu'),  # Lớp ẩn thứ hai với 256 neuron
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),  # Lớp ẩn thứ ba với 128 neuron
    layers.Dense(10, activation='softmax')  # Lớp đầu ra với 10 lớp phân loại
])

# Compile mô hình
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

# Đánh giá mô hình
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")
