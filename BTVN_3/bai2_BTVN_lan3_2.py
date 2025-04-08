import numpy as np
import matplotlib.pyplot as plt

# Định nghĩa hàm mục tiêu
def f(x):
    return x**2 + 6 * np.sin(x)

# Tạo dữ liệu cho x trong khoảng [-4, 4]
x = np.linspace(-4, 4, 400)
y = f(x)

# Định nghĩa đạo hàm của hàm mục tiêu
def df(x):
    return 2 * x + 6 * np.cos(x)

# Thuật toán gradient descent
def gradient_descent(x0, alpha, num_iters):
    x = x0
    history = [x0]
    for i in range(num_iters):
        x = x - alpha * df(x)
        history.append(x)
    return x, history

# Các tham số đầu vào
alpha = 0.1  # tốc độ học
x0 = -4  # giá trị khởi tạo trong khoảng [3, 4] hoặc [-4, -3]
num_iters = 10  # số lần lặp

# Thực hiện thuật toán gradient descent
xmin, history = gradient_descent(x0, alpha, num_iters)

# In kết quả
print(f"Giá trị x tối ưu: {xmin}")
print(f"Giá trị f(x) tại x tối ưu: {f(xmin)}")

# Trực quan hóa quá trình gradient descent
history_y = [f(x) for x in history]

plt.figure(figsize=(10, 6))
plt.plot(x, y, label=r"$f(x) = x^2 + 6\sin(x)$")
plt.scatter(history, history_y, color='red', label="Các bước Gradient Descent")
plt.title("Quá trình Gradient Descent")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()