import numpy as np
import matplotlib.pyplot as plt

# Hàm mục tiêu
def f(x):
    return x**2 + 6 * np.sin(x)

# Tạo các giá trị x trong khoảng [-4, 4]
x = np.linspace(-4, 4, 1000)

# Tính giá trị hàm f(x)
y = f(x)

# Gradient của hàm mục tiêu
def df(x):
    return 2*x + 6*np.cos(x)

# Hàm Gradient Descent
def gradient_descent(f, df, x0, learning_rate, num_iterations, epsilon):
    x = x0
    history = [x0]
    for i in range(num_iterations):
        grad = df(x)
        if abs(grad) <= epsilon:  # Điều kiện dừng nếu gradient nhỏ hơn ngưỡng epsilon
            break
        x = x - learning_rate * grad
        history.append(x)
    return x, history

# Tham số mặc định
learning_rate = 0.001
x0 = 0
num_iterations = 1000000
epsilon = 1e-3

# Tìm điểm cực tiểu
xmin, history = gradient_descent(f, df, x0, learning_rate, num_iterations, epsilon)

# In kết quả
print(f"Điểm cực tiểu ước lượng là: x = {xmin}, f(x) = {f(xmin)}")

# Trực quan hóa quá trình Gradient Descent
history = np.array(history)
plt.plot(x, y, label='f(x) = x^2 + 6sin(x)')
plt.plot(history, f(history), 'ro-', label='Gradient Descent')
plt.title('Quá trình tìm cực tiểu bằng Gradient Descent')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()

