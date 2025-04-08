import numpy as np
import matplotlib.pyplot as plt

# Hàm mục tiêu
def f(x):
    return x**2 + 6 * np.sin(x)

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
        
        # In ra kết quả của x tại mỗi vòng lặp
        print(f"Iteration {i+1}: x = {x}, f(x) = {f(x)}")
        
    return x, history

# Tham số đầu vào từ người dùng
learning_rate = input("Nhập tốc độ học (learning rate) (mặc định 0.001): ")
x0 = input("Nhập giá trị thiết lập ban đầu x0 (mặc định 0): ")
num_iterations = input("Nhập số lần lặp (mặc định 1000000): ")

# Sử dụng giá trị mặc định nếu người dùng không nhập
learning_rate = float(learning_rate) if learning_rate else 0.001
x0 = float(x0) if x0 else 0
num_iterations = int(num_iterations) if num_iterations else 1000000
epsilon = 1e-3

# Tìm điểm cực tiểu
xmin, history = gradient_descent(f, df, x0, learning_rate, num_iterations, epsilon)

# In kết quả cuối cùng
print(f"Điểm cực tiểu ước lượng là: x = {xmin}, f(x) = {f(xmin)}")

# Trực quan hóa quá trình Gradient Descent
x = np.linspace(-4, 4, 1000)
y = f(x)
history = np.array(history)

plt.plot(x, y, label='f(x) = x^2 + 6sin(x)')
plt.plot(history, f(history), 'ro-', label='Gradient Descent')
plt.title('Quá trình tìm cực tiểu bằng Gradient Descent')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()
