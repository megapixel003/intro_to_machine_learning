import numpy as np
import matplotlib.pyplot as plt

# Định nghĩa hàm mục tiêu
def f(x):
    return x**2 + 6 * np.sin(x)

# Tạo dữ liệu cho x trong khoảng [-4, 4]
x = np.linspace(-4, 4, 400)
y = f(x)

# Vẽ đồ thị
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=r"$f(x) = x^2 + 6\sin(x)$")
plt.title("Đồ thị hàm mục tiêu")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()
