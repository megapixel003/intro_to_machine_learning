import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# Tạo moon dataset với 100 mẫu và một ít nhiễu
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

# Vẽ tập dữ liệu
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn', edgecolors='k', s=100)
plt.title('Moon Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
