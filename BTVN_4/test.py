import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Tạo bộ dữ liệu moon dataset
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

# 2. Tạo và huấn luyện mô hình LinearSVC
model = make_pipeline(StandardScaler(), LinearSVC(random_state=42))
model.fit(X, y)

# 3. Vẽ đường bao phân lớp
def plot_decision_boundary(model, X, y, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = [X[:, 0].min() - 0.5, X[:, 0].max() + 0.5]
    ylim = [X[:, 1].min() - 0.5, X[:, 1].max() + 0.5]

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 500),
                         np.linspace(ylim[0], ylim[1], 500))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Vẽ các đường phân lớp
    ax.contour(xx, yy, Z, colors=['black','red','black'], levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # Vẽ các điểm dữ liệu
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn', edgecolors='k')

    # Tạo bảng chú thích cho các điểm dữ liệu
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)

    # Tạo bảng chú thích cho các đường phân lớp
    handles = [
        plt.Line2D([0], [0], color='black', linestyle='--', lw=2, label='Margin'),
        plt.Line2D([0], [0], color='red', linestyle='-', lw=2, label='Decision Boundary')
    ]
    ax.legend(handles=handles, loc='upper left')

# 4. Vẽ đồ thị
plt.figure(figsize=(8, 6))
plot_decision_boundary(model, X, y)
plt.title('Đường bao phân lớp LinearSVC trên bộ dữ liệu Moon_Dataset')
plt.show()
