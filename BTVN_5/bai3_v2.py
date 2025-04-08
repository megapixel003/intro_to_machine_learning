from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Bước 1: Tải dữ liệu hoa diên vĩ
iris = load_iris()
X, y = iris.data, iris.target

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Khởi tạo mô hình k-NN với k=1 và huấn luyện
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Khởi tạo mô hình Decision Tree và huấn luyện
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Bước 2: Tính toán độ chính xác của cả hai mô hình
knn_predictions = knn.predict(X_test)
decision_tree_predictions = decision_tree.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_predictions)
decision_tree_accuracy = accuracy_score(y_test, decision_tree_predictions)

# Hiển thị độ chính xác
print(f"Độ chính xác của mô hình k-NN (k=1): {knn_accuracy:.2f}")
print(f"Độ chính xác của mô hình Decision Tree: {decision_tree_accuracy:.2f}")

# So sánh độ chính xác giữa hai mô hình
if decision_tree_accuracy > knn_accuracy:
    print("Mô hình Decision Tree có độ chính xác cao hơn.")
elif decision_tree_accuracy < knn_accuracy:
    print("Mô hình k-NN có độ chính xác cao hơn.")
else:
    print("Hai mô hình có độ chính xác tương đương.")

# Bước 3: Hiển thị ma trận nhầm lẫn cho k-NN và Decision Tree
print("\nMa trận nhầm lẫn cho mô hình k-NN:")
ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test)
plt.show()

print("\nMa trận nhầm lẫn cho mô hình Decision Tree:")
ConfusionMatrixDisplay.from_estimator(decision_tree, X_test, y_test)
plt.show()

# Bước 4: Vẽ biên giới quyết định (decision boundary) cho k-NN và Decision Tree
# Sử dụng 2 thuộc tính đầu tiên để dễ hiển thị
X_train_2D = X_train[:, :2]  # Lấy 2 thuộc tính đầu tiên
X_test_2D = X_test[:, :2]

# Huấn luyện lại các mô hình chỉ với 2 thuộc tính
knn_2D = KNeighborsClassifier(n_neighbors=1)
knn_2D.fit(X_train_2D, y_train)

decision_tree_2D = DecisionTreeClassifier(random_state=42)
decision_tree_2D.fit(X_train_2D, y_train)

# Hàm vẽ biên giới quyết định
def plot_decision_boundaries(X, y, model, model_name):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    h = .02  # kích thước lưới
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Dự đoán nhãn cho mỗi điểm trong lưới
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Vẽ biên giới quyết định và các điểm dữ liệu
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.title(f"Decision Boundary for {model_name}")
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.show()

# Vẽ biên giới quyết định cho k-NN
plot_decision_boundaries(X_test_2D, y_test, knn_2D, "k-NN")

# Vẽ biên giới quyết định cho Decision Tree
plot_decision_boundaries(X_test_2D, y_test, decision_tree_2D, "Decision Tree")
