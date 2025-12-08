import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# BƯỚC 2: TẠO DỮ LIỆU
X, y = make_blobs(
    n_samples=100,
    n_features=2,
    centers=4,
    cluster_std=1,
    random_state=4
)

# BƯỚC 3: VẼ DỮ LIỆU BAN ĐẦU
plt.figure(figsize=(9, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', s=50)
plt.title("Phân bố dữ liệu ban đầu (4 lớp)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# BƯỚC 4: CHIA TRAIN/TEST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

plt.figure(figsize=(9, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='o', s=40)
plt.title("Dữ liệu Test (nhãn thật)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# BƯỚC 5: TRAIN KNN VỚI k=5
knn5 = KNeighborsClassifier(5)
knn5.fit(X_train, y_train)
y_pred_5 = knn5.predict(X_test)

plt.figure(figsize=(9, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_5, marker='o', s=40)
plt.title("Kết quả phân lớp với k=5")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# BƯỚC 6: TRAIN KNN VỚI k=1
knn1 = KNeighborsClassifier(1)
knn1.fit(X_train, y_train)
y_pred_1 = knn1.predict(X_test)

plt.figure(figsize=(9, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_1, marker='o', s=40)
plt.title("Kết quả phân lớp với k=1")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# BƯỚC 7: TÌM k TỐI ƯU BẰNG GridSearchCV
param_grid = {'n_neighbors': np.arange(1, 10)}

knn_grid = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    cv=5
)

knn_grid.fit(X, y)
print("Giá trị k tối ưu:", knn_grid.best_params_)

# BƯỚC 8: HÀM KNN TỰ CÀI ĐẶT
def KNN(X_train, X_test, y_train, k):
    num_test = X_test.shape[0]
    num_train = X_train.shape[0]
    distances = np.zeros((num_test, num_train))

    for i in range(num_test):
        for j in range(num_train):
            distances[i, j] = np.sqrt(np.sum((X_test[i, :] - X_train[j, :]) ** 2))

    results = []
    for i in range(num_test):
        zipped = list(zip(distances[i, :], y_train))
        res = sorted(zipped, key=lambda x: x[0])
        results_topk = res[:k]

        classes = {}
        for _, label in results_topk:
            label = int(label)
            classes[label] = classes.get(label, 0) + 1

        results.append(max(classes, key=classes.get))

    return np.array(results)

# TEST HÀM KNN THỦ CÔNG
X2, y2 = make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,
    cluster_std=1,
    random_state=4
)

X_test_manual = np.array([[1, 3]])
results = KNN(X2, X_test_manual, y2, 3)

print("Nhãn dự đoán cho điểm (1,3) với k=3:", results)
