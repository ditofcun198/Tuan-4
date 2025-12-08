import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# BƯỚC 2: TẠO DỮ LIỆU
means = np.array([[2, 2], [9, 2], [4, 9]])
cov = np.array([[2, 0], [0, 2]])
n_samples = 500
n_cluster = 3

X0 = np.random.multivariate_normal(means[0], cov, n_samples)
X1 = np.random.multivariate_normal(means[1], cov, n_samples)
X2 = np.random.multivariate_normal(means[2], cov, n_samples)
X = np.concatenate((X0, X1, X2), axis=0)

# BƯỚC 3: VẼ DỮ LIỆU BAN ĐẦU
plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Phân bố dữ liệu ban đầu')
plt.plot(X[:, 0], X[:, 1], 'bo', markersize=3)
plt.show()

# BƯỚC 4: KHỞI TẠO TÂM CỤM
def kmeans_init_centers(X, n_cluster):
    random_idx = np.random.choice(X.shape[0], n_cluster, replace=False)
    centers = X[random_idx]
    return centers

# BƯỚC 5: GÁN NHÃN
def kmeans_predict_labels(X, centers):
    D = cdist(X, centers)
    labels = np.argmin(D, axis=1)
    return labels

# BƯỚC 6: CẬP NHẬT TÂM CỤM
def kmeans_update_centers(X, labels, n_cluster):
    centers = np.zeros((n_cluster, X.shape[1]))
    for k in range(n_cluster):
        Xk = X[labels == k, :]
        centers[k, :] = np.mean(Xk, axis=0)
    return centers

# BƯỚC 7: KIỂM TRA HỘI TỤ
def kmeans_has_converged(centers, new_centers):
    old_set = set([tuple(c) for c in centers])
    new_set = set([tuple(c) for c in new_centers])
    return old_set == new_set

# BƯỚC 8: VẼ ĐỒ THỊ
def kmeans_visualize(X, centers, labels, n_cluster, title):
    plt.figure()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i in range(n_cluster):
        data = X[labels == i]
        plt.plot(data[:, 0], data[:, 1], plt_colors[i] + '^', markersize=3, label='cluster_' + str(i))
        plt.plot(centers[i][0], centers[i][1], plt_colors[i + 4] + 'o', markersize=10, label='center_' + str(i))
    plt.legend()
    plt.show()

# BƯỚC 9: THUẬT TOÁN K-MEANS
def kmeans(init_centers, init_labels, X, n_cluster):
    centers = init_centers
    labels = init_labels
    times = 0
    while True:
        labels = kmeans_predict_labels(X, centers)
        kmeans_visualize(X, centers, labels, n_cluster, 'Assigned label at time = ' + str(times + 1))
        new_centers = kmeans_update_centers(X, labels, n_cluster)
        if kmeans_has_converged(centers, new_centers):
            break
        centers = new_centers
        kmeans_visualize(X, centers, labels, n_cluster, 'Update center at time = ' + str(times + 1))
        times += 1
    return centers, labels, times

# BƯỚC 10: CHẠY CHƯƠNG TRÌNH
init_centers = kmeans_init_centers(X, n_cluster)
print("Tâm cụm khởi tạo ban đầu:")
print(init_centers)

init_labels = np.zeros(X.shape[0])

kmeans_visualize(
    X, init_centers, init_labels, n_cluster,
    'Init centers - tất cả dữ liệu gán về cluster 0'
)

centers, labels, times = kmeans(init_centers, init_labels, X, n_cluster)

print('Done! K-means hội tụ sau', times, 'lần lặp.')
print('Tọa độ các tâm cụm cuối cùng:')
print(centers)
