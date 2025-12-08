# ============================================
# BÀI THỰC HÀNH K-MEANS (THEO TÀI LIỆU)
# Có bổ sung giải thích chi tiết từng bước
# ============================================

# BƯỚC 1: IMPORT CÁC THƯ VIỆN CẦN THIẾT
import numpy as np                     # Thư viện tính toán số học, ma trận, vector
import matplotlib.pyplot as plt        # Thư viện vẽ đồ thị, trực quan dữ liệu
from scipy.spatial.distance import cdist   # Hàm cdist hỗ trợ tính ma trận khoảng cách

# -------------------------------
# BƯỚC 2: TẠO DỮ LIỆU GIẢ LẬP
# -------------------------------
# Mục tiêu: Khởi tạo dữ liệu gồm nhiều điểm nằm xung quanh 3 tâm cụm:
# (2, 2), (9, 2), (4, 9)

# Tọa độ 3 tâm cụm lý thuyết
means = np.array([[2, 2],
                  [9, 2],
                  [4, 9]])

# Ma trận hiệp phương sai (covariance matrix)
# Ở đây chọn ma trận đường chéo: phương sai = 2 theo cả 2 chiều, không tương quan
cov = np.array([[2, 0],
                [0, 2]])

n_samples = 500     # số lượng điểm cho MỖI cụm (tổng sẽ là 3 * 500 = 1500 điểm)
n_cluster = 3       # số cụm K = 3

# Sinh dữ liệu ngẫu nhiên phân phối chuẩn đa biến xung quanh từng tâm
X0 = np.random.multivariate_normal(means[0], cov, n_samples)
X1 = np.random.multivariate_normal(means[1], cov, n_samples)
X2 = np.random.multivariate_normal(means[2], cov, n_samples)

# Ghép 3 cụm thành một tập dữ liệu duy nhất X
# X có kích thước (1500, 2): 1500 điểm, mỗi điểm 2 chiều (x, y)
X = np.concatenate((X0, X1, X2), axis=0)

# ----------------------------------------
# BƯỚC 3: XEM PHÂN BỐ DỮ LIỆU BAN ĐẦU
# ----------------------------------------
plt.figure()
plt.xlabel('x')         # Nhãn trục hoành
plt.ylabel('y')         # Nhãn trục tung
plt.title('Phân bố dữ liệu ban đầu')
# Vẽ tất cả điểm dữ liệu màu xanh ('b') dạng chấm tròn ('o')
plt.plot(X[:, 0], X[:, 1], 'bo', markersize=3)
plt.show()


# ============================================
# PHẦN 2: CÀI ĐẶT THUẬT TOÁN K-MEANS
# ============================================

# ----------------------------------------
# BƯỚC 4: HÀM KHỞI TẠO K TÂM CỤM BAN ĐẦU
# ----------------------------------------
def kmeans_init_centers(X, n_cluster):
    """
    Chọn ngẫu nhiên n_cluster điểm trong X làm tâm cụm ban đầu.
    - X: dữ liệu, dạng (N, d)
    - n_cluster: số cụm K
    """
    # np.random.choice(X.shape[0], n_cluster, replace=False)
    #   -> chọn ngẫu nhiên n_cluster chỉ số hàng khác nhau trong [0, N-1]
    random_idx = np.random.choice(X.shape[0], n_cluster, replace=False)
    # Tâm cụm ban đầu là các điểm dữ liệu ứng với index đã chọn
    centers = X[random_idx]
    return centers


# ----------------------------------------
# BƯỚC 5: HÀM GÁN NHÃN (PREDICT LABELS)
# ----------------------------------------
def kmeans_predict_labels(X, centers):
    """
    Gán mỗi điểm dữ liệu vào cụm có tâm gần nhất.
    - X: dữ liệu, (N, d)
    - centers: tâm cụm hiện tại, (K, d)
    Trả về:
    - labels: mảng kích thước (N,), mỗi phần tử là chỉ số cụm 0..K-1
    """
    # cdist(X, centers) trả về ma trận khoảng cách D có kích thước (N, K)
    # D[i, j] = khoảng cách từ điểm X[i] đến tâm centers[j]
    D = cdist(X, centers)

    # np.argmin(D, axis=1) -> với mỗi hàng (mỗi điểm), lấy index cột nhỏ nhất
    # chính là cụm gần nhất.
    labels = np.argmin(D, axis=1)
    return labels


# ----------------------------------------
# BƯỚC 6: HÀM CẬP NHẬT TÂM CỤM
# ----------------------------------------
def kmeans_update_centers(X, labels, n_cluster):
    """
    Cập nhật lại vị trí các tâm cụm dựa trên nhãn hiện tại.
    - X: dữ liệu, (N, d)
    - labels: nhãn cụm của từng điểm, (N,)
    - n_cluster: số cụm K
    Trả về:
    - centers mới, (K, d)
    """
    # Khởi tạo ma trận tâm mới với giá trị 0
    centers = np.zeros((n_cluster, X.shape[1]))

    for k in range(n_cluster):
        # Lấy tất cả điểm thuộc cụm k:
        # labels == k sẽ là một mask True/False, dùng để lọc theo hàng.
        Xk = X[labels == k, :]

        # Tính trung bình theo hàng (axis=0), kết quả là vector 1 x d
        centers[k, :] = np.mean(Xk, axis=0)

    return centers


# ----------------------------------------
# BƯỚC 7: HÀM KIỂM TRA HỘI TỤ
# ----------------------------------------
def kmeans_has_converged(centers, new_centers):
    """
    Kiểm tra xem thuật toán đã hội tụ hay chưa.
    Ở đây, ta so sánh hai bộ tâm cụm (cũ và mới).
    Nếu chúng giống nhau (không đổi) thì coi như hội tụ.
    """
    # Đưa từng tâm (vector) thành tuple rồi cho vào set để so sánh.
    # Việc dùng set giúp không quan tâm đến thứ tự phần tử,
    # miễn là các tâm trùng tọa độ.
    old_set = set([tuple(c) for c in centers])
    new_set = set([tuple(c) for c in new_centers])
    return old_set == new_set


# ----------------------------------------
# BƯỚC 8: HÀM VẼ KẾT QUẢ LÊN ĐỒ THỊ
# ----------------------------------------
def kmeans_visualize(X, centers, labels, n_cluster, title):
    """
    Vẽ dữ liệu và tâm cụm lên đồ thị để quan sát.
    - X       : dữ liệu
    - centers : tâm cụm
    - labels  : nhãn cụm của từng điểm
    - n_cluster: số cụm
    - title   : tiêu đề đồ thị
    """
    plt.figure()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)

    # Danh sách màu sử dụng cho cụm và tâm
    # (Ở đây đủ dùng cho K <= 4)
    plt_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    for i in range(n_cluster):
        # Lấy tất cả điểm thuộc cụm i
        data = X[labels == i]

        # Vẽ điểm dữ liệu của cụm i: dùng tam giác '^', màu plt_colors[i]
        plt.plot(
            data[:, 0], data[:, 1],
            plt_colors[i] + '^',
            markersize=3,
            label='cluster_' + str(i)
        )

        # Vẽ tâm cụm i: dùng hình tròn 'o', dùng màu khác (i + 4)
        plt.plot(
            centers[i][0], centers[i][1],
            plt_colors[i + 4] + 'o',
            markersize=10,
            label='center_' + str(i)
        )

    plt.legend()   # Hiển thị chú thích cụm và tâm
    plt.show()


# ----------------------------------------
# BƯỚC 9: TOÀN BỘ THUẬT TOÁN K-MEANS
# ----------------------------------------
def kmeans(init_centers, init_labels, X, n_cluster):
    """
    Cài đặt toàn bộ thuật toán K-means:
    1. Khởi tạo tâm
    2. Lặp:
        - Gán nhãn (assign points to clusters)
        - Cập nhật tâm (update cluster centers)
        - Kiểm tra hội tụ
    3. Trả về tâm cuối cùng, nhãn và số lần lặp.
    """
    centers = init_centers
    labels = init_labels
    times = 0  # đếm số lần lặp

    while True:
        # Bước 1: Gán nhãn cho mỗi điểm theo tâm hiện tại
        labels = kmeans_predict_labels(X, centers)

        # Vẽ dữ liệu với nhãn hiện tại
        kmeans_visualize(
            X, centers, labels, n_cluster,
            'Assigned label for data at time = ' + str(times + 1)
        )

        # Bước 2: Cập nhật lại tâm cụm
        new_centers = kmeans_update_centers(X, labels, n_cluster)

        # Bước 3: Kiểm tra hội tụ
        if kmeans_has_converged(centers, new_centers):
            # Nếu hội tụ thì dừng vòng lặp
            break

        # Nếu chưa hội tụ thì cập nhật tâm và tiếp tục lặp
        centers = new_centers

        # Vẽ tâm cụm mới sau khi cập nhật
        kmeans_visualize(
            X, centers, labels, n_cluster,
            'Update center position at time = ' + str(times + 1)
        )

        times += 1

    # Trả về tâm cuối cùng, nhãn và số lần lặp
    return centers, labels, times


# ----------------------------------------
# BƯỚC 10: GỌI HÀM K-MEANS ĐỂ THỰC THI
# ----------------------------------------
# Khởi tạo tâm cụm ban đầu
init_centers = kmeans_init_centers(X, n_cluster)
print("Tâm cụm khởi tạo ban đầu:")
print(init_centers)

# Khởi tạo nhãn ban đầu: gán tất cả điểm về cụm 0 (chỉ để vẽ lần đầu)
init_labels = np.zeros(X.shape[0])

# Vẽ dữ liệu ban đầu với tâm cụm khởi tạo
kmeans_visualize(
    X, init_centers, init_labels, n_cluster,
    'Init centers in the first run. Assigned all data as cluster 0'
)

# Chạy thuật toán K-means
centers, labels, times = kmeans(init_centers, init_labels, X, n_cluster)

print('Done! K-means has converged after', times, 'iterations.')
print('Tọa độ các tâm cụm cuối cùng:')
print(centers)
