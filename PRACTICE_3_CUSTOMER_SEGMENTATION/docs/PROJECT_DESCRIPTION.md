# Mô tả Practice 3 – Customer Segmentation & Image Segmentation

## 1. Tổng quan

Practice 3 gồm hai nội dung thực hành độc lập sử dụng thuật toán K-Means và các phương pháp phân cụm:

1. **Customer Segmentation:** phân nhóm khách hàng dựa trên độ tuổi, thu nhập hằng năm và điểm chi tiêu.
2. **K-Means Image Segmentation:** phân nhóm pixel dựa trên màu RGB để tạo ảnh có số lượng màu đại diện ít hơn.

Mục tiêu là thực hành quy trình xử lý dữ liệu, khám phá dữ liệu (EDA), chuẩn hóa đặc trưng, huấn luyện thuật toán clustering, đánh giá kết quả và trực quan hóa.

## 2. Phần 1 – Customer Segmentation

### Input
- File: `data/raw/Mall_Customers.csv`
- Các đặc trưng dùng để phân cụm:
  - `Age`: tuổi khách hàng.
  - `Annual Income (k$)`: thu nhập hằng năm (nghìn USD).
  - `Spending Score (1-100)`: điểm chi tiêu.
- `Gender` có thể được dùng để mô tả hồ sơ cụm, không được đưa vào vector đặc trưng phân cụm.

### Xử lý
- Kiểm tra dữ liệu thiếu và ngoại lệ bằng IQR.
- Khám phá phân phối và mối quan hệ giữa các đặc trưng bằng histogram và scatter plot.
- Dùng StandardScaler để chuẩn hóa ba đặc trưng.
- Thử K-Means, DBSCAN và Agglomerative Clustering.
- Dùng Silhouette Coefficient để so sánh chất lượng phân cụm.
- Dùng PCA giảm xuống hai chiều để trực quan hóa kết quả K-Means.

### Output
Các tệp CSV/JSON trong `outputs/metrics/`, dữ liệu khách hàng có nhãn cụm trong `outputs/customer_segments.csv`, cùng các biểu đồ trong `outputs/figures/`.

## 3. Phần 2 – K-Means Image Segmentation

### Input
Một ảnh màu bất kỳ; nếu không truyền đường dẫn ảnh, chương trình dùng ảnh mẫu astronaut có sẵn trong scikit-image.

### Xử lý
- Đọc ảnh và chuyển sang RGB.
- Resize ảnh theo giới hạn kích thước.
- Chuyển ảnh thành ma trận pixel, mỗi pixel có ba giá trị R, G, B.
- K-Means gom các pixel thành K cụm.
- Thay màu mỗi pixel bằng màu centroid của cụm tương ứng.

### Output
- `outputs/segmented_images/segmented_kK.png`: ảnh sau phân đoạn.
- `outputs/segmented_images/comparison_kK.png`: ảnh gốc và ảnh phân đoạn đặt cạnh nhau.

## 4. Công nghệ sử dụng

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Pillow, scikit-image
- Flask cho dashboard minh họa kết quả

## 5. Giới hạn

- Kết quả phân cụm phụ thuộc vào đặc trưng, cách chuẩn hóa và tham số.
- Silhouette của DBSCAN được tính trên các điểm không bị gán nhãn noise; cần xem xét thêm tỷ lệ noise trước khi so sánh.
- PCA phục vụ trực quan hóa, không phải dữ liệu đầu vào thay thế cho quá trình phân cụm.
- Image segmentation chỉ gom pixel theo màu RGB, không nhận diện ngữ nghĩa hay ranh giới đối tượng.
