# Mô tả bài thực hành 3 – Phân nhóm khách hàng và phân đoạn ảnh bằng K-Means

## 1. Tổng quan

Bài thực hành 3 gồm hai nội dung độc lập, tập trung vào thuật toán K-Means và các phương pháp phân cụm:

1. **Phân nhóm khách hàng:** chia khách hàng thành các nhóm dựa trên độ tuổi, thu nhập hằng năm và điểm chi tiêu.
2. **Phân đoạn ảnh bằng K-Means:** nhóm các điểm ảnh theo màu RGB để tạo ra ảnh có số lượng màu đại diện ít hơn.

Mục tiêu là thực hành quy trình xử lý dữ liệu, phân tích khám phá dữ liệu, chuẩn hóa đặc trưng, xây dựng mô hình phân cụm, đánh giá kết quả và trực quan hóa.

## 2. Phần 1 – Phân nhóm khách hàng

### Dữ liệu đầu vào

- Tệp dữ liệu: `data/raw/Mall_Customers.csv`.
- Ba đặc trưng được sử dụng:
  - `Age`: tuổi khách hàng.
  - `Annual Income (k$)`: thu nhập hằng năm, tính theo nghìn USD.
  - `Spending Score (1-100)`: điểm chi tiêu.
- Cột `Gender` có thể được dùng để mô tả đặc điểm của từng nhóm, nhưng không đưa vào ma trận đặc trưng dùng để phân cụm.

### Quy trình xử lý

- Kiểm tra dữ liệu thiếu và giá trị ngoại lệ bằng phương pháp IQR.
- Khảo sát phân phối và mối quan hệ giữa các đặc trưng bằng biểu đồ histogram và biểu đồ phân tán.
- Sử dụng `StandardScaler` để chuẩn hóa ba đặc trưng.
- Thử nghiệm K-Means, DBSCAN và phân cụm phân cấp Agglomerative.
- Sử dụng hệ số Silhouette để so sánh chất lượng phân cụm.
- Sử dụng PCA để giảm dữ liệu xuống hai chiều nhằm trực quan hóa kết quả K-Means.

### Kết quả đầu ra

- Các tệp CSV/JSON trong `outputs/metrics/`.
- Dữ liệu khách hàng kèm nhãn nhóm trong `outputs/customer_segments.csv`.
- Các biểu đồ trong `outputs/figures/`.

## 3. Phần 2 – Phân đoạn ảnh bằng K-Means

### Dữ liệu đầu vào

Một ảnh màu bất kỳ. Nếu không truyền đường dẫn ảnh, chương trình sử dụng ảnh mẫu astronaut được cung cấp trong thư viện scikit-image.

### Quy trình xử lý

- Đọc ảnh và chuyển ảnh về không gian màu RGB.
- Thay đổi kích thước ảnh theo giới hạn đã thiết lập để giảm thời gian xử lý.
- Chuyển ảnh thành ma trận điểm ảnh; mỗi điểm ảnh được biểu diễn bởi ba giá trị đỏ, xanh lá và xanh dương (R, G, B).
- Sử dụng K-Means để chia các điểm ảnh thành K nhóm.
- Thay màu của mỗi điểm ảnh bằng màu tâm cụm tương ứng.

### Kết quả đầu ra

- `outputs/segmented_images/segmented_kK.png`: ảnh sau khi phân đoạn.
- `outputs/segmented_images/comparison_kK.png`: ảnh gốc và ảnh sau phân đoạn được đặt cạnh nhau để so sánh.

## 4. Công nghệ và thư viện sử dụng

- Python: ngôn ngữ lập trình chính.
- Pandas, NumPy: xử lý dữ liệu và tính toán số học.
- Matplotlib, Seaborn: trực quan hóa dữ liệu.
- Scikit-learn: chuẩn hóa dữ liệu, xây dựng và đánh giá mô hình phân cụm.
- Pillow, scikit-image: đọc, xử lý ảnh và cung cấp ảnh mẫu.
- Flask: xây dựng giao diện trang web để trình bày kết quả.

## 5. Giới hạn của bài thực hành

- Kết quả phân cụm phụ thuộc vào đặc trưng đầu vào, phương pháp chuẩn hóa và tham số của thuật toán.
- Điểm Silhouette của DBSCAN được tính trên các điểm không bị đánh dấu là nhiễu; cần xem xét thêm tỷ lệ điểm nhiễu khi so sánh.
- PCA chỉ phục vụ trực quan hóa kết quả trên mặt phẳng hai chiều, không thay thế dữ liệu ba chiều được sử dụng để huấn luyện mô hình phân cụm.
- Phân đoạn ảnh trong bài chỉ nhóm các điểm ảnh theo màu RGB, không nhận biết ý nghĩa của vật thể hoặc đối tượng trong ảnh.
