# Bài thực hành 3 – Phân nhóm khách hàng và phân đoạn ảnh bằng K-Means

## 1. Mục tiêu

Bài thực hành gồm **hai nội dung độc lập**:

1. **Phân nhóm khách hàng:** khám phá dữ liệu khách hàng, chuẩn hóa đặc trưng, phân nhóm và đánh giá chất lượng các nhóm.
2. **Phân đoạn ảnh bằng K-Means:** gom các điểm ảnh theo màu sắc và so sánh ảnh gốc với ảnh sau phân đoạn.

## 2. Cấu trúc thư mục

- `data/raw/Mall_Customers.csv`: bộ dữ liệu khách hàng gồm 200 bản ghi.
- `scripts/customer_clustering.py`: thực hiện phân tích dữ liệu, chuẩn hóa, thử nghiệm K-Means/DBSCAN/Agglomerative, tính hệ số Silhouette và thống kê đặc điểm từng nhóm.
- `scripts/image_segmentation.py`: phân đoạn ảnh bằng K-Means trong không gian màu RGB.
- `outputs/metrics/`: kết quả kiểm tra chất lượng dữ liệu, so sánh thuật toán và thống kê các nhóm.
- `outputs/figures/`: biểu đồ phân tích dữ liệu, hệ số Silhouette và biểu đồ PCA.
- `outputs/segmented_images/`: ảnh sau phân đoạn và ảnh so sánh.
- `docs/`: tài liệu mô tả, dữ liệu, hướng dẫn thực thi và báo cáo kết quả.

## 3. Cài đặt môi trường

Thực hiện lệnh tại thư mục `PRACTICE_3_CUSTOMER_SEGMENTATION`:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4. Phần 1 – Phân nhóm khách hàng

### Dữ liệu đầu vào

Tài liệu mô tả bộ dữ liệu và từ điển dữ liệu nằm tại `docs/DATASET_DOCUMENTATION.md`. Tệp `data/raw/Mall_Customers.csv` gồm 200 dòng và 5 cột; kết quả kiểm tra tệp được cung cấp cho thấy không có giá trị thiếu hoặc dòng trùng lặp.

Ba cột số được sử dụng để phân nhóm:

- `Age`: tuổi khách hàng.
- `Annual Income (k$)`: thu nhập hằng năm, tính theo nghìn USD.
- `Spending Score (1-100)`: điểm chi tiêu.

Có thể có thêm cột `Gender` để mô tả đặc điểm của từng nhóm. Cột này không được đưa vào phép tính khoảng cách khi phân nhóm.

### Cách chạy

```bash
python scripts/customer_clustering.py
```

Hoặc chỉ định bộ dữ liệu và thư mục lưu kết quả:

```bash
python scripts/customer_clustering.py --data data/raw/Mall_Customers.csv --output outputs
```

### Quy trình thực hiện

1. Kiểm tra giá trị thiếu và giá trị ngoại lệ bằng phương pháp IQR.
2. Vẽ biểu đồ phân phối và biểu đồ phân tán để quan sát đặc điểm, mối quan hệ giữa các thuộc tính.
3. Dùng `StandardScaler` để đưa tuổi, thu nhập và điểm chi tiêu về thang đo tương đương.
4. Thử K-Means với K từ 2 đến 10; thử DBSCAN với nhiều giá trị `eps`; thử Agglomerative với K từ 2 đến 5.
5. So sánh hệ số Silhouette. Với DBSCAN, hệ số này được tính trên các điểm không bị đánh dấu là nhiễu. Bảng so sánh cũng ghi số lượng điểm nhiễu, tỷ lệ điểm nhiễu và số điểm được đánh giá.
6. Chọn cấu hình K-Means có hệ số Silhouette cao nhất trong các cấu hình K-Means đã thử; trực quan hóa kết quả bằng PCA hai chiều và lập bảng đặc điểm từng nhóm.

**Lưu ý khi đánh giá:** Hệ số Silhouette càng gần 1 thường cho thấy các nhóm càng tách biệt; gần 0 cho thấy các nhóm có thể chồng lấn; giá trị âm có thể cho thấy điểm dữ liệu được gán vào nhóm chưa phù hợp. Chỉ tính hệ số này khi có ít nhất hai nhóm hợp lệ.

### Các tệp kết quả

- `outputs/metrics/data_quality.csv`
- `outputs/metrics/clustering_comparison.csv`
- `outputs/metrics/cluster_profiles.csv`
- `outputs/metrics/gender_distribution_by_cluster.csv` (nếu dữ liệu có cột giới tính)
- `outputs/metrics/run_summary.json`
- `outputs/customer_segments.csv`
- `outputs/figures/feature_distributions.png`
- `outputs/figures/eda_scatterplots.png`
- `outputs/figures/customer_clusters_pca.png`
- `outputs/figures/silhouette_by_k.png`

## 5. Phần 2 – Phân đoạn ảnh bằng K-Means

Bài thực hành cho phép sử dụng ảnh bất kỳ. Nếu không truyền đường dẫn ảnh, chương trình sử dụng ảnh mẫu có sẵn trong thư viện scikit-image.

### Cách chạy

Sử dụng ảnh mẫu tích hợp:

```bash
python scripts/image_segmentation.py
```

Sử dụng ảnh riêng:

```bash
python scripts/image_segmentation.py --image data/raw/my_image.jpg --k 5
```

Thử số nhóm khác:

```bash
python scripts/image_segmentation.py --image data/raw/my_image.jpg --k 8 --max-size 300
```

### Quy trình thực hiện

1. Đọc ảnh bằng Pillow và chuyển ảnh về hệ màu RGB.
2. Thay đổi kích thước ảnh để giảm thời gian tính toán.
3. Chuyển ảnh thành ma trận có kích thước `(số_điểm_ảnh, 3)`; mỗi điểm ảnh được biểu diễn bởi ba giá trị màu RGB.
4. Dùng K-Means với K tâm cụm; mỗi điểm ảnh được gán vào tâm cụm gần nhất theo khoảng cách Euclid.
5. Thay màu điểm ảnh bằng màu của tâm cụm tương ứng, chuyển ma trận về kích thước ảnh ban đầu và lưu kết quả.
6. Đặt ảnh gốc và ảnh sau phân đoạn cạnh nhau để so sánh.

Kết quả được lưu tại `outputs/segmented_images/`.

## 6. Chạy giao diện minh họa bằng Flask

Giao diện minh họa hiển thị kết quả thực tế được tạo ra từ hai chương trình, không tự tạo số liệu giả.

### Khởi chạy

Cài đặt các thư viện trong tệp yêu cầu:

```bash
pip install -r requirements.txt
```

Tạo kết quả phân nhóm khách hàng và ảnh với K = 3, 5, 8:

```bash
python scripts/customer_clustering.py
python scripts/image_segmentation.py --k 3
python scripts/image_segmentation.py --k 5
python scripts/image_segmentation.py --k 8
```

Khởi động ứng dụng:

```bash
python app.py
```

Mở trình duyệt tại địa chỉ `http://127.0.0.1:5000`.

### Các thành phần giao diện

- `app.py`: định nghĩa các đường dẫn truy cập và đọc kết quả từ thư mục đầu ra.
- `templates/index.html`: trang tổng quan kết quả phân nhóm khách hàng và phân đoạn ảnh.
- `static/css/style.css`: định dạng giao diện và khả năng hiển thị trên nhiều kích thước màn hình.
- `static/js/app.js`: xử lý tương tác trên trang.
- `/api/dashboard`: đường dẫn cung cấp dữ liệu tổng hợp, thống kê các nhóm và bảng so sánh thuật toán.

**Lưu ý:** thư mục `outputs/` được tạo khi chạy chương trình và không cần đưa lên Git. Cần chạy các chương trình xử lý trước khi mở giao diện để trang có dữ liệu và hình ảnh hiển thị.

## 7. Nguồn dữ liệu

Bộ dữ liệu Mall Customers gồm 200 dòng được lấy từ [geekquad/AlgoBook – Mall_Customers.csv](https://github.com/geekquad/AlgoBook/blob/35a2510f25faf077023b71c74236a498af53497d/ml/KMeans/Mall_Customers.csv). Các cột dữ liệu gồm mã khách hàng, giới tính, tuổi, thu nhập hằng năm và điểm chi tiêu.

## 8. Ghi chú khi báo cáo

- Phân cụm là phương pháp học không giám sát: dữ liệu không có sẵn nhãn mục tiêu để mô hình dự đoán.
- Chuẩn hóa dữ liệu quan trọng vì K-Means và các thuật toán dựa trên khoảng cách chịu ảnh hưởng của đơn vị đo.
- K-Means yêu cầu xác định trước số nhóm K; DBSCAN phân nhóm dựa trên mật độ và có thể đánh dấu điểm nhiễu; Agglomerative xây dựng các nhóm theo cấu trúc phân cấp.
- PCA chỉ giảm dữ liệu xuống hai chiều để trực quan hóa, không thay thế ba đặc trưng được dùng trong quá trình phân nhóm.
- Phân đoạn ảnh trong bài này chỉ nhóm điểm ảnh theo màu RGB, không nhận diện ý nghĩa của đối tượng trong ảnh.

## 9. Tài liệu trong thư mục docs

- [Mô tả bài thực hành](docs/PROJECT_DESCRIPTION.md)
- [Tài liệu bộ dữ liệu](docs/DATASET_DOCUMENTATION.md)
- [Hướng dẫn cài đặt và thực thi](docs/EXECUTION_GUIDE.md)
- [Báo cáo kết quả thực nghiệm](docs/RESULTS_REPORT.md)

## 10. Tài liệu tham khảo

- Scikit-learn – K-Means: https://scikit-learn.org/stable/modules/clustering.html#k-means
- Scikit-learn – DBSCAN: https://scikit-learn.org/stable/modules/clustering.html#dbscan
- Scikit-learn – Hệ số Silhouette: https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient
- Scikit-image – Dữ liệu ảnh mẫu: https://scikit-image.org/docs/stable/api/skimage.data.html
