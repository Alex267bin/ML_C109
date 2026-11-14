# Practice Exercise 3 – Customer Segmentation & K-Means Image Segmentation

## 1. Mục tiêu

Bài thực hành gồm **hai phần độc lập**:

1. **Customer Segmentation:** khám phá dữ liệu khách hàng, chuẩn hóa đặc trưng, phân cụm và đánh giá chất lượng cụm.
2. **K-Means Image Segmentation (additional):** gom các pixel theo màu bằng K-Means và so sánh ảnh gốc với ảnh sau phân đoạn.

## 2. Cấu trúc

- \`data/raw/Mall_Customers.csv\`: dữ liệu mẫu nhỏ để kiểm tra pipeline. Có thể thay bằng bộ Mall Customers đầy đủ.
- \`scripts/customer_clustering.py\`: EDA, scaling, thử K-Means/DBSCAN/Agglomerative, silhouette, hồ sơ cụm.
- \`scripts/image_segmentation.py\`: phân đoạn ảnh bằng K-Means trong không gian màu RGB.
- \`outputs/metrics/\`: báo cáo chất lượng dữ liệu, so sánh thuật toán và hồ sơ cụm.
- \`outputs/figures/\`: biểu đồ EDA, silhouette và PCA.
- \`outputs/segmented_images/\`: ảnh phân đoạn và ảnh so sánh.

## 3. Cài đặt

Chạy tại thư mục \`PRACTICE_3_CUSTOMER_SEGMENTATION\`:

\`\`\`bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
\`\`\`

## 4. Phần 1 – Customer Segmentation

### Dataset

Tài liệu mô tả dataset và data dictionary nằm tại `docs/DATASET_DOCUMENTATION.md`. File `data/raw/Mall_Customers.csv` gồm 200 dòng và 5 cột, đã được kiểm tra không có missing values hoặc duplicate rows.

Dữ liệu cần có ba cột số:
- \`Age\`
- \`Annual Income (k$)\`
- \`Spending Score (1-100)\`

Có thể có thêm cột \`Gender\` để mô tả đặc điểm của từng cụm. Gender không được đưa vào khoảng cách phân cụm trong bài này, đúng theo danh sách đặc trưng cần chuẩn hóa của đề.

Tệp CSV mẫu trong repository chỉ có 50 dòng để chạy thử. Để có kết quả phân tích đầy đủ hơn, thay bằng bộ Mall Customers đầy đủ và truyền đường dẫn bằng \`--data\`. Không nên trình bày kết quả từ dữ liệu mẫu như kết quả trên toàn bộ bộ dữ liệu.

### Chạy

\`\`\`bash
python scripts/customer_clustering.py
# Hoặc chỉ định dataset:
python scripts/customer_clustering.py --data data/raw/Mall_Customers.csv --output outputs
\`\`\`

### Quy trình

1. Kiểm tra missing values và outliers theo IQR.
2. Vẽ histogram và scatter plot để quan sát phân phối và quan hệ giữa các đặc trưng.
3. Dùng StandardScaler để đưa Age, Annual Income và Spending Score về thang đo tương đương.
4. Thử K-Means với K từ 2 đến 10; thử DBSCAN với nhiều eps; thử Agglomerative với K từ 2 đến 5.
5. So sánh Silhouette Coefficient. Với DBSCAN, điểm silhouette được tính trên các điểm không bị gán nhãn noise. Bảng so sánh đồng thời ghi `noise_points`, `noise_ratio` và `evaluated_points` để thể hiện rõ mức độ bao phủ của kết quả.
6. Chọn K-Means có silhouette cao nhất trong nhóm K-Means làm mô hình phân cụm cuối cùng; trực quan hóa bằng PCA 2D và lập hồ sơ cụm.

**Lưu ý đánh giá:** Silhouette càng gần 1 thường biểu thị cụm càng tách biệt; gần 0 biểu thị các cụm chồng lấn; giá trị âm có thể cho thấy điểm được gán vào cụm chưa phù hợp. Silhouette chỉ được tính khi có ít nhất hai cụm hợp lệ.

### Kết quả

- \`outputs/metrics/data_quality.csv\`
- \`outputs/metrics/clustering_comparison.csv\`
- \`outputs/metrics/cluster_profiles.csv\`
- \`outputs/metrics/gender_distribution_by_cluster.csv\` (nếu có Gender)
- \`outputs/metrics/run_summary.json\`
- \`outputs/customer_segments.csv\`
- \`outputs/figures/feature_distributions.png\`
- \`outputs/figures/eda_scatterplots.png\`
- \`outputs/figures/customer_clusters_pca.png\`
- \`outputs/figures/silhouette_by_k.png\`

## 5. Phần 2 – K-Means Image Segmentation

Đề cho phép sử dụng ảnh bất kỳ. Nếu không truyền ảnh, chương trình dùng ảnh chân dung mẫu có sẵn trong scikit-image để minh họa.

### Chạy

\`\`\`bash
# Dùng ảnh mẫu tích hợp:
python scripts/image_segmentation.py

# Dùng ảnh của bạn:
python scripts/image_segmentation.py --image data/raw/my_image.jpg --k 5

# Thử số cụm khác:
python scripts/image_segmentation.py --image data/raw/my_image.jpg --k 8 --max-size 300
\`\`\`

### Quy trình

1. Đọc ảnh bằng Pillow và chuyển về RGB.
2. Resize ảnh để giảm thời gian tính toán.
3. Flatten ảnh thành ma trận \`(số_pixel, 3)\`; mỗi pixel là vector màu RGB.
4. Dùng K-Means với K centroid; mỗi pixel được gán vào centroid gần nhất theo Euclidean distance.
5. Thay màu pixel bằng màu centroid, reshape về kích thước ảnh và lưu ảnh kết quả.
6. Lưu ảnh gốc và ảnh phân đoạn cạnh nhau để so sánh.

Đầu ra nằm trong \`outputs/segmented_images/\`.


## 6. Demo giao diện Flask (theo format Practice 2)

Demo hiển thị kết quả thực tế được sinh ra từ hai script, không tự tạo số liệu minh họa.

### Chạy demo

Cài thêm Flask cùng các thư viện còn lại:

```bash
pip install -r requirements.txt
```

Sinh kết quả phân cụm khách hàng và ảnh với K = 3, 5, 8:

```bash
python scripts/customer_clustering.py
python scripts/image_segmentation.py --k 3
python scripts/image_segmentation.py --k 5
python scripts/image_segmentation.py --k 8
```

Khởi động website:

```bash
python app.py
```

Mở trình duyệt tại `http://127.0.0.1:5000`.

### Thành phần demo

- `app.py`: Flask routes và đọc kết quả từ thư mục outputs.
- `templates/index.html`: dashboard Customer Segmentation và Image Segmentation.
- `static/css/style.css`: giao diện responsive theo phong cách Practice 2.
- `static/js/app.js`: xử lý tương tác phía trình duyệt.
- `/api/dashboard`: API trả về summary, cluster profiles và bảng so sánh thuật toán.

Lưu ý: thư mục `outputs/` được tạo khi chạy các script và không cần commit lên Git. Hãy chạy pipeline trước khi mở demo để website có dữ liệu và hình ảnh hiển thị.

## 6. Nguồn dữ liệu

Bộ dữ liệu Mall Customers 200 dòng được lấy từ [geekquad/AlgoBook – Mall_Customers.csv](https://github.com/geekquad/AlgoBook/blob/35a2510f25faf077023b71c74236a498af53497d/ml/KMeans/Mall_Customers.csv). Dữ liệu gồm CustomerID, Genre, Age, Annual Income và Spending Score.

## 7. Ghi chú khi báo cáo

- Clustering là học không giám sát: không có nhãn mục tiêu được cung cấp trước.
- Scaling quan trọng vì K-Means và các thuật toán dựa trên khoảng cách nhạy với đơn vị đo.
- K-Means cần chọn trước K; DBSCAN dùng mật độ và có thể đánh dấu điểm nhiễu; Agglomerative xây dựng cụm theo cách phân cấp.
- PCA chỉ giảm dữ liệu xuống hai chiều để trực quan hóa, không thay thế dữ liệu ba chiều dùng để huấn luyện clustering.
- Image segmentation trong bài này phân cụm theo màu RGB, không hiểu ngữ nghĩa đối tượng trong ảnh.

## 8. Tài liệu tham khảo

- Scikit-learn – K-Means: https://scikit-learn.org/stable/modules/clustering.html#k-means
- Scikit-learn – DBSCAN: https://scikit-learn.org/stable/modules/clustering.html#dbscan
- Scikit-learn – Silhouette Coefficient: https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient
- Scikit-image – Sample data: https://scikit-image.org/docs/stable/api/skimage.data.html


## Tài liệu trong thư mục docs

- [Mô tả bài thực hành](docs/PROJECT_DESCRIPTION.md)
- [Tài liệu dataset](docs/DATASET_DOCUMENTATION.md)
- [Hướng dẫn thực thi](docs/EXECUTION_GUIDE.md)
- [Báo cáo kết quả](docs/RESULTS_REPORT.md)
