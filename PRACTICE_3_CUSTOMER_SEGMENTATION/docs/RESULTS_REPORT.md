# Báo cáo kết quả thực nghiệm – Practice 3

> Báo cáo này ghi lại kết quả của lần chạy đã được thực hiện trong quá trình phát triển. Khi chạy lại trên môi trường hoặc dữ liệu khác, hãy cập nhật số liệu từ các tệp output mới.

## 1. Môi trường và dữ liệu

- Bài toán: Customer Segmentation và K-Means Image Segmentation.
- Dataset khách hàng: Mall Customers.
- Số bản ghi được sử dụng trong lần chạy được ghi nhận: **200**.
- Số đặc trưng dùng để phân cụm: Age, Annual Income (k$), Spending Score (1-100).
- Phương pháp chuẩn hóa: StandardScaler.

## 2. Kết quả Customer Segmentation

### 2.1. So sánh mô hình

| Thuật toán / cấu hình | Silhouette | Ghi chú |
|---|---:|---|
| K-Means, K = 6 | 0.4284 | Tốt nhất trong các cấu hình K-Means đã thử |
| DBSCAN, eps = 0.5, min_samples = 5 | 0.4817 | Tính trên các điểm không phải noise |

Trong lần chạy được ghi nhận:
- K-Means chọn K = 6.
- DBSCAN tạo 6 cụm trên 140 điểm không phải noise.
- DBSCAN đánh dấu 60/200 điểm là noise, tương đương 30%.
- Silhouette 0.4817 của DBSCAN chỉ được tính trên 140 điểm được giữ lại để đánh giá.

**Diễn giải:** DBSCAN có silhouette cao hơn trong lần chạy này, nhưng đã loại 30% dữ liệu khỏi phép tính silhouette. Vì vậy không thể chỉ dựa vào hai con số để kết luận DBSCAN tốt hơn K-Means. K-Means phân cụm toàn bộ 200 khách hàng và phù hợp để mô tả toàn bộ tập dữ liệu trong báo cáo này.

### 2.2. Hồ sơ 6 cụm K-Means

| Cluster | Tuổi trung bình | Thu nhập trung bình (k$) | Spending Score trung bình | Số khách hàng |
|---:|---:|---:|---:|---:|
| 0 | 56.33 | 54.27 | 49.07 | 45 |
| 1 | 26.79 | 57.10 | 48.13 | 39 |
| 2 | 41.94 | 88.94 | 16.97 | 33 |
| 3 | 32.69 | 86.54 | 82.13 | 39 |
| 4 | 25.00 | 25.26 | 77.61 | 23 |
| 5 | 45.52 | 26.29 | 19.38 | 21 |

### 2.3. Nhận xét các nhóm khách hàng

- **Cluster 0:** nhóm khách hàng lớn tuổi hơn, thu nhập trung bình và điểm chi tiêu ở mức trung bình.
- **Cluster 1:** nhóm khách hàng trẻ, thu nhập trung bình và điểm chi tiêu trung bình.
- **Cluster 2:** nhóm có thu nhập cao nhưng điểm chi tiêu thấp.
- **Cluster 3:** nhóm có thu nhập cao và điểm chi tiêu cao.
- **Cluster 4:** nhóm trẻ, thu nhập thấp nhưng điểm chi tiêu cao.
- **Cluster 5:** nhóm tuổi trung niên hơn, thu nhập thấp và điểm chi tiêu thấp.

Đây là diễn giải dựa trên giá trị trung bình của từng cụm, không phải kết luận về hành vi của từng cá nhân.

## 3. Kết quả Image Segmentation

Chương trình đã được chạy với các giá trị K = 3 và K = 8 trong một lần thử, và lần chạy sau tạo ảnh so sánh với K = 5.

Các tệp đầu ra được tạo theo quy tắc:
- `comparison_k3.png`, `comparison_k5.png`, `comparison_k8.png`: ảnh gốc và ảnh sau phân đoạn.
- `segmented_k3.png`, `segmented_k5.png`, `segmented_k8.png`: ảnh sau phân đoạn.

Nhận xét định tính:
- K nhỏ tạo ít màu đại diện hơn, ảnh thường đơn giản hóa mạnh hơn.
- K lớn tạo nhiều màu đại diện hơn và thường giữ lại nhiều sắc thái màu hơn.
- Tăng K không đồng nghĩa chắc chắn ảnh tốt hơn; cần quan sát ảnh kết quả và cân nhắc mục tiêu phân đoạn.

## 4. Kiểm thử

Lần kiểm thử được ghi nhận trước đó:

```text
2 passed in 4.11s
```

Khi nộp bài, nên chạy lại `pytest -q` trên phiên bản code cuối cùng và cập nhật kết quả thực tế tại đây.

## 5. Kết luận

Practice 3 đã triển khai quy trình phân cụm khách hàng bằng K-Means, DBSCAN và Agglomerative Clustering; đánh giá bằng Silhouette Coefficient; đồng thời trực quan hóa các cụm bằng PCA. Phần Image Segmentation áp dụng K-Means trên các vector màu RGB của pixel để tạo ảnh với số màu đại diện được giới hạn bởi K.

Kết quả cho thấy K-Means với K = 6 là cấu hình có silhouette cao nhất trong nhóm K-Means đã thử trên bộ dữ liệu 200 khách hàng. DBSCAN đạt silhouette cao hơn trên tập điểm không nhiễu, nhưng có 30% điểm bị loại khỏi phép tính nên cần diễn giải thận trọng.

## 6. Tệp dùng để đối chiếu số liệu

Sau khi chạy lại chương trình, đối chiếu báo cáo với:
- `outputs/metrics/run_summary.json`
- `outputs/metrics/clustering_comparison.csv`
- `outputs/metrics/cluster_profiles.csv`
- `outputs/metrics/data_quality.csv`
- `outputs/segmented_images/`
