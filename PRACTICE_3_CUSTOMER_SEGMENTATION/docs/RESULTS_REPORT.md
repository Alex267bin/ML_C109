# Báo cáo kết quả thực nghiệm – Bài thực hành 3

> Báo cáo này ghi lại kết quả của lần chạy đã được thực hiện trong quá trình phát triển. Khi chạy lại trên môi trường hoặc bộ dữ liệu khác, cần cập nhật số liệu theo các tệp kết quả mới.

## 1. Môi trường và dữ liệu thực nghiệm

- Nội dung thực nghiệm: phân nhóm khách hàng và phân đoạn ảnh bằng K-Means.
- Bộ dữ liệu khách hàng: Mall Customers.
- Số bản ghi được sử dụng trong lần chạy đã ghi nhận: **200**.
- Số đặc trưng dùng để phân nhóm: tuổi, thu nhập hằng năm và điểm chi tiêu.
- Phương pháp chuẩn hóa: `StandardScaler`.

## 2. Kết quả phân nhóm khách hàng

### 2.1. So sánh các mô hình

| Thuật toán và cấu hình | Hệ số Silhouette | Ghi chú |
|---|---:|---|
| K-Means, K = 6 | 0.4284 | Cao nhất trong các cấu hình K-Means đã thử |
| DBSCAN, eps = 0.5, min_samples = 5 | 0.4817 | Chỉ tính trên các điểm không bị đánh dấu là nhiễu |

Trong lần chạy được ghi nhận:

- K-Means chọn số nhóm K = 6.
- DBSCAN tạo được 6 nhóm trên 140 điểm không bị đánh dấu là nhiễu.
- DBSCAN đánh dấu 60 trên tổng số 200 điểm là nhiễu, tương đương 30%.
- Hệ số Silhouette 0.4817 của DBSCAN chỉ được tính trên 140 điểm được giữ lại để đánh giá.

**Nhận xét:** DBSCAN có hệ số Silhouette cao hơn trong lần chạy này, nhưng 30% số điểm đã bị loại khỏi phép tính Silhouette do được xem là nhiễu. Vì vậy, không thể chỉ dựa vào hai giá trị Silhouette để kết luận DBSCAN tốt hơn K-Means. K-Means phân nhóm toàn bộ 200 khách hàng nên thuận tiện hơn khi mô tả toàn bộ tập dữ liệu trong báo cáo này.

### 2.2. Đặc điểm của 6 nhóm khách hàng theo K-Means

| Nhóm | Tuổi trung bình | Thu nhập trung bình (nghìn USD) | Điểm chi tiêu trung bình | Số khách hàng |
|---:|---:|---:|---:|---:|
| 0 | 56.33 | 54.27 | 49.07 | 45 |
| 1 | 26.79 | 57.10 | 48.13 | 39 |
| 2 | 41.94 | 88.94 | 16.97 | 33 |
| 3 | 32.69 | 86.54 | 82.13 | 39 |
| 4 | 25.00 | 25.26 | 77.61 | 23 |
| 5 | 45.52 | 26.29 | 19.38 | 21 |

### 2.3. Nhận xét từng nhóm khách hàng

- **Nhóm 0:** khách hàng có độ tuổi trung bình cao hơn, thu nhập ở mức trung bình và điểm chi tiêu ở mức trung bình.
- **Nhóm 1:** khách hàng trẻ, thu nhập trung bình và điểm chi tiêu trung bình.
- **Nhóm 2:** khách hàng có thu nhập cao nhưng điểm chi tiêu thấp.
- **Nhóm 3:** khách hàng có thu nhập cao và điểm chi tiêu cao.
- **Nhóm 4:** khách hàng trẻ, thu nhập thấp nhưng điểm chi tiêu cao.
- **Nhóm 5:** khách hàng ở độ tuổi trung niên, thu nhập thấp và điểm chi tiêu thấp.

Các nhận xét trên được rút ra từ giá trị trung bình của từng nhóm, không phải kết luận về hành vi của từng khách hàng riêng lẻ.

## 3. Kết quả phân đoạn ảnh

Chương trình đã được chạy với K = 3 và K = 8 trong một lần thử; một lần chạy khác đã tạo ảnh so sánh với K = 5.

Các tệp kết quả được tạo theo quy tắc:

- `comparison_k3.png`, `comparison_k5.png`, `comparison_k8.png`: ảnh gốc và ảnh sau phân đoạn được đặt cạnh nhau.
- `segmented_k3.png`, `segmented_k5.png`, `segmented_k8.png`: ảnh sau khi phân đoạn.

### Nhận xét

- Khi K nhỏ, ảnh được biểu diễn bằng ít màu đại diện hơn và thường bị đơn giản hóa nhiều hơn.
- Khi K lớn, ảnh có nhiều màu đại diện hơn và thường giữ lại được nhiều sắc thái màu hơn.
- Tăng K không đồng nghĩa chắc chắn ảnh sẽ tốt hơn. Cần quan sát ảnh đầu ra và lựa chọn số nhóm phù hợp với mục tiêu thực hiện.

## 4. Kết quả kiểm thử

Kết quả kiểm thử đã được ghi nhận ở lần chạy trước:

```text
2 passed in 4.11s
```

Trước khi nộp bài, cần chạy lại lệnh `pytest -q` trên phiên bản mã nguồn cuối cùng và cập nhật kết quả thực tế tại đây.

## 5. Kết luận

Bài thực hành 3 đã triển khai quy trình phân nhóm khách hàng bằng K-Means, DBSCAN và phân cụm phân cấp Agglomerative; sử dụng hệ số Silhouette để đánh giá và PCA để trực quan hóa các nhóm. Ở phần phân đoạn ảnh, thuật toán K-Means được áp dụng trên các véc-tơ màu RGB của điểm ảnh nhằm tạo ra ảnh có số lượng màu đại diện được giới hạn bởi K.

Trên bộ dữ liệu 200 khách hàng trong lần chạy đã ghi nhận, K-Means với K = 6 có hệ số Silhouette cao nhất trong các cấu hình K-Means đã thử. DBSCAN đạt hệ số Silhouette cao hơn trên tập điểm không nhiễu, tuy nhiên có 30% số điểm bị loại khỏi phép tính. Vì vậy, cần thận trọng khi so sánh và diễn giải kết quả của hai thuật toán.

## 6. Các tệp dùng để đối chiếu kết quả

Sau khi chạy lại chương trình, đối chiếu báo cáo với các tệp:

- `outputs/metrics/run_summary.json`
- `outputs/metrics/clustering_comparison.csv`
- `outputs/metrics/cluster_profiles.csv`
- `outputs/metrics/data_quality.csv`
- `outputs/segmented_images/`
