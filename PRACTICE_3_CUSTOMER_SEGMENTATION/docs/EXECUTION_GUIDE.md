# Hướng dẫn cài đặt và thực thi bài thực hành 3

## 1. Chuẩn bị môi trường

Mở thiết bị đầu cuối và di chuyển vào thư mục:

```bash
cd PRACTICE_3_CUSTOMER_SEGMENTATION
```

Tạo môi trường Python ảo:

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo:

```bash
source .venv/bin/activate
```

Nếu sử dụng Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## 2. Kiểm tra dữ liệu đầu vào

Kiểm tra các tệp dữ liệu hiện có:

```bash
ls data/raw/
```

Tệp dữ liệu mặc định là `data/raw/Mall_Customers.csv`. Dữ liệu phải có ba cột số bắt buộc: `Age`, `Annual Income (k$)` và `Spending Score (1-100)`.

## 3. Thực thi phần phân nhóm khách hàng

Chạy chương trình bằng lệnh:

```bash
python scripts/customer_clustering.py
```

Có thể chỉ định rõ tệp dữ liệu đầu vào và thư mục lưu kết quả:

```bash
python scripts/customer_clustering.py --data data/raw/Mall_Customers.csv --output outputs
```

Sau khi chương trình chạy xong, kiểm tra các kết quả:

- `outputs/metrics/data_quality.csv`: thống kê dữ liệu thiếu và giá trị ngoại lệ.
- `outputs/metrics/clustering_comparison.csv`: bảng so sánh các cấu hình phân cụm.
- `outputs/metrics/cluster_profiles.csv`: thống kê đặc điểm trung bình của từng nhóm.
- `outputs/metrics/run_summary.json`: kết quả tổng hợp của lần chạy.
- `outputs/customer_segments.csv`: dữ liệu khách hàng kèm nhãn nhóm.
- `outputs/figures/`: các biểu đồ phân tích và trực quan hóa.

## 4. Thực thi phần phân đoạn ảnh

Chạy ảnh mẫu với số nhóm mặc định:

```bash
python scripts/image_segmentation.py
```

Thử nghiệm với các số nhóm khác nhau:

```bash
python scripts/image_segmentation.py --k 3
python scripts/image_segmentation.py --k 5
python scripts/image_segmentation.py --k 8
```

Sử dụng ảnh riêng:

```bash
python scripts/image_segmentation.py --image data/raw/my_image.jpg --k 5
```

Kết quả được lưu trong thư mục `outputs/segmented_images/`.

## 5. Chạy kiểm thử chương trình

Thực hiện lệnh:

```bash
pytest -q
```

Cần ghi nhận kết quả kiểm thử của lần chạy thực tế trên phiên bản mã nguồn cuối cùng. Không sử dụng kết quả của lần chạy cũ để khẳng định phiên bản hiện tại đã kiểm thử thành công.

## 6. Khởi chạy giao diện minh họa bằng Flask

Trước khi mở giao diện, cần chạy các chương trình phân nhóm khách hàng và phân đoạn ảnh để tạo dữ liệu, biểu đồ và ảnh kết quả.

Khởi chạy ứng dụng:

```bash
python app.py
```

Mở trình duyệt tại địa chỉ:

```text
http://127.0.0.1:5000
```

Nếu sử dụng GitHub Codespaces, mở mục **Ports (Cổng)** và truy cập cổng 5000 đã được chuyển tiếp.

## 7. Thứ tự thực hiện được đề xuất

```bash
pip install -r requirements.txt
pytest -q
python scripts/customer_clustering.py
python scripts/image_segmentation.py --k 3
python scripts/image_segmentation.py --k 5
python scripts/image_segmentation.py --k 8
python app.py
```

**Lưu ý:** thư mục `outputs/` được tạo ra trong quá trình chạy chương trình và có thể được loại khỏi Git theo cấu hình trong `.gitignore`. Nếu giao diện chưa hiển thị biểu đồ hoặc ảnh, hãy chạy các chương trình tạo kết quả trước.
