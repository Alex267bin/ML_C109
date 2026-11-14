# Hướng dẫn thực thi Practice 3

## 1. Môi trường

Chạy các lệnh tại thư mục:

```bash
cd PRACTICE_3_CUSTOMER_SEGMENTATION
```

Tạo và kích hoạt môi trường ảo:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

## 2. Kiểm tra cấu trúc dữ liệu

Đảm bảo file dữ liệu tồn tại:

```bash
ls data/raw/
```

Dataset mặc định là `data/raw/Mall_Customers.csv`. Ba cột số bắt buộc là `Age`, `Annual Income (k$)` và `Spending Score (1-100)`.

## 3. Chạy Customer Segmentation

```bash
python scripts/customer_clustering.py
```

Có thể chỉ định dataset và thư mục output:

```bash
python scripts/customer_clustering.py --data data/raw/Mall_Customers.csv --output outputs
```

Sau khi chạy, kiểm tra:
- `outputs/metrics/data_quality.csv`
- `outputs/metrics/clustering_comparison.csv`
- `outputs/metrics/cluster_profiles.csv`
- `outputs/metrics/run_summary.json`
- `outputs/customer_segments.csv`
- `outputs/figures/`

## 4. Chạy Image Segmentation

Chạy ảnh mẫu với K mặc định:

```bash
python scripts/image_segmentation.py
```

Thử nhiều giá trị K:

```bash
python scripts/image_segmentation.py --k 3
python scripts/image_segmentation.py --k 5
python scripts/image_segmentation.py --k 8
```

Dùng ảnh riêng:

```bash
python scripts/image_segmentation.py --image data/raw/my_image.jpg --k 5
```

Kết quả nằm trong `outputs/segmented_images/`.

## 5. Chạy kiểm thử

```bash
pytest -q
```

Kết quả kiểm thử phải được ghi nhận theo lần chạy thực tế; không dùng kết quả cũ để khẳng định lần chạy hiện tại đã thành công.

## 6. Chạy Flask dashboard

Trước tiên cần chạy hai script để sinh dữ liệu và hình ảnh đầu ra. Sau đó:

```bash
python app.py
```

Mở địa chỉ:

```text
http://127.0.0.1:5000
```

Trong GitHub Codespaces, mở tab Ports và truy cập cổng 5000 được chuyển tiếp.

## 7. Thứ tự chạy khuyến nghị

```bash
pip install -r requirements.txt
pytest -q
python scripts/customer_clustering.py
python scripts/image_segmentation.py --k 3
python scripts/image_segmentation.py --k 5
python scripts/image_segmentation.py --k 8
python app.py
```

**Lưu ý:** thư mục `outputs/` được tạo trong quá trình chạy và có thể bị loại khỏi Git theo `.gitignore`. Nếu dashboard chưa hiển thị biểu đồ, hãy chạy các script tạo output trước.
