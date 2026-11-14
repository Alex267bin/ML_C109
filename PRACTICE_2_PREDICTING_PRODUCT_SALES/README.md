# Bài thực hành 2 – Dự đoán doanh số sản phẩm

## 1. Tổng quan
Dự án xây dựng quy trình học máy để dự đoán doanh số hằng ngày của từng nhóm sản phẩm tại các cửa hàng bán lẻ, sử dụng bộ dữ liệu Corporación Favorita Store Sales.

- **Biến mục tiêu:** `sales` (doanh số dạng số).
- **Mức dự đoán:** ngày × cửa hàng × nhóm sản phẩm.

## 2. Bộ dữ liệu
Tải các tệp của cuộc thi [Kaggle – Dự báo doanh số cửa hàng theo chuỗi thời gian](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) và đặt vào thư mục `data/raw/`:

- `train.csv` (bắt buộc; có cột doanh số cần dự đoán)
- `stores.csv` (thông tin cửa hàng)
- `holidays_events.csv` (ngày lễ và sự kiện)
- `oil.csv` (giá dầu)
- `transactions.csv` (không bắt buộc; chỉ chứa thông tin lịch sử)

Tệp ngày lễ phải được đặt tên là `holidays_events.csv`.

**Giới hạn của bộ dữ liệu:** Mô tả bài tập có đề cập đến nhân khẩu học khách hàng, đánh giá, xếp hạng, giá trị vòng đời khách hàng và chi phí quảng cáo. Những trường này không có trong bộ dữ liệu Favorita được cung cấp. Dự án không tự tạo dữ liệu giả mà sử dụng các trường thực tế: ngày, cửa hàng, nhóm sản phẩm, số lượng khuyến mãi, thông tin cửa hàng, giá dầu và thông tin ngày lễ/sự kiện.

## 3. Cài đặt môi trường
Khuyến nghị Python 3.10 trở lên.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\\Scripts\\activate     # Windows
pip install -r requirements.txt
```

## 4. Chạy chương trình
Thực hiện tại thư mục Practice 2:

```bash
python -m scripts.run_pipeline --data-dir data/raw --output-dir outputs
```

Chương trình thực hiện các bước: đọc dữ liệu, tạo đặc trưng, chia tập kiểm định theo thứ tự thời gian, huấn luyện các mô hình Hồi quy tuyến tính, Cây quyết định, Rừng ngẫu nhiên và Tăng cường Gradient, đánh giá bằng MAE, MSE, RMSE và lưu mô hình.

## 5. Tài liệu mô tả thuật toán

Xem [Mô tả các thuật toán học máy](docs/MO_TA_CAC_THUAT_TOAN.md) để tìm hiểu mục tiêu bài toán, tiền xử lý dữ liệu, nguyên lý hoạt động, đầu vào/đầu ra, tham số, ưu nhược điểm của bốn thuật toán và cách đánh giá mô hình.

## 6. Kết quả đầu ra
- `outputs/metrics/model_comparison.csv`: bảng so sánh các mô hình.
- `outputs/metrics/best_model.json`: thông tin mô hình tốt nhất.
- `outputs/predictions/validation_predictions.csv`: kết quả dự đoán trên tập kiểm định.
- `models/saved_models/best_model.joblib`: mô hình tốt nhất đã lưu.

Các chỉ số được tạo khi chạy chương trình; tài liệu không điền sẵn hoặc tự tạo số liệu kết quả.

## 7. Lưu ý về phương pháp
- Tập kiểm định được tách theo trình tự thời gian, không chia ngẫu nhiên.
- Giai đoạn cuối của dòng thời gian trong tập huấn luyện được dùng làm tập kiểm định.
- Đặc trưng khuyến mãi và lịch được tạo từ những dữ liệu thực tế hiện có.
- Quy trình cơ sở chưa sử dụng đặc trưng doanh số trễ hoặc trung bình trượt vì chúng cần cách dự báo đệ quy cẩn thận; nếu tạo sai có thể làm rò rỉ thông tin tương lai.
- Dữ liệu giao dịch không được đưa vào bộ đặc trưng cơ sở vì tổng giao dịch trong tương lai có thể chưa biết tại thời điểm dự đoán.
- Đây là quy trình huấn luyện và minh họa chạy cục bộ, chưa phải dịch vụ triển khai thực tế.

## 8. Chạy giao diện minh họa
Sau khi huấn luyện và tạo tệp mô hình tốt nhất:

```bash
python app.py
```

Mở địa chỉ `http://127.0.0.1:5000` trên trình duyệt. Giao diện cho phép chọn cửa hàng, nhóm sản phẩm, ngày dự đoán và số mặt hàng khuyến mãi.
