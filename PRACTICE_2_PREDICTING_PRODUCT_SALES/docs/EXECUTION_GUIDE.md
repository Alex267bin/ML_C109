# Hướng dẫn chạy chương trình

Thực hiện các lệnh bên dưới từ thư mục `PRACTICE_2_PREDICTING_PRODUCT_SALES/`.

1. Cài đặt các thư viện cần thiết: `pip install -r requirements.txt`
2. Đặt các tệp CSV tải từ Kaggle vào thư mục `data/raw/`.
3. Chạy phân tích dữ liệu khám phá (EDA): `python -m scripts.run_eda`
4. Chạy quy trình huấn luyện và đánh giá mô hình: `python -m scripts.run_pipeline`
5. Chạy kiểm thử: `python -m pytest -q`

Kết quả EDA được lưu trong `outputs/eda/`; các chỉ số đánh giá mô hình và kết quả dự đoán được lưu lần lượt trong `outputs/metrics/` và `outputs/predictions/`.

Các tệp dữ liệu được loại trừ khỏi Git để tránh đưa những bộ dữ liệu dung lượng lớn của cuộc thi lên kho mã nguồn. Giữ nguyên các tệp dữ liệu thô ban đầu, không chỉnh sửa trực tiếp chúng.