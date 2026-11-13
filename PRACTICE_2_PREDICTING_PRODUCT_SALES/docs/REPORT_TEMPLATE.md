# Bài thực hành 2 – Dự đoán doanh số bán sản phẩm: Báo cáo

> Chỉ hoàn thành báo cáo này sau khi đã chạy quy trình xử lý dữ liệu và huấn luyện mô hình. Không tự tạo hoặc suy đoán các chỉ số, kết quả hay nhận xét.

## 1. Xác định bài toán

Mô tả bài toán dự đoán doanh số bán lẻ, biến mục tiêu (`sales`) và mức độ chi tiết của mỗi dự đoán (ngày, cửa hàng và nhóm sản phẩm).

## 2. Mô tả bộ dữ liệu

Liệt kê từng tệp CSV được sử dụng, số hàng/cột, khoảng thời gian, biến mục tiêu và các đặc trưng đầu vào hiện có. Giải thích những khái niệm của đề bài không thể khai thác từ bộ dữ liệu, theo nội dung trong `DATASET_NOTES.md`.

## 3. Phân tích dữ liệu khám phá (EDA)

Chèn các biểu đồ được tạo trong `outputs/eda/`. Dựa trên kết quả thực tế để thảo luận về phân phối biến mục tiêu, xu hướng doanh số theo tháng, doanh số theo nhóm sản phẩm, giá trị bị thiếu và các bản ghi trùng lặp.

## 4. Tiền xử lý dữ liệu

Giải thích cách chuyển đổi dữ liệu ngày tháng, kết hợp các bảng dữ liệu, xử lý giá trị bị thiếu, mã hóa biến phân loại và chia tập kiểm định theo thứ tự thời gian.

## 5. Xây dựng đặc trưng (Feature Engineering)

Giải thích các đặc trưng lịch, số lượng chương trình khuyến mãi, thuộc tính cửa hàng, giá dầu và biến chỉ báo ngày lễ. Nêu rõ lý do không tự tạo dữ liệu về khách hàng, đánh giá sản phẩm hoặc hoạt động tiếp thị khi bộ dữ liệu không cung cấp những thông tin này.

## 6. Các mô hình

Mô tả Linear Regression (Hồi quy tuyến tính), Decision Tree Regressor (Cây quyết định hồi quy) và Random Forest Regressor (Rừng ngẫu nhiên hồi quy), bao gồm các siêu tham số đã lựa chọn.

## 7. Đánh giá mô hình

Chèn các giá trị MAE, MSE và RMSE thực tế từ tệp `outputs/metrics/model_comparison.csv`. Xác định mô hình tốt nhất dựa trên RMSE của tập kiểm định.

## 8. Thảo luận và hạn chế

Thảo luận về phương pháp kiểm định theo thời gian, khả năng phân phối dữ liệu thay đổi, việc thiếu dữ liệu ở cấp độ khách hàng và giới hạn của quy trình hiện tại: đây là thử nghiệm chạy cục bộ, chưa phải hệ thống triển khai trong môi trường thực tế.

## 9. Kết luận và hướng phát triển

Tóm tắt các kết quả đã đo lường được và đề xuất những hướng phát triển tiếp theo như xây dựng đặc trưng trễ (lag features) an toàn, không gây rò rỉ dữ liệu; dự báo đệ quy/nhiều bước; tìm kiếm siêu tham số và giám sát mô hình.