# Tài liệu bộ dữ liệu – Phân nhóm khách hàng tại trung tâm thương mại

## 1. Tổng quan bộ dữ liệu

Bài thực hành sử dụng bộ dữ liệu **Mall Customer Segmentation** trong tệp `Mall_Customers.csv`.

- Số bản ghi: **200 khách hàng**
- Số cột: **5**
- Định dạng: CSV
- Loại dữ liệu: thông tin nhân khẩu học và mức chi tiêu của khách hàng
- Mục đích: phân nhóm khách hàng bằng phương pháp học không giám sát.

> **Thông tin về nguồn dữ liệu:** Tệp CSV được người học cung cấp cho thư mục dự án. Cấu trúc của tệp tương đồng với bộ dữ liệu Mall Customer Segmentation thường được chia sẻ. Tuy nhiên, tệp được cung cấp không kèm trang Kaggle gốc, tên chủ sở hữu, phiên bản hoặc giấy phép sử dụng. Vì vậy, tài liệu này không khẳng định đường dẫn Kaggle hay giấy phép đã được xác minh. Nếu có đường dẫn Kaggle chính xác, hãy bổ sung tại đây.

## 2. Từ điển dữ liệu

| Tên cột | Ý nghĩa | Kiểu dữ liệu | Cách sử dụng |
|---|---|---|---|
| `CustomerID` | Mã định danh duy nhất của khách hàng | Số nguyên | Chỉ dùng để nhận diện, không dùng để phân cụm |
| `Genre` | Giới tính khách hàng (Nam/Nữ) | Phân loại | Có thể dùng để mô tả kết quả, không dùng để tính khoảng cách |
| `Age` | Tuổi khách hàng, tính theo năm | Số nguyên | Đặc trưng phân cụm |
| `Annual_Income_(k$)` | Thu nhập hằng năm, tính theo nghìn đô la Mỹ | Số nguyên | Đặc trưng phân cụm |
| `Spending_Score` | Điểm chi tiêu trên thang từ 1 đến 100 | Số nguyên | Đặc trưng phân cụm |

Ba đặc trưng được sử dụng để phân cụm là tuổi, thu nhập hằng năm và điểm chi tiêu. Không sử dụng `CustomerID` vì đây chỉ là mã định danh, không mang ý nghĩa định lượng cho bài toán.

## 3. Kiểm tra và xác thực dữ liệu

Tệp dữ liệu được kiểm tra trước khi đưa vào phân tích.

| Nội dung kiểm tra | Kết quả |
|---|---:|
| Số dòng | 200 |
| Số cột | 5 |
| Giá trị bị thiếu | 0 |
| Dòng trùng lặp | 0 |
| Mã khách hàng trùng lặp | 0 |
| Khoảng tuổi | 18–70 |
| Khoảng thu nhập hằng năm | 15–137 (nghìn USD) |
| Khoảng điểm chi tiêu | 1–99 |

### Thống kê mô tả

| Đặc trưng | Giá trị trung bình | Độ lệch chuẩn | Nhỏ nhất | Trung vị | Lớn nhất |
|---|---:|---:|---:|---:|---:|
| Tuổi | 38.85 | 13.97 | 18 | 36 | 70 |
| Thu nhập hằng năm (nghìn USD) | 60.56 | 26.26 | 15 | 61.5 | 137 |
| Điểm chi tiêu | 50.20 | 25.82 | 1 | 50 | 99 |

Các giá trị trên mô tả tệp CSV được cung cấp, không phải chỉ số đánh giá mô hình.

## 4. Tiền xử lý và lựa chọn đặc trưng

1. Đọc dữ liệu CSV bằng thư viện Pandas.
2. Chuẩn hóa tên cột trong chương trình khi cần thiết; ví dụ, cột `Genre` có thể được đổi tên nội bộ thành `Gender`.
3. Kiểm tra giá trị thiếu và khảo sát giá trị ngoại lệ bằng phương pháp khoảng tứ phân vị (IQR).
4. Chọn ba cột `Age`, `Annual_Income_(k$)` và `Spending_Score` làm ma trận đặc trưng.
5. Sử dụng `StandardScaler` để chuẩn hóa dữ liệu trước khi áp dụng các thuật toán dựa trên khoảng cách.
6. Không đưa `CustomerID` và `Genre` vào ma trận đặc trưng phân cụm. Hai cột này chỉ phục vụ nhận diện bản ghi hoặc mô tả cụm sau khi phân nhóm.

## 5. Cách sử dụng tệp dữ liệu

Đặt tệp tại đường dẫn:

```text
data/raw/Mall_Customers.csv
```

Chạy chương trình phân cụm từ thư mục Practice 3:

```bash
python scripts/customer_clustering.py --data data/raw/Mall_Customers.csv --output outputs
```

## 6. Giới hạn của bộ dữ liệu

- Bộ dữ liệu chỉ có 200 bản ghi, vì vậy không nên khái quát kết quả cho toàn bộ khách hàng của mọi trung tâm thương mại.
- Tài liệu đi kèm không xác định cách thức và thời điểm thu thập dữ liệu.
- Ý nghĩa và cách xây dựng chỉ số Spending Score không được giải thích thêm trong tệp CSV.
- Tệp được cung cấp không có đường dẫn nguồn Kaggle hoặc giấy phép tái sử dụng đã được xác minh. Nếu môn học yêu cầu trích dẫn nguồn chính thức, cần bổ sung trang dữ liệu gốc và tuân thủ giấy phép tương ứng.
