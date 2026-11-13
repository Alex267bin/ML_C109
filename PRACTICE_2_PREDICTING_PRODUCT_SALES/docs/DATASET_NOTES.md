# Kiểm tra bộ dữ liệu và các đặc trưng hiện có

Dự án sử dụng bộ dữ liệu chuỗi thời gian bán lẻ Corporación Favorita. Bảng dữ liệu huấn luyện chính chứa doanh số hằng ngày theo cửa hàng và nhóm sản phẩm, bao gồm biến mục tiêu `sales` và số lượng mặt hàng được khuyến mãi `onpromotion`.

| Nội dung trong yêu cầu bài tập | Có trong dữ liệu Favorita không? | Cách dự án xử lý |
|---|---|---|
| Sản phẩm / danh mục sản phẩm | Có (`family`) | Được sử dụng |
| Cửa hàng / địa điểm | Có | Được sử dụng |
| Ngày / tính mùa vụ | Có | Tạo các đặc trưng lịch |
| Khuyến mãi | Có (`onpromotion`) | Được sử dụng |
| Ngày lễ / sự kiện | Có | Tổng hợp thành biến đánh dấu ngày lễ theo ngày |
| Giá dầu | Có | Được sử dụng và điền giá trị thiếu bằng trung vị |
| Nhân khẩu học khách hàng | Không | Không tự tạo dữ liệu giả |
| Đánh giá / xếp hạng | Không | Không tự tạo dữ liệu giả |
| Giá trị vòng đời khách hàng | Không | Không tự tạo dữ liệu giả |
| Chi phí quảng cáo / ngân sách chiến dịch | Không | Không tự tạo dữ liệu giả |

Quy trình cơ sở hiện tại không sử dụng tổng số giao dịch hoặc các đặc trưng doanh số trễ (lag). Những dữ liệu này cần được xác định rõ có sẵn tại thời điểm dự báo hay không để tránh rò rỉ thông tin mục tiêu (target leakage).
