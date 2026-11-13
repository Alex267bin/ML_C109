# Mô tả các thuật toán học máy – Bài thực hành 2

## 1. Mục tiêu bài toán

Bài toán thuộc nhóm **học có giám sát (Supervised Learning)** và cụ thể là **hồi quy (Regression)**. Mô hình học từ dữ liệu bán hàng trong quá khứ để ước lượng doanh số (`sales`) của một nhóm sản phẩm tại một cửa hàng vào một ngày nhất định.

- **Đầu vào (Input):** mã cửa hàng, nhóm sản phẩm, số mặt hàng đang khuyến mãi, thành phố, bang/tỉnh, loại cửa hàng, cụm cửa hàng, giá dầu, trạng thái ngày lễ và các đặc trưng lịch.
- **Đầu ra (Output):** một giá trị số biểu thị doanh số dự đoán.
- **Mục đích (Purpose):** hỗ trợ ước lượng nhu cầu bán hàng và so sánh hiệu quả của nhiều thuật toán hồi quy.

Đây là dự đoán doanh số theo dữ liệu tổng hợp ngày – cửa hàng – nhóm sản phẩm, không phải dự đoán hành vi của từng khách hàng.

## 2. Tiền xử lý dữ liệu và tạo đặc trưng

Trước khi huấn luyện, chương trình thực hiện:

1. Đọc dữ liệu doanh số, thông tin cửa hàng, ngày lễ/sự kiện và giá dầu.
2. Kết hợp dữ liệu theo mã cửa hàng (`store_nbr`) và ngày (`date`).
3. Tạo các đặc trưng thời gian: năm, tháng, ngày, thứ trong tuần, tuần trong năm và trạng thái cuối tuần.
4. Chuyển doanh số thành kiểu số; chỉ giữ các dòng có doanh số hợp lệ và không âm.
5. Chia đặc trưng thành hai nhóm:
   - **Dữ liệu số:** thay giá trị thiếu bằng trung vị (`median`), sau đó chuẩn hóa bằng `StandardScaler`.
   - **Dữ liệu phân loại:** thay giá trị thiếu bằng giá trị xuất hiện nhiều nhất (`most_frequent`), sau đó mã hóa One-Hot (`OneHotEncoder`). Giá trị phân loại chưa gặp khi huấn luyện được bỏ qua thay vì gây lỗi.

Toàn bộ bước tiền xử lý được đóng gói trong `Pipeline` và `ColumnTransformer), sau đó ghép với từng mô hình để bảo đảm các phép biến đổi được học từ tập huấn luyện.

## 3. Các thuật toán được sử dụng

### 3.1. Hồi quy tuyến tính (Linear Regression)

**Khái niệm:** Hồi quy tuyến tính mô hình hóa mối quan hệ giữa các đặc trưng đầu vào và doanh số bằng một hàm tuyến tính.

**Cách hoạt động:**
1. Nhận các đặc trưng đã được tiền xử lý.
2. Tìm hệ số cho từng đặc trưng và hệ số chặn.
3. Chọn các hệ số sao cho tổng bình phương sai số giữa giá trị thực tế và giá trị dự đoán nhỏ nhất.
4. Dùng phương trình đã học để dự đoán doanh số cho dữ liệu mới.

**Đặc điểm:**
- Dễ hiểu, dễ triển khai và phù hợp làm mô hình cơ sở để so sánh.
- Huấn luyện tương đối nhanh.
- Khó mô tả những mối quan hệ phi tuyến hoặc tương tác phức tạp nếu chưa tạo thêm đặc trưng phù hợp.

### 3.2. Cây quyết định hồi quy (Decision Tree Regressor)

**Khái niệm:** Cây quyết định chia dữ liệu thành các nhóm nhỏ dựa trên điều kiện của đặc trưng. Với bài toán hồi quy, mỗi vùng lá đưa ra một giá trị dự đoán, thường là trung bình doanh số của các mẫu huấn luyện trong vùng đó.

**Cách hoạt động:**
1. Xem xét các đặc trưng và ngưỡng chia có thể có.
2. Chọn phép chia làm giảm sai số trong các nhóm con.
3. Tiếp tục chia các nút con cho đến khi đạt điều kiện dừng, chẳng hạn độ sâu tối đa hoặc số mẫu tối thiểu ở nút lá.
4. Khi dự đoán, đi theo các điều kiện từ nút gốc đến nút lá và trả về giá trị của nút lá.

**Tham số trong mã nguồn hiện tại:**
- `max_depth=16`: giới hạn độ sâu tối đa của cây.
- `min_samples_leaf=30`: mỗi nút lá phải có tối thiểu 30 mẫu.
- `random_state=42`: cố định yếu tố ngẫu nhiên để hỗ trợ tái lập kết quả.

**Đặc điểm:**
- Có thể học các mối quan hệ phi tuyến.
- Dễ giải thích bằng các điều kiện phân nhánh.
- Một cây đơn lẻ có thể nhạy với dữ liệu huấn luyện và dễ quá khớp nếu không giới hạn độ phức tạp.

### 3.3. Rừng ngẫu nhiên (Random Forest Regressor)

**Khái niệm:** Rừng ngẫu nhiên là phương pháp tập hợp nhiều cây quyết định. Các cây được huấn luyện trên những mẫu dữ liệu lấy ngẫu nhiên có hoàn lại và xem xét một tập con đặc trưng khi tìm phép chia.

**Cách hoạt động:**
1. Tạo nhiều tập mẫu ngẫu nhiên từ dữ liệu huấn luyện bằng phương pháp lấy mẫu có hoàn lại.
2. Huấn luyện một cây quyết định trên mỗi tập mẫu; tại các nút, chỉ xét một phần đặc trưng được chọn ngẫu nhiên.
3. Mỗi cây đưa ra một dự đoán riêng.
4. Lấy trung bình dự đoán của các cây để tạo kết quả cuối cùng cho bài toán hồi quy.

**Tham số trong mã nguồn hiện tại:**
- `n_estimators=60`: sử dụng 60 cây.
- `max_depth=16`: giới hạn độ sâu mỗi cây.
- `min_samples_leaf=10`: tối thiểu 10 mẫu tại mỗi nút lá.
- `max_features=0.8`: mỗi phép chia xem xét tối đa 80% số đặc trưng.
- `n_jobs=-1`: tận dụng các lõi xử lý có sẵn.
- `random_state=42`: cố định yếu tố ngẫu nhiên.

**Đặc điểm:**
- Kết hợp nhiều cây giúp giảm độ biến động so với một cây quyết định đơn lẻ.
- Có khả năng mô hình hóa quan hệ phi tuyến và tương tác giữa các đặc trưng.
- Tốn bộ nhớ và thời gian hơn một cây đơn, nhất là khi dữ liệu lớn.

### 3.4. Tăng cường Gradient (Gradient Boosting Regressor)

**Khái niệm:** Gradient Boosting xây dựng mô hình theo từng giai đoạn. Mỗi cây mới được thêm vào để cải thiện những sai số còn lại của mô hình hiện tại bằng cách tối ưu hàm mất mát theo hướng gradient.

**Cách hoạt động:**
1. Khởi tạo mô hình bằng một giá trị dự đoán cơ sở.
2. Tính sai số/gradient của hàm mất mát trên dữ liệu huấn luyện.
3. Huấn luyện một cây nhỏ để mô hình hóa hướng điều chỉnh cần thiết.
4. Cập nhật mô hình hiện tại bằng dự đoán của cây mới nhân với tốc độ học.
5. Lặp lại quá trình cho đến khi đạt số cây quy định.
6. Cộng đóng góp của các cây để tạo dự đoán cuối cùng.

**Tham số trong mã nguồn hiện tại:**
- `n_estimators=80`: xây dựng 80 cây theo tuần tự.
- `learning_rate=0.08`: hệ số điều chỉnh mức đóng góp của mỗi cây.
- `max_depth=3`: giới hạn độ sâu của từng cây.
- `min_samples_leaf=30`: tối thiểu 30 mẫu tại nút lá.
- `subsample=0.8`: mỗi giai đoạn sử dụng 80% mẫu huấn luyện.
- `random_state=42`: cố định yếu tố ngẫu nhiên.

**Đặc điểm:**
- Có thể học các quan hệ phi tuyến và cải thiện dần mô hình qua nhiều giai đoạn.
- Các cây được xây dựng tuần tự nên khó huấn luyện song song như Random Forest.
- Tốc độ học và số cây cần được lựa chọn phù hợp; cấu hình chưa phù hợp có thể dẫn đến thiếu khớp hoặc quá khớp.

## 4. So sánh các thuật toán

| Thuật toán | Cơ chế chính | Điểm mạnh | Hạn chế |
|---|---|---|---|
| Hồi quy tuyến tính | Tìm hàm tuyến tính phù hợp dữ liệu | Nhanh, dễ diễn giải | Hạn chế với quan hệ phi tuyến |
| Cây quyết định | Chia dữ liệu thành các vùng theo điều kiện | Dễ trực quan hóa, học phi tuyến | Có thể nhạy với dữ liệu |
| Rừng ngẫu nhiên | Trung bình kết quả của nhiều cây độc lập tương đối | Ổn định hơn một cây đơn, học quan hệ phức tạp | Tốn tài nguyên hơn |
| Tăng cường Gradient | Thêm cây tuần tự để giảm sai số | Có thể đạt độ chính xác tốt khi tinh chỉnh phù hợp | Huấn luyện tuần tự, nhạy với tham số |

Không có thuật toán nào luôn tốt nhất cho mọi bộ dữ liệu. Mô hình phù hợp được lựa chọn bằng kết quả đánh giá trên tập kiểm định của chính bài toán này.

## 5. Chia dữ liệu và đánh giá

Chương trình chia dữ liệu theo **thứ tự thời gian**: các ngày ở giai đoạn trước dùng để huấn luyện, giai đoạn ngày về sau dùng để kiểm định. Ranh giới chia được xác định trên các ngày duy nhất để một ngày lịch không bị tách sang cả hai tập.

Các chỉ số đánh giá:

- **MAE (Mean Absolute Error – Sai số tuyệt đối trung bình):** trung bình trị tuyệt đối chênh lệch giữa doanh số thực tế và dự đoán. Càng thấp càng tốt.
- **MSE (Mean Squared Error – Sai số bình phương trung bình):** trung bình bình phương sai số; phạt sai số lớn mạnh hơn MAE. Càng thấp càng tốt.
- **RMSE (Root Mean Squared Error – Căn bậc hai sai số bình phương trung bình):** căn bậc hai của MSE, có cùng đơn vị với doanh số. Càng thấp càng tốt.

Mã nguồn sắp xếp bảng so sánh theo RMSE tăng dần và chọn mô hình đứng đầu làm mô hình tốt nhất. Việc chọn mô hình chỉ dựa trên tập kiểm định, không sử dụng tập kiểm tra cuối cùng để lựa chọn.

## 6. Giới hạn và lưu ý khi diễn giải

- Bộ dữ liệu Favorita không cung cấp nhân khẩu học khách hàng, đánh giá/xếp hạng, giá trị vòng đời khách hàng hoặc chi phí quảng cáo; dự án không giả lập các trường này.
- Quy trình hiện tại không đưa doanh số trễ (lag) hoặc trung bình trượt vào đặc trưng.
- Kết quả đánh giá phản ánh dữ liệu, cách chia thời gian và cấu hình tham số đang dùng; không bảo đảm chất lượng tương tự trên mọi giai đoạn hoặc cửa hàng.
- Các tham số trong tài liệu này được đối chiếu theo mã nguồn `scripts/pipeline.py`. Nếu thay đổi cấu hình trong mã, cần cập nhật tài liệu tương ứng.
