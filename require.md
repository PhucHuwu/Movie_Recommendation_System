# I. MÔ TẢ BÀI TOÁN

Bạn được giao nhiệm vụ xây dựng một hệ thống recommendation cho một nền tảng Gợi ý phim

Hệ thống cần có khả năng:
1. Thu thập dữ liệu
2. Làm sạch dữ liệu
3. Trực quan hóa dữ liệu
4. Xây dựng mô hình recommendation
5. Hiển thị gợi ý cho người dùng

Mô tả về dataset:
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("parasharmanas/movie-recommendation-system")

print("Path to dataset files:", path)
```
```bash
#!/bin/bash
curl -L -o ~/Downloads/movie-recommendation-system.zip\
  https://www.kaggle.com/api/v1/datasets/download/parasharmanas/movie-recommendation-system
```
movies.csv bao gồm các trường: movieId, title, genres
ratings.csv bao gồm các trường: userId, movieId, rating, timestamp

# II. YÊU CẦU & NHIỆM VỤ CHI TIẾT

1. Thu thập dữ liệu
- Dataset ≥ 2.000 items
- Có ít nhất 5 features mô tả item

2. Làm sạch và chuẩn bị dữ liệu
- Missing values
- Chuẩn hóa dữ liệu
- Loại bỏ duplicate
- Xử lý outlier
- Vector hóa (TF-IDF, BOW, embeddings)
- Split train/test (xử lý dữ liệu khác miền)

3. Phân tích & trực quan hóa dữ liệu
- Phân bố rating
- Tần suất nhóm sản phẩm
- Top items
- Heatmap, bar chart, histogram

4. Xây dựng hệ gợi ý
- Xây dựng 4 model recommendation

5. Đánh giá mô hình
- RMSE, MAE, Precision@K, Recall@K
- Sử dụng ma trận thưa để tối ưu hóa tốc độ tính toán

6. Giao diện hiển thị
- Web interface Streamlit
- Giao diện user: đăng nhập (không bao gồm đăng ký), tìm kiếm phim, gợi ý phim, profile, logout, không bao gồm đánh giá phim, hệ thống phải đảm bảo bảo toàn dataset
- Giao diện admin: mọi user đều có thể truy cập, thống kê dữ liệu, xem trực quan hóa dữ liệu, xem đánh giá mô hình, xem so sánh mô hình