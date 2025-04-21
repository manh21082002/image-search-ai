# 🚀 Image Search AI - Hệ thống Tìm kiếm Hình ảnh bằng Trí tuệ Nhân tạo

## 📝 Giới thiệu
**Image Search AI** là một hệ thống tìm kiếm hình ảnh món ăn sử dụng Machine Learning và Deep Learning như **YOLO, R-CNN, FAISS** để tìm kiếm ảnh tương tự từ cơ sở dữ liệu.

🔹 **Chức năng chính**:
- 📷 **Nhận diện vật thể** trong ảnh
- 🚀 **Mã hóa ảnh thành vector đặc trưng** để tìm kiếm nhanh
- 

---

## 📂 Cấu trúc thư mục
```plaintext
```
root/
├── app/
│   ├── config.py                # Cấu hình đường dẫn và tham số hệ thống
│   ├── data/                    # Dữ liệu ảnh
│   ├── models/                  # (tuỳ chọn) lưu model custom
│   ├── services/                # Các service xử lý detection, embedding, faiss
│   ├── static/                  # Ảnh nền, ảnh placeholder
│   ├── templates/              # HTML sử dụng Jinja2
│   └── utils/                   # Hàm hỗ trợ thêm (nếu có)
├── main.py                     # Chạy FastAPI app
├── notebooks/                  # Các notebook huấn luyện thử nghiệm
├── scripts/                    # Script train model
├── requirements.txt            # Thư viện cần cài
├── Dockerfile                  # Đóng gói docker (nếu deploy)
```

### 🟦 **1. Cài đặt môi trường**
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
uvicorn main:app --reload

Truy cập: [http://localhost:8000](http://localhost:8000)
## 🧠 Công nghệ sử dụng
- `FastAPI` + `Jinja2` template
- `YOLOv8` để detect món ăn trong ảnh
- `ResNet50` để mã hoá ảnh thành vector
- `FAISS` để tìm ảnh tương đồng
- `Pillow`, `numpy`, `base64` để xử lý ảnh