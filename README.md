# FireWatch Cloud — Triển khai trên Koyeb / Render (Gói 512MB RAM Miễn phí, Không cần thẻ)

## Bước 1: Chuyển mô hình sang ONNX (giảm RAM từ 900MB xuống ~150MB)
1. Chép file `best_final.pt` vào thư mục `models/`.
2. Chạy lệnh:
   ```bash
   pip install ultralytics onnx slim onnxruntime
   python export_onnx.py
   ```
3. Sau khi thấy file `models/best_final.onnx` xuất hiện, **xóa file `best_final.pt`** khỏi thư mục `models/` (chỉ giữ lại `best_final.onnx`).

## Bước 2: Đẩy lên GitHub
Tạo một repository mới trên GitHub và push toàn bộ thư mục này lên.

## Bước 3: Deploy lên Koyeb (0 đồng, không cần thẻ Visa)
1. Đăng nhập [app.koyeb.com](https://app.koyeb.com) bằng tài khoản **GitHub**.
2. Chọn **Create Web Service** → **GitHub** → chọn repository vừa tạo.
3. Ở mục **Builder**, chọn **Dockerfile**.
4. Ở mục **Instance**, giữ nguyên gói **Free (512MB RAM · 0.1 vCPU)**.
5. Ở mục **Exposed ports**, đặt cổng **`8000`**.
6. Nhấn **Deploy** và chờ ~2 phút để nhận link `.koyeb.app` công khai!
