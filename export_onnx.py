"""Chạy script này trên máy tính (hoặc Google Colab) để chuyển best_final.pt sang best_final.onnx.
Lệnh chạy: python export_onnx.py
"""
import glob
import os
from ultralytics import YOLO

models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
pt_files = sorted(glob.glob(os.path.join(models_dir, "*.pt")))

if not pt_files:
    print("Không tìm thấy file .pt nào trong thư mục models/! Hãy chép best_final.pt vào thư mục models/ trước.")
else:
    pt_path = pt_files[0]
    print("Đang xuất mô hình sang ONNX:", pt_path)
    model = YOLO(pt_path)
    onnx_path = model.export(format="onnx", imgsz=640, simplify=True)
    print("Hoàn tất! File ONNX đã được tạo tại:", onnx_path)
    print("Bây giờ bạn có thể xóa file .pt (để nhẹ dung lượng GitHub) và push lên Koyeb/Render!")
