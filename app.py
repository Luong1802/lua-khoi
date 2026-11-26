"""FireWatch - Giám sát lửa và khói trên nền tảng Đám mây (Tối ưu ONNX Runtime < 200MB RAM).

- Trên Cloud (Koyeb / Render 512MB RAM): Đặt file best_final.onnx (hoặc bất kỳ file .onnx nào) vào thư mục models/.
- Chạy local: Hỗ trợ cả file .onnx lẫn .pt.
"""
import base64
import glob
import os
import tempfile
import threading
import time
import uuid

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMGSZ = int(os.environ.get("IMGSZ", 640))
MAX_STREAM_WIDTH = 960
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "firewatch_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

COLORS = {"fire": (31, 90, 255), "smoke": (255, 179, 143)}
CLASS_NAMES = {0: "fire", 1: "smoke"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

_session = None
_backend_type = None  # "onnx" | "pt"
_model_path = None
_lock = threading.Lock()
jobs = {}


def find_model():
    p = os.environ.get("MODEL_PATH")
    if p and os.path.isfile(p):
        return p
    onnx_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.onnx")))
    if onnx_files:
        return onnx_files[0]
    pt_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pt")))
    return pt_files[0] if pt_files else None


def get_model():
    global _session, _backend_type, _model_path
    if _session is None:
        _model_path = find_model()
        if not _model_path:
            return None
        if _model_path.lower().endswith(".onnx"):
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            _session = ort.InferenceSession(_model_path, sess_options=opts, providers=["CPUExecutionProvider"])
            _backend_type = "onnx"
        else:
            from ultralytics import YOLO
            _session = YOLO(_model_path)
            _backend_type = "pt"
    return _session


def no_model_error():
    return jsonify(error="Chưa có model. Hãy bỏ file best_final.onnx (hoặc .pt) vào thư mục models/ rồi tải lại trang."), 503


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # (h, w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def infer_onnx(session, frame, conf_thres=0.25, iou_thres=0.45):
    orig_h, orig_w = frame.shape[:2]
    img, ratio, (dw, dh) = letterbox(frame, (IMGSZ, IMGSZ))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blob = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})[0]  # YOLOv8 ONNX output: [1, 4+nc, 8400]
    preds = np.squeeze(outputs, axis=0).T  # [8400, 4+nc]

    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    mask = confidences >= conf_thres
    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes_xywh) == 0:
        return []

    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    w = boxes_xywh[:, 2]
    h = boxes_xywh[:, 3]
    nms_boxes = np.stack([x1, y1, w, h], axis=1).tolist()

    indices = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), conf_thres, iou_thres)
    dets = []
    if len(indices) > 0:
        for idx in np.array(indices).flatten():
            bx, by, bw, bh = nms_boxes[idx]
            x_min = max(0.0, min(orig_w, (bx - dw) / ratio))
            y_min = max(0.0, min(orig_h, (by - dh) / ratio))
            x_max = max(0.0, min(orig_w, (bx + bw - dw) / ratio))
            y_max = max(0.0, min(orig_h, (by + bh - dh) / ratio))
            cls_id = int(class_ids[idx])
            dets.append({
                "label": CLASS_NAMES.get(cls_id, str(cls_id)),
                "conf": round(float(confidences[idx]), 3),
                "box": [round(x_min, 1), round(y_min, 1), round(x_max, 1), round(y_max, 1)]
            })
    return dets


def draw_boxes(frame, dets):
    """Vẽ khung kiểu 'khóa mục tiêu': tô mờ, 4 góc đậm, nhãn có nền."""
    out = frame.copy()
    h, w = out.shape[:2]
    th = max(2, w // 300)
    ov = out.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(ov, (x1, y1), (x2, y2), COLORS.get(d["label"], (0, 255, 0)), -1)
    out = cv2.addWeighted(ov, 0.2, out, 0.8, 0)
    scale = max(0.5, w / 1400)
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box"])
        c = COLORS.get(d["label"], (0, 255, 0))
        L = max(16, int(min(x2 - x1, y2 - y1) * 0.25))
        for (x, y, a, b) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(out, (x, y), (x + a * L, y), c, th + 1)
            cv2.line(out, (x, y), (x, y + b * L), c, th + 1)
        txt = f"{d['label'].upper()} {int(d['conf'] * 100)}%"
        (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, scale, 1)
        ty = max(tht + 10, y1)
        cv2.rectangle(out, (x1, ty - tht - 10), (x1 + tw + 12, ty), c, -1)
        cv2.putText(out, txt, (x1 + 6, ty - 6), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def detect(frame, conf):
    model = get_model()
    with _lock:
        if _backend_type == "onnx":
            dets = infer_onnx(model, frame, conf_thres=conf)
        else:
            res = model.predict(frame, conf=conf, imgsz=IMGSZ, verbose=False)[0]
            dets = [{"label": res.names[int(b.cls)], "conf": round(float(b.conf), 3),
                     "box": [round(v, 1) for v in b.xyxy[0].tolist()]} for b in res.boxes]
    return draw_boxes(frame, dets), dets


def summarize(dets):
    def best(k):
        c = [d["conf"] for d in dets if d["label"] == k]
        return round(max(c), 3) if c else 0
    return {"fire": sum(d["label"] == "fire" for d in dets),
            "smoke": sum(d["label"] == "smoke" for d in dets),
            "fire_conf": best("fire"), "smoke_conf": best("smoke")}


def get_conf():
    try:
        return min(max(float(request.form.get("conf", 0.25)), 0.01), 0.99)
    except ValueError:
        return 0.25


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/info")
def info():
    p = find_model()
    runtime = "ONNX Runtime (Cloud CPU)" if (p and p.endswith(".onnx")) else "CPU"
    return jsonify(model=os.path.basename(p) if p else None, device=runtime)


@app.post("/api/detect/image")
def detect_image():
    if get_model() is None:
        return no_model_error()
    f = request.files.get("file")
    if not f:
        return jsonify(error="Thiếu tệp ảnh."), 400
    arr = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="Không đọc được ảnh."), 400
    t0 = time.perf_counter()
    plotted, dets = detect(img, get_conf())
    ms = round((time.perf_counter() - t0) * 1000)
    _, buf = cv2.imencode(".jpg", plotted, [cv2.IMWRITE_JPEG_QUALITY, 88])
    b64 = base64.b64encode(buf).decode("ascii")
    return jsonify(image="data:image/jpeg;base64," + b64, detections=dets,
                   summary=summarize(dets), time_ms=ms)


@app.post("/api/video/upload")
def video_upload():
    if get_model() is None:
        return no_model_error()
    f = request.files.get("file")
    if not f:
        return jsonify(error="Thiếu tệp video."), 400
    jid = uuid.uuid4().hex[:12]
    ext = os.path.splitext(f.filename or "")[1] or ".mp4"
    path = os.path.join(UPLOAD_DIR, jid + ext)
    f.save(path)
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    jobs[jid] = {"path": path, "conf": get_conf(), "frame": 0, "total": total,
                 "fire_frames": 0, "smoke_frames": 0,
                 "current": {"fire": 0, "smoke": 0, "fire_conf": 0, "smoke_conf": 0},
                 "fps": 0, "done": False, "error": None}
    return jsonify(job_id=jid, total=total)


def video_frames(jid):
    job = jobs[jid]
    cap = cv2.VideoCapture(job["path"])
    last = time.time()
    skip = int(os.environ.get("SKIP_FRAMES", 3))
    last_dets = []
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > MAX_STREAM_WIDTH:
                frame = cv2.resize(frame, (MAX_STREAM_WIDTH, int(h * MAX_STREAM_WIDTH / w)))
            if job["frame"] % skip == 0:
                plotted, last_dets = detect(frame, job["conf"])
            else:
                plotted = draw_boxes(frame, last_dets)
            s = summarize(last_dets)
            job["frame"] += 1
            job["fire_frames"] += s["fire"] > 0
            job["smoke_frames"] += s["smoke"] > 0
            job["current"] = s
            now = time.time()
            job["fps"] = round(1 / max(now - last, 1e-3), 1)
            last = now
            _, buf = cv2.imencode(".jpg", plotted, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
    finally:
        cap.release()
        job["done"] = True
        try:
            os.remove(job["path"])
        except OSError:
            pass


@app.get("/api/video/stream/<job_id>")
def video_stream(job_id):
    if job_id not in jobs or jobs[job_id]["done"]:
        return "Not found", 404
    return Response(video_frames(job_id), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/video/status/<job_id>")
def video_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify(error="not found"), 404
    return jsonify({k: v for k, v in job.items() if k != "path"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print("Model:", find_model() or "CHƯA CÓ - hãy bỏ file best_final.onnx vào thư mục models/")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
