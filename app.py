# app.py
from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import os
import threading
import re

app = Flask(__name__)
IS_DEPLOY = os.getenv("DEPLOY") == "1"  # set DEPLOY=1 on Render to disable server webcam stream
# ตรวจสอบว่าไฟล์โมเดลมีอยู่จริง
model_path = 'best.pt'
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit()

model = YOLO(model_path)
class_names = model.names
if not class_names:
    print("Error: Could not retrieve class names from model.")
    exit()

# Warmup in deploy mode to avoid first-request timeout (loads weights outside request lifecycle)
if IS_DEPLOY:
    try:
        import numpy as _np
        _dummy = _np.zeros((320,320,3), dtype=_np.uint8)
        model.predict(_dummy, imgsz=320, conf=0.5, verbose=False)
        print("Model warmup complete")
    except Exception as e:
        print("Warmup failed:", e)

TARGET_CLASS_IDS = {0, 1, 2, 3, 4, 5}
np.random.seed(58)
class_colors = {
    cls_id: (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))
    for cls_id in class_names.keys()
}

# ---------------- Class Mapping Helpers -----------------
# Flexible normalization for grade / ripeness / defect labels
# so that variations like "GradeA", "grade a", "A", "grade_a" map to the same logical value.

# Precompile regex for grades like (Grade)? + letter (A-C) optionally with separators
GRADE_PATTERN = re.compile(r'^(?:grade\s*[-_ ]*)?([abc])\d*$')  # allow optional digits like A1

RIPENESS_KEYWORDS = {
    'unripe': 'Unripe',
    'ripe': 'Ripe',
    'overripe': 'Overripe',
    'over-ripe': 'Overripe',
    'over_ripe': 'Overripe',
    'half ripe': 'Half Ripe',
    'half-ripe': 'Half Ripe',
    'half_ripe': 'Half Ripe',
    'medium ripe': 'Medium Ripe',
    'medium-ripe': 'Medium Ripe',
    'medium_ripe': 'Medium Ripe',
    'midripe': 'Medium Ripe'
}

DEFECT_KEYWORDS = ['defect', 'disease', 'bruise', 'rot', 'fungus', 'spot', 'mold', 'scar', 'crack']

def normalize_text(raw: str) -> str:
    return raw.lower().replace('_', ' ').replace('-', ' ').strip()

def map_grade(norm: str):
    compact = norm.replace(' ', '')
    m = GRADE_PATTERN.match(compact)
    if m:
        letter = m.group(1).upper()
        return f"Grade {letter}"
    # If contains word 'grade' plus letter token a/b/c
    if 'grade' in norm:
        for letter in ['a','b','c']:
            if letter in norm.split():  # exact token
                return f"Grade {letter.upper()}"
            # pattern like 'gradea' already handled above; allow 'grade a good'
        # search any letter a/b/c after 'grade'
        for letter in ['a','b','c']:
            if f'grade{letter}' in compact:
                return f"Grade {letter.upper()}"
    # Single letter only
    if norm in ['a','b','c']:
        return f"Grade {norm.upper()}"
    return None

def map_ripeness(norm: str):
    if norm in RIPENESS_KEYWORDS:
        return RIPENESS_KEYWORDS[norm]
    # handle words containing ripe/unripe when not explicitly caught
    if 'unripe' in norm:
        return 'Unripe'
    if 'overripe' in norm or ('ripe' in norm and 'over' in norm):
        return 'Overripe'
    if 'ripe' in norm:
        return 'Ripe'
    return None

def is_defect(norm: str):
    return any(k in norm for k in DEFECT_KEYWORDS)

def categorize_class_name(class_name: str):
    norm = normalize_text(class_name)
    grade = map_grade(norm)
    if grade:
        return ('grade', grade)
    ripeness = map_ripeness(norm)
    if ripeness:
        return ('ripeness', ripeness)
    if is_defect(norm):
        return ('defect', class_name)  # keep original for defect naming
    return ('other', class_name)

# ฟังก์ชันช่วยวาดผลลัพธ์ (จาก realtime_yolo.py)
def draw_results_multi_class(frame, detections, class_names, class_colors, current_frame_counts):
    """Draw raw YOLO detections (without tracker ID to avoid confusing users)."""
    for det in detections:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        conf = float(det.conf[0])
        cls_id = int(det.cls[0])
        class_name = class_names.get(cls_id, f"Unknown_{cls_id}")
        color = class_colors.get(cls_id, (255, 255, 255))
        label = f"{class_name}: {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    y_offset = 30
    cv2.putText(frame, "Current Counts:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    y_offset += 25
    for cls_id, count in current_frame_counts.items():
        class_name = class_names.get(cls_id, f"Unknown_{cls_id}")
        text = f"- {class_name}: {count}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        y_offset += 25
    return frame

# --- 2. ฟังก์ชันสำหรับสตรีมวิดีโอ (Generator) ---
latest_summary_lock = threading.Lock()
latest_summary = {"overview": {"total": 0, "grades": {}, "ripeness": {}, "defects": {}}, "details": []}

def build_frame_summary(detections, class_names):
    """สรุปผลแบบรวมกลุ่มผลไม้ต่อ 'ลูก' โดยไม่ใช้ tracker id แต่ใช้การซ้อนทับ/ระยะใกล้
    Strategy:
      1. ใช้เฉพาะกล่องที่เป็น grade / ripeness / other (ไม่ใช่ defect) เพื่อสร้างกลุ่มผลไม้ (anchor boxes)
      2. รวมกล่องเข้ากลุ่มถ้า IoU > IOU_MERGE_THRESHOLD หรือ centroid อยู่ภายในกลุ่ม
      3. กล่อง defect จะถูกผูกกับกลุ่มที่ใกล้ที่สุด (IoU หรือระยะศูนย์กลาง) ถ้าไม่มีจะไม่เพิ่มจำนวนลูกใหม่
    """
    IOU_MERGE_THRESHOLD = 0.3
    DEFECT_ATTACH_IOU = 0.05
    DEFECT_ATTACH_DIST = 150  # pixels

    def box_iou(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / float(area_a + area_b - inter + 1e-6)

    def center(box):
        x1, y1, x2, y2 = box
        return ( (x1 + x2) / 2.0, (y1 + y2) / 2.0 )

    def dist(c1, c2):
        return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) ** 0.5

    # Collect enriched detection objects
    enriched = []
    for det in detections:
        cls_id = int(det.cls[0])
        conf = float(det.conf[0])
        class_name = class_names.get(cls_id, f"Unknown_{cls_id}")
        box = [int(x) for x in det.xyxy[0]]
        cat_type, normalized = categorize_class_name(class_name)
        area = (box[2]-box[0]) * (box[3]-box[1])
        enriched.append({
            'box': box,
            'conf': conf,
            'class_name': class_name,
            'cat_type': cat_type,
            'normalized': normalized,
            'area': area
        })

    # Sort larger boxes first for anchor grouping
    enriched.sort(key=lambda d: d['area'], reverse=True)

    fruits = []  # list of groups
    for d in enriched:
        if d['cat_type'] == 'defect':
            continue  # handle later
        # find existing group
        best_idx = None
        best_score = 0.0
        for idx, g in enumerate(fruits):
            iou = box_iou(d['box'], g['box'])
            if iou > IOU_MERGE_THRESHOLD and iou > best_score:
                best_idx = idx; best_score = iou
            else:
                # fallback: center inclusion
                cx, cy = center(d['box'])
                gx1, gy1, gx2, gy2 = g['box']
                if gx1 <= cx <= gx2 and gy1 <= cy <= gy2 and 0.01 > best_score:
                    best_idx = idx; best_score = 0.01
        if best_idx is None:
            fruits.append({
                'box': d['box'],
                'grade': None,
                'grade_confidence': 0.0,
                'ripeness': None,
                'ripeness_confidence': 0.0,
                'defects': [],
                'defect_names': []
            })
            best_idx = len(fruits)-1
        g = fruits[best_idx]
        # merge box (union) to stabilize
        gx1, gy1, gx2, gy2 = g['box']; x1,y1,x2,y2 = d['box']
        g['box'] = [min(gx1,x1), min(gy1,y1), max(gx2,x2), max(gy2,y2)]
        if d['cat_type'] == 'grade' and d['conf'] > g['grade_confidence']:
            g['grade'] = d['normalized']; g['grade_confidence'] = d['conf']
        elif d['cat_type'] == 'ripeness' and d['conf'] > g['ripeness_confidence']:
            g['ripeness'] = d['normalized']; g['ripeness_confidence'] = d['conf']

    # Attach defects
    for d in enriched:
        if d['cat_type'] != 'defect':
            continue
        best_idx = None; best_metric = 0.0
        dc = center(d['box'])
        for idx, g in enumerate(fruits):
            iou = box_iou(d['box'], g['box'])
            if iou >= DEFECT_ATTACH_IOU and iou > best_metric:
                best_idx = idx; best_metric = iou
            else:
                gc = center(g['box'])
                distance = dist(dc, gc)
                if distance < DEFECT_ATTACH_DIST and (1.0/(distance+1e-6)) > best_metric:
                    best_idx = idx; best_metric = 1.0/(distance+1e-6)
        if best_idx is not None:
            g = fruits[best_idx]
            if d['normalized'] not in g['defect_names']:
                g['defects'].append({'name': d['normalized'], 'confidence': d['conf']})
                g['defect_names'].append(d['normalized'])

    # --- Rule-based grading by defect count ---
    # Grade A: 0 defects, Grade B: 1-2 defects, Grade C: >2 defects
    for g in fruits:
        defect_count = len(g['defect_names'])
        if defect_count == 0:
            g['grade'] = 'Grade A'
        elif defect_count <= 2:
            g['grade'] = 'Grade B'
        else:
            g['grade'] = 'Grade C'

    # Build overview & details
    overview = {'total': len(fruits), 'grades': {}, 'ripeness': {}, 'defects': {}}
    details = []
    for idx, g in enumerate(fruits, start=1):
        if g['grade']:
            overview['grades'][g['grade']] = overview['grades'].get(g['grade'], 0) + 1
        if g['ripeness']:
            overview['ripeness'][g['ripeness']] = overview['ripeness'].get(g['ripeness'], 0) + 1
        for d in g['defects']:
            overview['defects'][d['name']] = overview['defects'].get(d['name'], 0) + 1
        details.append({
            'id': idx,  # sequential fruit number
            'box': g['box'],
            'grade': g['grade'],
            # remove grade_confidence per user request
            'ripeness': g['ripeness'],
            'ripeness_confidence': round(g['ripeness_confidence'],2) if g['ripeness'] else None,
            'defects': [{'name': d['name'], 'confidence': round(d['confidence'],2)} for d in g['defects']]
        })
    return {'overview': overview, 'details': details}

last_raw_detections = []  # for debug

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n'
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ประมวลผลเฟรมด้วยโมเดล YOLO (dev only uses tracking)
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.5, iou=0.5) if not IS_DEPLOY else model.predict(frame, conf=0.5, iou=0.5, verbose=False)

        current_frame_counts = defaultdict(int)
        detections_to_draw = []

        last_raw_detections.clear()
        if results and len(results):
            for det in results[0].boxes:
                int_cls_id = int(det.cls[0])
                class_name = class_names.get(int_cls_id, f"Unknown_{int_cls_id}")
                conf_val = float(det.conf[0])
                last_raw_detections.append({'cls_id': int_cls_id, 'class': class_name, 'conf': round(conf_val,2)})
                if not TARGET_CLASS_IDS or int_cls_id in TARGET_CLASS_IDS:
                    current_frame_counts[int_cls_id] += 1
                    detections_to_draw.append(det)

        processed_frame = draw_results_multi_class(frame, detections_to_draw, class_names, class_colors, current_frame_counts)

        # บันทึก summary ของเฟรมนี้ (ใช้ detections_to_draw สำหรับการสรุป)
        frame_summary = build_frame_summary(detections_to_draw, class_names)
        with latest_summary_lock:
            global latest_summary
            latest_summary = frame_summary

        # เข้ารหัสเฟรมเป็น JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
    cap.release() # <<< การปล่อยทรัพยากรจะอยู่ตรงนี้

# --- 3. Route สำหรับหน้าเว็บหลัก ---
@app.route('/')
def index():
    # pass flag to template for choosing frontend mode
    return render_template('index.html', is_deploy=IS_DEPLOY)

if not IS_DEPLOY:
    # Local development: provide live video streaming + latest summary polling
    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/latest_summary')
    def latest_summary_route():
        with latest_summary_lock:
            return jsonify(latest_summary)

    @app.route('/raw_detections')
    def raw_detections_route():
        return jsonify({'detections': last_raw_detections})
else:
    # Simple health endpoint for Render
    @app.route('/health')
    def health():
        return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# --- 5. Route สำหรับรับภาพและแสดงผลแบบละเอียด ---
from flask import request, jsonify
import base64
import io
from PIL import Image

def analyze_detections(detections, class_names):
    summary = {
        'total': 0,
        'grades': {},
        'ripeness': {},
        'defects': {},
        'details': []
    }
    for det in detections:
        cls_id = int(det.cls[0])
        conf = float(det.conf[0])
        class_name = class_names.get(cls_id, f"Unknown_{cls_id}")
        cat_type, normalized = categorize_class_name(class_name)
        if cat_type == 'grade':
            summary['grades'][normalized] = summary['grades'].get(normalized, 0) + 1
        elif cat_type == 'ripeness':
            summary['ripeness'][normalized] = summary['ripeness'].get(normalized, 0) + 1
        elif cat_type == 'defect':
            summary['defects'][normalized] = summary['defects'].get(normalized, 0) + 1
        summary['details'].append({
            'class': class_name,
            'category': cat_type,
            'normalized': normalized,
            'confidence': round(conf, 2),
            'box': [int(x) for x in det.xyxy[0]],
            'id': int(det.id[0]) if det.id is not None else None
        })
        summary['total'] += 1
    return summary

@app.route('/debug_classes')
def debug_classes():
    data = []
    for cid, cname in class_names.items():
        cat, norm = categorize_class_name(cname)
        data.append({'id': cid, 'raw': cname, 'category': cat, 'normalized': norm})
    return jsonify({'classes': data})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_b64 = data.get('image')
    if not image_b64:
        return jsonify({'error': 'No image provided'}), 400
    # ตัด prefix data:image/jpeg;base64,...
    if ',' in image_b64:
        image_b64 = image_b64.split(',')[1]
    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({'error': f'Invalid image: {str(e)}'}), 400

    # ประมวลผลด้วย YOLO
    try:
        # ใช้ predict() ทั้งสองโหมด (ลดภาระ track) conf=0.5 ตามผู้ใช้ระบุ
        results = model.predict(frame, conf=0.5, iou=0.5, verbose=False)
    except SystemExit:
        return jsonify({'error': 'Model inference system exit (possibly OOM). Try smaller model.'}), 500
    except Exception as e:
        return jsonify({'error': f'Inference failed: {e}'}), 500

    detections = []
    if results and len(results):
        for det in results[0].boxes:
            int_cls_id = int(det.cls[0])
            if not TARGET_CLASS_IDS or int_cls_id in TARGET_CLASS_IDS:
                detections.append(det)

    summary = analyze_detections(detections, class_names)

    # สร้างข้อมูลภาพรวม
    overview = {
        'total': summary['total'],
        'grades': summary['grades'],
        'ripeness': summary['ripeness'],
        'defects': summary['defects']
    }
    # ส่งข้อมูลกลับ frontend
    return jsonify({
        'overview': overview,
        'details': summary['details']
    })
