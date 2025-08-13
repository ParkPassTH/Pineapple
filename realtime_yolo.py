import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# --- 1. โหลดโมเดลและตั้งค่า ---
# Path ไปยังโมเดล YOLOv11 ของคุณ
model_path = r'C:\De_Project\best.pt' # <--- เปลี่ยน path นี้
model = YOLO(model_path)

# ดึงชื่อคลาสทั้งหมดจากโมเดล
class_names = model.names
if not class_names:
    print("Error: Could not retrieve class names from model.")
    exit()

# กำหนด Class ID ที่คุณต้องการนับ (ถ้าต้องการนับทุกคลาส ให้ comment บรรทัดนี้)
TARGET_CLASS_IDS = {0, 1, 2, 3, 4, 5} # <--- เปลี่ยนเป็นคลาสที่สนใจ หรือ comment ทิ้งเพื่อใช้ทุกคลาส

# สร้างสีสุ่มสำหรับแต่ละคลาส
np.random.seed(58)
class_colors = {
    cls_id: (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))
    for cls_id in class_names.keys()
}


# --- 2. ฟังก์ชันช่วยวาดผลลัพธ์ ---
def draw_results_multi_class(frame, detections, class_names, class_colors, current_frame_counts):
    """
    วาด Bounding Box และ Label รวมถึงจำนวนวัตถุที่นับได้ลงบนเฟรม
    """
    for det in detections:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        conf = det.conf[0]
        cls_id = int(det.cls[0])
        track_id = int(det.id[0]) if det.id is not None else None

        class_name = class_names.get(cls_id, f"Unknown_{cls_id}")
        color = class_colors.get(cls_id, (255, 255, 255))

        label = f"{class_name}: {conf:.2f}"
        if track_id is not None:
            label += f" ID:{track_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # แสดงผลการนับในเฟรมปัจจุบัน
    y_offset = 30
    cv2.putText(frame, "Current Counts:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    y_offset += 25
    for cls_id, count in current_frame_counts.items():
        class_name = class_names.get(cls_id, f"Unknown_{cls_id}")
        text = f"- {class_name}: {count}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        y_offset += 25
    return frame


# --- 3. เริ่มต้น Web Camera และประมวลผล Real-time ---
# ใช้ cv2.VideoCapture(0) สำหรับ Web Camera (0 หมายถึงกล้องหลัก)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting real-time object detection. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ส่งเฟรมเข้าโมเดล YOLOv11 พร้อม Tracking
    # ใช้ persist=True เพื่อให้ tracker จดจำวัตถุข้ามเฟรม
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.4, iou=0.5)

    current_frame_counts = defaultdict(int)
    detections_to_draw = []

    # ประมวลผลผลลัพธ์การตรวจจับ
    if results and results[0].boxes.id is not None:
        for det in results[0].boxes:
            int_cls_id = int(det.cls[0])

            # กรองเฉพาะคลาสที่สนใจ (ถ้าคุณไม่ได้ comment บรรทัด TARGET_CLASS_IDS)
            # ถ้าต้องการนับทุกคลาส ให้ลบเงื่อนไข if นี้ออกไป
            if not TARGET_CLASS_IDS or int_cls_id in TARGET_CLASS_IDS:
                current_frame_counts[int_cls_id] += 1
                detections_to_draw.append(det)

    # วาดผลลัพธ์ลงบนเฟรม
    processed_frame = draw_results_multi_class(
        frame,
        detections_to_draw,
        class_names,
        class_colors,
        current_frame_counts
    )

    # แสดงผลเฟรมที่ประมวลผลแล้ว
    cv2.imshow('Real-time Object Counting', processed_frame)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. ปล่อยทรัพยากร ---
cap.release()
cv2.destroyAllWindows()
print("Program terminated.")
