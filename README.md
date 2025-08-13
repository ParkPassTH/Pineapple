# Pineapple Quality Inspection (Flask + YOLO)

Realtime pineapple quality & defect detection with dynamic grade rules:
- Grade A: 0 defects
- Grade B: 1-2 defects
- Grade C: >2 defects

Two run modes:
1. Local development (uses server webcam /video_feed + polling /latest_summary)
2. Deploy mode (DEPLOY=1) – client browser webcam, sends frames to /predict

## Quick Start (Local)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Open http://localhost:5000

## Deploy on Render
1. Push repo to GitHub (optionally store best.pt via LFS or external URL)
2. In Render: New + Web Service -> connect repo
3. Build Command: `pip install -r requirements.txt`
4. Start Command: (Procfile picked automatically) or `gunicorn app:app --timeout 120`
5. Add Environment Variable: `DEPLOY=1`
6. (Optional) Add `MODEL_URL` if downloading model at runtime; code snippet can be added before model load.

## Endpoints (Deploy Mode)
- `/` main UI (client camera)
- `/predict` JSON inference
- `/health` health check

## Endpoints (Local Dev Extra)
- `/video_feed` MJPEG stream
- `/latest_summary` aggregated frame summary
- `/raw_detections` raw per-frame class list
- `/debug_classes` mapping debug

## Notes
- Adjust confidence threshold & grouping logic inside `build_frame_summary` if needed.
- Large model file: prefer external hosting (e.g. Hugging Face) + download.

## License
Internal prototype.
