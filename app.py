import os
import io
import base64
import urllib.request
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

# ── Cargar modelo YOLOv8 una sola vez al iniciar ──
MODEL = None
MODEL_PATH = Path("goat_model.pt")

# URL del modelo público de Roboflow Universe (Goat Looker - Brookside Research)
# Descargado directamente como archivo .pt de YOLOv8
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL   = os.environ.get("ROBOFLOW_MODEL", "goat-looker/1")
ROBOFLOW_WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "brookside-research")

def download_model():
    """Descarga el modelo .pt desde Roboflow si no existe localmente."""
    if MODEL_PATH.exists():
        print(f"✅ Modelo encontrado en disco: {MODEL_PATH}")
        return True

    if not ROBOFLOW_API_KEY:
        print("⚠️  Sin API key — usando YOLOv8n genérico (detecta sheep/goat approx)")
        return False

    try:
        print("📥 Descargando modelo desde Roboflow...")
        from roboflow import Roboflow
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_MODEL.split("/")[0])
        version  = project.version(int(ROBOFLOW_MODEL.split("/")[1]))
        version.download("yolov8", location="./")

        # Roboflow descarga en subcarpeta — mover al directorio raíz
        import glob, shutil
        pts = glob.glob("./**/best.pt", recursive=True)
        if pts:
            shutil.move(pts[0], str(MODEL_PATH))
            print(f"✅ Modelo descargado: {MODEL_PATH}")
            return True
    except Exception as e:
        print(f"❌ Error descargando modelo Roboflow: {e}")

    return False

def load_model():
    global MODEL
    from ultralytics import YOLO

    has_custom = download_model()

    if has_custom and MODEL_PATH.exists():
        print("🧠 Cargando modelo YOLOv8 de cabras (Roboflow)...")
        MODEL = YOLO(str(MODEL_PATH))
    else:
        # Fallback: YOLOv8n preentrenado en COCO (tiene clase 'sheep' como aproximación)
        print("🧠 Cargando YOLOv8n COCO (fallback)...")
        MODEL = YOLO("yolov8n.pt")

    print("✅ Modelo listo")

# Cargar al iniciar
load_model()

# ── ENDPOINTS ──

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    """
    Recibe imagen en base64 o como archivo,
    devuelve detecciones en JSON.
    """
    try:
        # Obtener imagen
        if request.content_type and "application/json" in request.content_type:
            data   = request.get_json()
            b64    = data.get("image", "").split(",")[-1]  # quitar prefijo data:image/...
            img_bytes = base64.b64decode(b64)
        else:
            # multipart/form-data
            f = request.files.get("image")
            if not f:
                return jsonify({"error": "No image provided"}), 400
            img_bytes = f.read()

        # Decodificar imagen
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)

        # Obtener confianza del query param (default 0.35)
        conf = float(request.args.get("confidence", 0.35))
        conf = max(0.1, min(0.95, conf))

        # Inferencia
        results = MODEL.predict(
            source=img_np,
            conf=conf,
            iou=0.45,
            verbose=False
        )[0]

        # Determinar si es modelo de cabras (custom) o COCO (fallback)
        is_custom = MODEL_PATH.exists()
        # En COCO: clase 19 = "sheep" (la más cercana a cabra)
        COCO_GOAT_CLASSES = {19, 20}  # sheep, cow como fallback

        predictions = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = MODEL.names[cls_id]

            # Filtro: si es modelo custom acepta todo,
            # si es COCO solo acepta sheep/cow
            if not is_custom and cls_id not in COCO_GOAT_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            predictions.append({
                "x":          (x1 + x2) / 2,
                "y":          (y1 + y2) / 2,
                "width":      x2 - x1,
                "height":     y2 - y1,
                "confidence": float(box.conf[0]),
                "class":      cls_name,
            })

        return jsonify({
            "predictions": predictions,
            "image": {"width": img.width, "height": img.height},
            "model": "custom-goat" if is_custom else "coco-fallback"
        })

    except Exception as e:
        print(f"Error en /detect: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_type": "custom-goat" if MODEL_PATH.exists() else "coco-fallback"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
