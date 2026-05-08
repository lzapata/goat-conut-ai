import os
import io
import base64
import requests as req

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
MODEL_ID         = os.environ.get("ROBOFLOW_MODEL", "goat-looker/6")
API_URL          = f"https://serverless.roboflow.com/{MODEL_ID}"

print(f"✅ Servidor listo — modelo: {MODEL_ID}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if not ROBOFLOW_API_KEY:
        return jsonify({"error": "Sin API key", "predictions": []}), 503
    try:
        # Obtener imagen
        if request.content_type and "application/json" in request.content_type:
            data      = request.get_json()
            b64       = data.get("image", "").split(",")[-1]
            img_bytes = base64.b64decode(b64)
        else:
            f = request.files.get("image")
            if not f:
                return jsonify({"error": "No image"}), 400
            img_bytes = f.read()

        conf = float(request.args.get("confidence", 0.35))
        conf = max(0.1, min(0.95, conf))

        # Redimensionar para reducir tiempo de respuesta
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((640, 640))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64_resized = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Llamar a Roboflow API con requests
        resp = req.post(
            API_URL,
            params={"api_key": ROBOFLOW_API_KEY, "confidence": int(conf * 100), "overlap": 30},
            data=b64_resized,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15
        )

        if not resp.ok:
            return jsonify({"error": f"Roboflow {resp.status_code}: {resp.text}", "predictions": []}), 502

        data_rf = resp.json()

        predictions = [
            {
                "x":          p["x"],
                "y":          p["y"],
                "width":      p["width"],
                "height":     p["height"],
                "confidence": p["confidence"],
                "class":      p.get("class", "goat"),
            }
            for p in data_rf.get("predictions", [])
            if p["confidence"] >= conf
        ]

        return jsonify({
            "predictions": predictions,
            "image":       {"width": img.width, "height": img.height},
            "model":       MODEL_ID
        })

    except Exception as e:
        print(f"Error /detect: {e}")
        return jsonify({"error": str(e), "predictions": []}), 500

@app.route("/health")
def health():
    return jsonify({
        "status":     "ok",
        "model":      MODEL_ID,
        "model_type": "roboflow-hosted" if ROBOFLOW_API_KEY else "no-key"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
