# 🐐 Goat Conut AI

Aplicación web para **contar cabras automáticamente** usando YOLOv8 y Python Flask.
Desplegada en Render.com — funciona en cualquier dispositivo móvil con cámara.

## Arquitectura

```
📱 Móvil (Safari / Chrome)
    │  frame JPEG cada 350ms
    ▼
🐍 Python Flask (Render.com)
    │  YOLOv8 detecta cabras
    ▼
📱 Dibuja bounding boxes en canvas
```

---

## 🚀 Despliegue paso a paso

### 1. Subir a GitHub

```bash
git init
git add .
git commit -m "🐐 Goat Conut AI inicial"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/goat-conut-ai.git
git push -u origin main
```

### 2. Crear cuenta en Render.com

1. Ve a [render.com](https://render.com) → **Sign up** con GitHub
2. Dashboard → **New +** → **Web Service**
3. Conecta tu repositorio `goat-conut-ai`
4. Render detecta automáticamente el `render.yaml`

### 3. Configurar variables de entorno en Render

En el dashboard de tu servicio → **Environment**:

| Variable | Valor |
|----------|-------|
| `ROBOFLOW_API_KEY` | Tu API key de roboflow.com |
| `ROBOFLOW_MODEL` | `goat-looker/1` |
| `ROBOFLOW_WORKSPACE` | `brookside-research` |

> Obtén tu API key gratis en [roboflow.com](https://roboflow.com) → Settings → API Keys

### 4. Deploy

Render construye y despliega automáticamente.
La URL de tu app será: `https://goat-conut-ai.onrender.com`

---

## ⚠️ Nota sobre el modelo de Roboflow

El modelo **Goat Looker** de Brookside Research requiere que lo copies
a tu workspace de Roboflow antes de descargarlo:

1. Ve a: https://universe.roboflow.com/brookside-research/goat-looker
2. Clic en **Fork** o **Try this model**
3. Se copia a tu workspace
4. Usa tu API key → el servidor lo descargará al iniciar

**Sin modelo custom:** la app usa YOLOv8n (COCO) con clase `sheep`
como aproximación. Funciona pero con menor precisión.

---

## Uso local (desarrollo)

```bash
pip install -r requirements.txt
export ROBOFLOW_API_KEY=rf_tu_key_aqui
python app.py
# Abre http://localhost:5000
```
