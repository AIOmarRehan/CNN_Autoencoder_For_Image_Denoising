import numpy as np
import io
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("unet_model.h5", compile = False)

app = Flask(__name__)
CORS(app)  # allow frontend to fetch

# Preprocess function
def preprocess_image(image, target_size = (192, 176)):
    image = image.resize((target_size[1], target_size[0]))  # width, height
    image = np.array(image) / 255.0
    if image.ndim == 2:
        image = np.expand_dims(image, axis = -1)
    return np.expand_dims(image, axis = 0)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(file.stream).convert("L")  # grayscale

    input_data = preprocess_image(img)

    pred = model.predict(input_data)[0]

    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = np.squeeze(pred, axis = -1)

    pred_img = (pred * 255).astype(np.uint8)
    pred_img = Image.fromarray(pred_img)

    buf = io.BytesIO()
    pred_img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype = "image/png")

if __name__ == "__main__":
    app.run(host = "127.0.0.1", port = 5000, debug = True)