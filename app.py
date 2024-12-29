from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
import os
from PIL import Image, ImageDraw, UnidentifiedImageError
import numpy as np
import torch

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Tambahkan CORS untuk memungkinkan komunikasi lintas asal
CORS(app)

# Folder untuk menyimpan gambar sementara
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Memuat model YOLOv5
try:
    model = torch.hub.load('yolov5', 'custom', './static/best.pt', source='local', device='cpu', force_reload=True)
    logging.info("Model YOLOv5 berhasil dimuat.")
except Exception as e:
    logging.error(f"Error memuat model: {e}")
    model = None  # Atur model ke None jika terjadi error

@app.route('/predict', methods=['POST'])
def predict():
    """Proses gambar untuk deteksi objek."""
    if model is None:
        logging.error("Model tidak dimuat. Tidak dapat melakukan prediksi.")
        return jsonify({"error": "Model gagal dimuat. Silakan periksa konfigurasi."}), 500

    # Memeriksa apakah ada file gambar
    if 'image' not in request.files:
        logging.error("Tidak ada file gambar dalam permintaan.")
        return jsonify({"error": "Tidak ada file gambar yang diberikan."}), 400

    try:
        image_file = request.files['image']
        if image_file.filename == '':
            logging.error("Tidak ada file yang dipilih.")
            return jsonify({"error": "Tidak ada file yang dipilih."}), 400

        logging.info(f"File {image_file.filename} diterima.")
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(image_path)

        # Validasi apakah file benar-benar gambar
        try:
            image = Image.open(image_path)
        except UnidentifiedImageError:
            logging.error("File yang diunggah bukan gambar.")
            return jsonify({"error": "File yang diunggah bukan gambar."}), 400

        # Konversi gambar ke numpy array
        img_array = np.array(image)

        # Inferensi model
        logging.info("Melakukan inferensi...")
        results = model(img_array)
        predictions = results.pandas().xyxy[0].to_dict(orient="records")

        if not predictions:
            logging.info("Tidak ada objek yang terdeteksi.")
            return jsonify({"error": "Tidak ada objek yang terdeteksi dalam gambar."}), 200

        # Menambahkan bounding box pada gambar
        draw = ImageDraw.Draw(image)
        for pred in predictions:
            x1, y1, x2, y2 = pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax']
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), f"{pred['name']} {pred['confidence']:.2f}", fill="red")

        # Simpan gambar hasil
        result_image_name = "result_" + os.path.basename(image_file.filename)
        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], result_image_name)
        image.save(result_image_path)
        logging.info(f"Saving result image at: {result_image_path}")

        # URL untuk menampilkan gambar hasil
        image_url = f"http://127.0.0.1:5000/uploads/{result_image_name}"
        logging.info("Prediksi selesai dan gambar disimpan.")

        # Kirim respons JSON ke React
        return jsonify({
            "image_url": image_url,
            "predictions": predictions
        }), 200

    except Exception as e:
        logging.error(f"Error saat memproses gambar: {e}")
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Endpoint untuk menyajikan file hasil deteksi."""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return "File not found", 404
    logging.info(f"Serving file from: {file_path}")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Jalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)