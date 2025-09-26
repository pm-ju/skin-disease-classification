from flask import Flask, request, jsonify, send_from_directory
import torch
from PIL import Image
import io
import os
import logging
from model_loader import load_model, get_lesion_info, preprocess_image

# Correctly configure Flask to find your HTML, CSS, and JS files
app = Flask(__name__, static_folder='../static', template_folder='..')

# Global Variables
MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LESION_INFO = get_lesion_info()
logging.basicConfig(level=logging.INFO)

# --- App Routes ---
@app.route('/')
def index():
    """Serve the main index.html page."""
    return send_from_directory('..', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint."""
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    try:
        image = Image.open(io.BytesIO(file.read()))
        input_tensor = preprocess_image(image, DEVICE)
    except Exception as e:
        logging.error(f"Image preprocessing error: {e}")
        return jsonify({'error': 'Invalid image format'}), 400
    with torch.no_grad():
        outputs = MODEL(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    results = []
    for i, prob in enumerate(probabilities):
        lesion = LESION_INFO[i]
        results.append({
            'name': lesion['name'],
            'description': lesion['description'],
            'probability': float(prob),
            'severity': lesion['severity'],
        })
    results.sort(key=lambda x: x['probability'], reverse=True)
    return jsonify({ 'success': True, 'predictions': results })

# --- Main Execution ---
if __name__ == '__main__':
    model_path = os.path.join('models', 'supcon_rp_RP_30_v3_temp0.1_best_model.pth')
    try:
        MODEL = load_model(model_path, DEVICE)
        logging.info(f"Model loaded successfully on {DEVICE}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        exit(1)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)