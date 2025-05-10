from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms

# Flask setup
app = Flask(__name__)
CORS(app)

# Upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Allowed file check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Class names
known_classes = ['Dasheri', 'Kesar', 'Langdo', 'Rajapuri', 'Totapuri']
class_names = known_classes + ['Others']

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights="DEFAULT")
model.fc = torch.nn.Linear(model.fc.in_features, len(known_classes))
model.load_state_dict(torch.load("mango_model_5class.pth", map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return "✅ Mango Classifier API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load and preprocess image
        image = Image.open(filepath).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)[0]
            top_prob, top_class = torch.max(probs, 0)

        threshold = 0.7
        if top_prob.item() < threshold:
            predicted_label = "Others"
        else:
            predicted_label = known_classes[top_class.item()]

        # Return prediction
        return jsonify({
            'class': predicted_label,
            'confidence': round(top_prob.item(), 2),
            'filename': filename
        })
    else:
        return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)
