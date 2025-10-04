import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64

# --- U-Net model definition for dental caries detection ---
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder (Downsampling)
        self.conv_down1 = self.double_conv(3, 64)
        self.conv_down2 = self.double_conv(64, 128)
        self.conv_down3 = self.double_conv(128, 256)
        self.conv_down4 = self.double_conv(256, 512)

        # Decoder (Upsampling)
        self.up_trans1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = self.double_conv(512, 256)
        self.up_trans2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = self.double_conv(256, 128)
        self.up_trans3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = self.double_conv(128, 64)

        # Output layer
        self.out = nn.Conv2d(64, 1, kernel_size=1)

        # Max pooling
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def double_conv(self, in_channels, out_channels):
        """ Two consecutive convolutional layers with BatchNorm and ReLU """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, image):
        """ Forward pass through U-Net """
        # Downsampling path
        x1 = self.conv_down1(image)
        x2 = self.max_pool_2x2(x1)

        x3 = self.conv_down2(x2)
        x4 = self.max_pool_2x2(x3)

        x5 = self.conv_down3(x4)
        x6 = self.max_pool_2x2(x5)

        x7 = self.conv_down4(x6)

        # Upsampling path
        x8 = self.up_trans1(x7)
        x9 = torch.cat([x8, x5], dim=1)
        x10 = self.conv_up1(x9)

        x11 = self.up_trans2(x10)
        x12 = torch.cat([x11, x3], dim=1)
        x13 = self.conv_up2(x12)

        x14 = self.up_trans3(x13)
        x15 = torch.cat([x14, x1], dim=1)
        x16 = self.conv_up3(x15)

        output = self.out(x16)
        return output


# Initialize Flask app
app = Flask(__name__)

# Load model and device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    model = UNet().to(device)
    model.load_state_dict(torch.load("unet_caries_detection.pth", map_location=device))
    model.eval()
    print("Model successfully loaded.")
except FileNotFoundError:
    print("Error: 'unet_caries_detection.pth' file not found. Please place it in the project root directory.")
    model = None
except Exception as e:
    print(f"Error while loading model: {e}")
    model = None


def preprocess_image(img_stream):
    """ Preprocess input image from uploaded file stream """
    nparr = np.frombuffer(img_stream.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    original_height, original_width = img.shape[:2]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).float().unsqueeze(0)

    return img, original_width, original_height


# -------------------- Flask Routes --------------------

@app.route('/')
def home():
    """ Render homepage (index.html) """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """ Handle prediction requests (called from frontend JS via POST) """
    if model is None:
        return jsonify(success=False, error="Model could not be loaded. Check server logs."), 500

    if 'image' not in request.files:
        return jsonify(success=False, error="No image file uploaded."), 400

    file = request.files['image']

    try:
        # Preprocess image
        img_tensor, original_width, original_height = preprocess_image(file.stream)
        img_tensor = img_tensor.to(device)

        # Run model inference
        with torch.no_grad():
            output = model(img_tensor)
            output = torch.sigmoid(output)
            # Lower threshold (0.3) to detect more possible caries regions
            mask = (output.squeeze().cpu().numpy() > 0.3).astype(np.uint8) * 255

        # Reload original file
        file.seek(0)
        orig_nparr = np.frombuffer(file.read(), np.uint8)
        orig = cv2.imdecode(orig_nparr, cv2.IMREAD_COLOR)
        orig = cv2.resize(orig, (512, 512))

        # Overlay caries mask in red
        result_img = orig.copy()
        result_img[mask == 255] = [0, 0, 255]

        # Encode result image to Base64
        _, encoded_img = cv2.imencode('.png', result_img)
        base64_encoded_image = base64.b64encode(encoded_img).decode('utf-8')

        # Prepare response messages
        if np.sum(mask) > 0:
            predictions = ["According to the analysis, dental caries were detected. Please consult your dentist."]
        else:
            predictions = ["No dental caries detected in the analysis. Keep up with your regular dental check-ups."]

        return jsonify(
            success=True,
            image_data=base64_encoded_image,
            predictions=predictions
        )

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify(success=False, error="An error occurred during prediction."), 500


if __name__ == '__main__':
    app.run(debug=True)
