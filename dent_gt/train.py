import torch
import torch.nn as nn
import cv2
import numpy as np
import os

# --- UNet model used during training ---
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv_down1 = self.double_conv(3, 64)
        self.conv_down2 = self.double_conv(64, 128)
        self.conv_down3 = self.double_conv(128, 256)
        self.conv_down4 = self.double_conv(256, 512)
        self.up_trans1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = self.double_conv(512, 256)
        self.up_trans2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = self.double_conv(256, 128)
        self.up_trans3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = self.double_conv(128, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, image):
        x1 = self.conv_down1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.conv_down2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.conv_down3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.conv_down4(x6)
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

# --- Preprocess input image ---
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).float().unsqueeze(0)
    return img

# --- Prediction and visualization ---
def predict_and_show(image_path, model, device):
    model.eval()
    img_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        output = torch.sigmoid(output)
        mask = output.squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255

    # Load and resize original image
    orig = cv2.imread(image_path)
    orig = cv2.resize(orig, (512, 512))

    # Overlay mask in red
    result = orig.copy()
    result[mask == 255] = [0, 0, 255]

    # Show results
    cv2.imshow("Original", orig)
    cv2.imshow("Caries Mask", mask)
    cv2.imshow("Detected Caries (Red)", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model and load weights (state_dict)
    model = UNet().to(device)
    model.load_state_dict(torch.load("unet_caries_detection.pth", map_location=device))

    # Ask user for test image path
    image_path = input("Enter the name of the test image (e.g., tooth.jpg): ")
    if not os.path.exists(image_path):
        print("Error: File not found!")
    else:
        predict_and_show(image_path, model, device)