import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

class FaceKeypointDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.keypoint_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.keypoint_cals = ['lex', 'ley', 'rex', 'rey', 'nox', 'noy']

    def __len__(self):
        return len(self.keypoint_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.keypoint_frame.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        original_width, original_height = image.size
        keypoints = self.keypoint_frame.iloc[idx, 1:].values
        keypoints = keypoints.astype('float').reshape(-1, 2)
        
        keypoints[:, 0] = keypoints[:, 0] / original_width
        keypoints[:, 1] = keypoints[:, 1] / original_height
        
        if self.transform:
            image = self.transform(image)

        keypoints = torch.from_numpy(keypoints).flatten().float()

        return image, keypoints, original_width, original_height, img_name
    
class KeypointModel(nn.Module):
    def __init__(self, num_keypoints):
        super(KeypointModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_keypoints * 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_csv_file = "data/part4/annoExpr/trainLabel.csv"
test_csv_file = "data/part4/annoExpr/testLabel.csv"
train_directory = "data/part4/trainImages/"
test_directory = "data/part4/testImages/"

train_dataset = FaceKeypointDataset(
    csv_file=train_csv_file,
    root_dir=train_directory,
    transform=image_transform
)

test_dataset = FaceKeypointDataset(
    csv_file=test_csv_file,
    root_dir=test_directory,
    transform=image_transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,  num_workers=8)

num_keypoints = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    model = KeypointModel(num_keypoints=num_keypoints).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50
    print(f"Training on {device}")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, keypoints, _, _, _ in train_loop:
            images, keypoints = images.to(device), keypoints.to(device)
            outputs = model(images)
            loss = criterion(outputs, keypoints)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            train_loop.set_postfix(loss=loss.item())
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    print("Training complete!")
    torch.save(model.state_dict(), "keypoint_model.pth")
    print("Model saved!")
    model.eval()
    test_mse = 0.0
    print("Testing the model...")
    with torch.no_grad():
        test_loop = tqdm(test_loader, desc="[Test]")
        for images, keypoints, _, _, _ in test_loop:
            images, keypoints = images.to(device), keypoints.to(device)
            outputs = model(images)
            mse = torch.abs(outputs - keypoints).mean()
            test_mse += mse.item() * images.size(0)
            test_loop.set_postfix(mse=mse.item())
    test_mse /= len(test_dataset)
    print(f"Test MSE: {test_mse:.4f}")
    
    output_dir = "./predictions/"
    os.makedirs(output_dir, exist_ok=True)
    print("Visualizing predictions...")
    model.eval()
    with torch.no_grad():
        viz_loop = tqdm(test_loader, desc="[Visualize]")
        for images, true_keypoints, original_width, original_height, img_names in viz_loop:
            images = images.to(device)
            predicted_keypoints = model(images).cpu().numpy()
            true_keypoints = true_keypoints.cpu().numpy()
            
        for i in range(images.size(0)):
            img_name = img_names[i]
            original_width = original_width[i].item()
            original_height = original_height[i].item()
            true_kp = true_keypoints[i].reshape(-1, 2)
            pred_kp = predicted_keypoints[i].reshape(-1, 2)
            original_image_path = os.path.join(test_directory, os.path.basename(img_name))
            image_to_draw = cv2.imread(original_image_path)
            true_kp[:, 0] = true_kp[:, 0] * original_width
            true_kp[:, 1] = true_kp[:, 1] * original_height
            pred_kp[:, 0] = pred_kp[:, 0] * original_width
            pred_kp[:, 1] = pred_kp[:, 1] * original_height
            for kp in true_kp:
                cv2.circle(image_to_draw, (int(kp[0]), int(kp[1])), 5, (0, 0, 255), -1)
            for kp in pred_kp:
                cv2.circle(image_to_draw, (int(kp[0]), int(kp[1])), 5, (0, 255, 255), -1)
            output_path = os.path.join(output_dir, os.path.basename(img_name))
            cv2.imwrite(output_path, image_to_draw)
    print("Predictions saved!")

if __name__ == "__main__":
    main()
